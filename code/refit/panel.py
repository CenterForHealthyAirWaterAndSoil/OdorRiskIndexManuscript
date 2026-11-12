"""Zip-day panel assembly: every zip x local day, cell weather, exposure terms and report targets."""
import holidays
import numpy as np
import pandas as pd

import odor_forecast_core as core

COUNT_COLUMNS = ["n_reports", "n_report_hours", "n_reports_value1", "n_reports_excl1", "smell_value_sum",
                 "burden_hourly_mean", "burden_count_weighted", "burden_hourly_mean_excl1",
                 "burden_count_weighted_excl1"]
BURDENS = {"burden_hourly_mean": "event_hourly_mean", "burden_count_weighted": "event_count_weighted",
           "burden_hourly_mean_excl1": "event_hourly_mean_excl1",
           "burden_count_weighted_excl1": "event_count_weighted_excl1"}

# (column, unit, meaning). Output order of the panel; DATA_DICTIONARY.md is written from this list.
COLUMNS = [
    ("city", "", "pgh (Pittsburgh) or lou (Louisville)"),
    ("zipcode", "", "5-digit zip code of the reports"),
    ("date", "YYYY-MM-DD", "local calendar day (America/New_York for pgh, America/Kentucky/Louisville for lou)"),
    ("year", "", "calendar year of date (for city x year intercepts)"),
    ("month", "1-12", "calendar month of date"),
    ("year_month", "YYYY-MM", "calendar month of date (for city x year-month intercepts)"),
    ("dow", "0-6", "day of week, Monday = 0"),
    ("is_holiday", "bool", "US federal holiday (holidays.US, as in the original scripts)"),
    ("sparse_reporting_period", "bool", "Louisville before 2019-03-01, when almost nobody reported; always False for pgh"),
    ("zip_in_original_set", "bool", "zip was in the old training panel (pgh: the old top-61; lou: all 36)"),
    ("zip_lat", "deg", "zip centroid = mean skewed latitude of all of that zip's raw reports"),
    ("zip_lon", "deg", "zip centroid = mean skewed longitude of all of that zip's raw reports"),
    ("zip_n_reports_all", "count", "all raw reports in the zip, any date (the old zip-selection count)"),
    ("cell_id", "", "ERA5 0.25-degree grid cell of the zip centroid (lat_lon); the weather series and a CV group"),
    ("cell_lat", "deg", "grid-cell latitude"),
    ("cell_lon", "deg", "grid-cell longitude"),
    ("cell_elevation_m", "m", "grid-cell mean elevation as reported by Open-Meteo (reference only; weather is not downscaled to the zip)"),
    ("n_hours", "count", "hours in the local day: 23 (spring forward), 24, or 25 (fall back)"),
    ("temperature", "degF", "mean of hourly ERA5 2 m temperature"),
    ("temperature_squared", "degF^2", "temperature squared"),
    ("temp_min", "degF", "minimum hourly ERA5 2 m temperature"),
    ("temp_max", "degF", "maximum hourly ERA5 2 m temperature"),
    ("diurnal_temperature_range", "degF", "temp_max - temp_min over the whole local day"),
    ("relative_humidity", "%", "mean of hourly RH from ERA5 2 m temperature and dew point (Magnus), capped at 100"),
    ("wind_speed", "mph", "mean of hourly ERA5 10 m wind speed (from u/v)"),
    ("wind_direction", "deg from", "speed-weighted vector mean of hourly 10 m wind (as weather_history and the live forecast)"),
    ("wind_direction_circmean", "deg from", "unweighted circular mean of hourly directions (as the original training scripts)"),
    ("precipitation", "inch", "sum of hourly ERA5 rain (Open-Meteo models=era5; excludes snow)"),
    ("solar_radiation", "W/m2", "mean of hourly ERA5 surface solar radiation downwards (J/m2 per hour / 3600)"),
    ("boundary_layer_height", "ft", "mean of hourly boundary-layer height, true feet (Copernicus ERA5 blh x 3.28084)"),
    ("atmospheric_pressure", "hPa", "mean of hourly ERA5 surface pressure (at ERA5's surface for the cell)"),
    ("pressure_anomaly", "hPa", "atmospheric_pressure minus that zip's mean over all panel days"),
    ("nearest_source", "", "emission source nearest the zip centroid"),
    ("dist_nearest_mi", "miles", "haversine distance from zip centroid to nearest_source"),
    ("multi_source_exposure", "0-1", "exp(-0.02 * dist_nearest_mi) (odor_forecast_core.compute_multisource_terms)"),
    ("wind_align_weighted", "0-1", "wind alignment from nearest_source to the zip, using wind_direction (deployment convention)"),
    ("wind_align_weighted_circmean", "0-1", "the same alignment using wind_direction_circmean (original training convention)"),
    ("n_reports", "count", "smell reports in the zip on the local day"),
    ("has_report", "bool", "n_reports > 0"),
    ("n_report_hours", "count", "local wall-clock hours with at least one report"),
    ("n_reports_value1", "count", "reports with smell value 1 ('just fine')"),
    ("n_reports_excl1", "count", "n_reports - n_reports_value1"),
    ("smell_value_sum", "", "sum of smell values"),
    ("smell_value_max", "1-5", "highest smell value (NaN without reports)"),
    ("smell_value_mean_hourly", "1-5", "mean over report hours of each hour's mean smell value (NaN without reports)"),
    ("smell_value_mean_hourly_excl1", "1-5", "the same, leaving out value-1 reports (NaN if none remain)"),
    ("burden_hourly_mean", "", "n_reports x smell_value_mean_hourly (the original target's burden); 0 without reports"),
    ("burden_count_weighted", "", "n_reports x count-weighted mean smell value = smell_value_sum"),
    ("burden_hourly_mean_excl1", "", "burden_hourly_mean leaving out value-1 reports"),
    ("burden_count_weighted_excl1", "", "burden_count_weighted leaving out value-1 reports"),
    ("event_hourly_mean", "0/1", "burden_hourly_mean > the city's mean burden over zip_in_original_set rows"),
    ("event_count_weighted", "0/1", "burden_count_weighted > the same kind of city threshold"),
    ("event_hourly_mean_excl1", "0/1", "burden_hourly_mean_excl1 > its city threshold"),
    ("event_count_weighted_excl1", "0/1", "burden_count_weighted_excl1 > its city threshold"),
]
COLUMN_NAMES = [c for c, _, _ in COLUMNS]


class PanelError(RuntimeError):
    """The panel cannot be built without guessing (missing weather, duplicate keys)."""


def assemble_city_panel(city, zips, cell_daily, targets, sources, dates, sparse_before=None):
    """Every zip x date with its cell's daily weather, exposure terms and report targets (no events yet).

    zips: zipcode, lat, lon, cell_id, in_original_set (optional: n_reports_all, cell_lat, cell_lon, cell_elevation_m)
    cell_daily: cell_id, date and the daily weather columns. Missing weather raises PanelError.
    """
    if cell_daily.duplicated(["cell_id", "date"]).any():
        raise PanelError("cell_daily has duplicate (cell_id, date) rows")
    grid = zips.merge(pd.DataFrame({"date": list(dates)}), how="cross")
    p = grid.merge(cell_daily, on=["cell_id", "date"], how="left", validate="many_to_one")
    weather = [c for c in cell_daily.columns if c not in ("cell_id", "date")]
    miss = p[weather].isna().any(axis=1)
    if miss.any():
        ex = p.loc[miss, ["zipcode", "date", "cell_id"]].head(3).to_dict("records")
        raise PanelError(f"{city}: no weather for {int(miss.sum())} zip-days, e.g. {ex}")
    p = p.merge(targets, on=["zipcode", "date"], how="left", validate="one_to_one")
    for c in COUNT_COLUMNS:
        p[c] = p[c].fillna(0)
    for c in ["n_reports", "n_report_hours", "n_reports_value1", "n_reports_excl1"]:
        p[c] = p[c].astype(int)
    p["has_report"] = p.n_reports > 0

    p = p.rename(columns={"lat": "zip_lat", "lon": "zip_lon", "in_original_set": "zip_in_original_set",
                          "n_reports_all": "zip_n_reports_all"})
    p.insert(0, "city", city)
    d = pd.to_datetime(p.date)
    p["year"], p["month"], p["year_month"] = d.dt.year, d.dt.month, d.dt.strftime("%Y-%m")
    p["dow"] = d.dt.dayofweek
    us = holidays.US(years=range(d.dt.year.min(), d.dt.year.max() + 1))
    p["is_holiday"] = d.dt.date.map(lambda x: x in us).astype(bool)
    p["sparse_reporting_period"] = (p.date < sparse_before) if sparse_before else False
    p["pressure_anomaly"] = p.atmospheric_pressure - p.groupby("zipcode").atmospheric_pressure.transform("mean")

    near = {z: _nearest(lat, lon, sources) for z, lat, lon in zips[["zipcode", "lat", "lon"]].itertuples(index=False)}
    p["nearest_source"] = p.zipcode.map(lambda z: near[z][0])
    p["dist_nearest_mi"] = p.zipcode.map(lambda z: near[z][1])
    for col, wd in [("wind_align_weighted", "wind_direction"), ("wind_align_weighted_circmean", "wind_direction_circmean")]:
        terms = [core.compute_multisource_terms(a, b, w, sources=sources)
                 for a, b, w in zip(p.zip_lat, p.zip_lon, p[wd])]
        p["multi_source_exposure"] = [t[0] for t in terms]
        p[col] = [t[1] for t in terms]
    return p


def _nearest(lat, lon, sources):
    d = [(core._pair_distance_miles(lat, lon, s["lat"], s["lon"]), s["name"]) for s in sources]
    dist, name = min(d)
    return name, dist


def add_events(p, reference):
    """Event columns: burden above the city's mean burden over the reference rows. Returns (panel, thresholds)."""
    thr = {b: float(p.loc[reference, b].mean()) for b in BURDENS}
    p = p.copy()
    for b, e in BURDENS.items():
        p[e] = (p[b] > thr[b]).astype(int)
    return p, thr


def order_columns(p):
    """Panel in documented column order; raises if a column is undocumented or missing."""
    extra = set(p.columns) - set(COLUMN_NAMES)
    missing = set(COLUMN_NAMES) - set(p.columns)
    if extra or missing:
        raise PanelError(f"panel columns differ from the data dictionary: extra {sorted(extra)}, missing {sorted(missing)}")
    return p[COLUMN_NAMES]
