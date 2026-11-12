"""ERA5 weather per 0.25-degree grid cell for the training panel, one source per variable for every date.

  * Copernicus ERA5 (the Climate Data Store's point time-series form, reanalysis-era5-single-levels-timeseries,
    which equals the standard gridded dataset value for value): temperature, relative humidity (from dew
    point), surface pressure, wind speed and direction (from u/v), shortwave radiation and boundary-layer height.
  * Open-Meteo's ERA5 archive (models=era5, requested at the grid cell itself: cell_selection=nearest,
    elevation=nan): rain only, because the Copernicus time series has no rain field.

Why not Open-Meteo for everything: its ERA5 has no boundary-layer height for 2024-01-01..06-29, is not current
ERA5 on 2021-12-01..12 (all cells), and its surface pressure is derived, not ERA5's surface-pressure field. No
variable mixes sources and nothing is filled; any gap raises. Open-Meteo's values of the Copernicus variables are
kept as om_* only to check that the two sources carry the same quantities (check_sources). Raw responses are
cached under training_panel_v2/raw/ with a sidecar recording the request and retrieval time.

    python -m training.era5_cells fetch-open-meteo   # stops starting requests at --stop-at (default 22:45)
    python -m training.era5_cells fetch-cds          # ERA5 time series per cell
    python -m training.era5_cells fetch-cds-standard-check
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weather_history as W  # noqa: E402
from odor_forecast_core import fetch_open_meteo_json  # noqa: E402
from training import config, reports  # noqa: E402

OM_DIR = os.path.join(config.RAW_DIR, "open_meteo_era5")
CDS_DIR = os.path.join(config.RAW_DIR, "cds_era5")
OM_UNITS = {"temperature_2m": "°F", "relative_humidity_2m": "%", "surface_pressure": "hPa",
            "wind_speed_10m": "mp/h", "wind_direction_10m": "°", "rain": "inch",
            "shortwave_radiation": "W/m²", "boundary_layer_height": "ft"}
# Where each hourly panel input comes from, for every date.
PANEL_SOURCES = {v: "copernicus" for v in W.HOURLY_VARS}
PANEL_SOURCES["rain"] = "open-meteo"
GRID = 0.25


class WeatherSourceError(RuntimeError):
    """A response is not what the panel assumes (wrong cell, units or coverage)."""


def grid_cell(lat, lon):
    """Nearest point of the 0.25-degree ERA5 grid (the cell Open-Meteo and Copernicus both return)."""
    return W._grid_point(lat, lon)


def cell_id(glat, glon):
    return f"{glat:.2f}_{glon:.2f}"


def city_cells(city):
    """Grid cells of every zip with coordinates in the city's raw reports (a superset of the panel's)."""
    cfg = config.CITIES[city]
    zc = reports.zip_centroids(reports.load_reports(cfg["reports"], cfg["tz"]))
    return sorted({grid_cell(a, b) for a, b in zip(zc.lat, zc.lon)})


def city_area(cells):
    """CDS area [N, W, S, E] covering the cells."""
    lats, lons = [c[0] for c in cells], [c[1] for c in cells]
    return [max(lats), min(lons), min(lats), max(lons)]


def _year_span(year):
    s = f"{year}-01-01"
    e = min(f"{year}-12-31", config.FETCH_END)
    return s, e


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- Open-Meteo ERA5 (all but BLH)

def om_path(glat, glon, year):
    return os.path.join(OM_DIR, f"{cell_id(glat, glon)}_{year}.json.gz")


def fetch_open_meteo_cell(glat, glon, year, fetch=fetch_open_meteo_json):
    """Fetch (or reuse) one cell-year from Open-Meteo's ERA5 archive. Returns True if a request was made."""
    path = om_path(glat, glon, year)
    if os.path.exists(path):
        return False
    s, e = _year_span(year)
    url = W.archive_url(glat, glon, s, e, models="era5", grid_cell=True)
    data = fetch(url)
    check_open_meteo_response(data, glat, glon)
    os.makedirs(OM_DIR, exist_ok=True)
    tmp = path + ".part"
    with gzip.open(tmp, "wt") as f:
        json.dump({"url": url, "retrieved_utc": _now_utc(), "data": data}, f)
    os.replace(tmp, path)
    return True


def check_open_meteo_response(data, glat, glon):
    if "hourly" not in data:
        raise WeatherSourceError(f"no hourly data for cell {cell_id(glat, glon)}: {str(data)[:200]}")
    if abs(data["latitude"] - glat) > 0.01 or abs(data["longitude"] - glon) > 0.01:
        raise WeatherSourceError(f"asked for cell {cell_id(glat, glon)}, Open-Meteo returned "
                                 f"({data['latitude']}, {data['longitude']})")
    units = data.get("hourly_units", {})
    bad = {v: units.get(v) for v, u in OM_UNITS.items() if units.get(v) != u}
    if bad:
        raise WeatherSourceError(f"unexpected Open-Meteo units for cell {cell_id(glat, glon)}: {bad}")


def load_open_meteo_cell(glat, glon, years):
    """Hourly Open-Meteo ERA5 (UTC index) for a cell over the cached years, plus response metadata."""
    parts, meta = [], []
    for y in years:
        with gzip.open(om_path(glat, glon, y), "rt") as f:
            rec = json.load(f)
        check_open_meteo_response(rec["data"], glat, glon)
        parts.append(W.parse_archive(rec["data"]))
        meta.append({"cell_id": cell_id(glat, glon), "year": y, "source": "open-meteo models=era5",
                     "url": rec["url"], "retrieved_utc": rec["retrieved_utc"], "hours": len(parts[-1]),
                     "elevation_m": rec["data"].get("elevation"), "file": os.path.relpath(om_path(glat, glon, y), config.OUT_DIR)})
    h = pd.concat(parts).sort_index()
    if h.index.duplicated().any():
        raise WeatherSourceError(f"duplicate hours for cell {cell_id(glat, glon)}")
    return h, meta


# ---------------------------------------------------------------- Copernicus ERA5 (BLH, all dates)
# The point time-series form of ERA5 on the Climate Data Store: one request per grid cell for the whole period.
# BLH is the panel's source. The other fields are fetched only to check that Open-Meteo's ERA5 values are the
# same quantities (check_sources); they never enter the panel.

TS_DATASET = "reanalysis-era5-single-levels-timeseries"
TS_FIELDS = ["boundary_layer_height", "2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
             "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation",
             "surface_solar_radiation_downwards"]
MS_TO_MPH = 2.2369363
M_TO_IN = 39.3700787


def cds_ts_path(glat, glon):
    return os.path.join(CDS_DIR, f"{cell_id(glat, glon)}_era5_timeseries.zip")


def cds_ts_dir(glat, glon):
    return os.path.join(CDS_DIR, f"{cell_id(glat, glon)}_era5_timeseries")


def cds_ts_request(glat, glon):
    return {"variable": TS_FIELDS, "location": {"longitude": glon, "latitude": glat},
            "date": [f"{config.START_DATE}/{config.FETCH_END}"], "data_format": "netcdf"}


def fetch_cds(request, path, dataset):
    """Retrieve one Copernicus request to path (skipped if cached) with a provenance sidecar."""
    if os.path.exists(path):
        return False
    import cdsapi
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    cdsapi.Client(quiet=True).retrieve(dataset, request, tmp)
    os.replace(tmp, path)
    with open(path + ".json", "w") as f:
        json.dump({"dataset": dataset, "request": request, "retrieved_utc": _now_utc(),
                   "bytes": os.path.getsize(path), "sha256": _sha256(path)}, f, indent=1)
    return True


def fetch_cds_cell(glat, glon):
    """Fetch (or reuse) a cell's ERA5 time series and unpack it next to the zip."""
    import zipfile
    made = fetch_cds(cds_ts_request(glat, glon), cds_ts_path(glat, glon), TS_DATASET)
    if not os.path.isdir(cds_ts_dir(glat, glon)):
        with zipfile.ZipFile(cds_ts_path(glat, glon)) as z:
            z.extractall(cds_ts_dir(glat, glon))
    return made


def _rh_magnus(t_c, td_c):
    return 100 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))


def load_cds_cell(glat, glon):
    """A cell's Copernicus ERA5 series (UTC index) in Open-Meteo's names and units (BLH in feet), plus
    cds_total_precipitation for the rain check. Raises if the files are not at the requested grid point."""
    import glob
    import xarray as xr
    files = sorted(glob.glob(os.path.join(cds_ts_dir(glat, glon), "*.nc")))
    if not files:
        raise WeatherSourceError(f"no Copernicus time series for cell {cell_id(glat, glon)}")
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        la, lo = float(ds.latitude.values), float(ds.longitude.values)
        if abs(la - glat) > 1e-3 or abs(lo - glon) > 1e-3:
            raise WeatherSourceError(f"{f} is at grid point ({la}, {lo}), not cell {cell_id(glat, glon)}")
        parts.append(ds.drop_vars([v for v in ("latitude", "longitude", "number", "expver") if v in ds.variables])
                     .to_dataframe())
        ds.close()
    c = pd.concat(parts, axis=1).sort_index().astype(float)
    c.index = pd.to_datetime(c.index)
    c = c[~c.index.duplicated()]
    t, td = c.t2m - 273.15, c.d2m - 273.15
    out = pd.DataFrame({
        "temperature_2m": t * 9 / 5 + 32,
        # ERA5's dew point can sit a hair above its temperature; relative humidity is capped at 100 (as Open-Meteo does)
        "relative_humidity_2m": _rh_magnus(t, td).clip(upper=100.0),
        "surface_pressure": c.sp / 100,
        "wind_speed_10m": np.hypot(c.u10, c.v10) * MS_TO_MPH,
        "wind_direction_10m": np.degrees(np.arctan2(-c.u10, -c.v10)) % 360,
        "shortwave_radiation": c.ssrd / 3600,
        "boundary_layer_height": c.blh * W.M_TO_FT,
        "cds_total_precipitation": c.tp * M_TO_IN,     # rain + snow; for the rain check only
    }, index=c.index)
    side = json.load(open(cds_ts_path(glat, glon) + ".json"))
    meta = [{"cell_id": cell_id(glat, glon), "year": "all", "source": f"copernicus cds {TS_DATASET}",
             "url": "cds:" + json.dumps(side["request"], separators=(",", ":")),
             "retrieved_utc": side["retrieved_utc"], "hours": int(out.boundary_layer_height.notna().sum()),
             "file": os.path.relpath(cds_ts_path(glat, glon), config.OUT_DIR)}]
    return out, meta


def cell_hourly(city, glat, glon, years, start_utc, end_utc):
    """Hourly model inputs for a cell, each variable from its PANEL_SOURCES source for every hour.

    Returns (hourly frame over [start_utc, end_utc], provenance rows). The frame has the panel inputs under
    weather_history's names, Open-Meteo's values of every variable as om_*, and cds_total_precipitation; the
    om_* and cds_* columns are for check_sources only.
    """
    om, meta_om = load_open_meteo_cell(glat, glon, years)
    cds, meta_cds = load_cds_cell(glat, glon)
    idx = pd.date_range(start_utc, end_utc, freq="h")
    om, cds = om.reindex(idx).add_prefix("om_"), cds.reindex(idx)
    h = pd.DataFrame(index=idx)
    for v, src in PANEL_SOURCES.items():
        h[v] = cds[v] if src == "copernicus" else om["om_" + v]
    h = h.join(om).join(cds[["cds_total_precipitation"]])
    return h, meta_om + meta_cds


# The standard (gridded) ERA5 dataset for Pittsburgh's cells in 2018: checks that the time-series form
# serves the same boundary-layer height.
STD_DATASET = "reanalysis-era5-single-levels"


def std_check_path():
    return os.path.join(CDS_DIR, "standard_check", "pgh_blh_2018.nc")


def std_check_request():
    return {"product_type": ["reanalysis"], "variable": ["boundary_layer_height"], "year": ["2018"],
            "month": [f"{m:02d}" for m in range(1, 13)], "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)], "data_format": "netcdf",
            "download_format": "unarchived", "area": city_area(city_cells("pgh"))}


def dec2021_check_path():
    return os.path.join(CDS_DIR, "standard_check", "pgh_2021-12-01_12.nc")


def dec2021_check_request():
    """Open-Meteo's ERA5 and the Copernicus time series disagree on 2021-12-01..12 at every cell; the standard
    dataset decides which copy is ECMWF's current ERA5."""
    return {"product_type": ["reanalysis"], "variable": ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
            "10m_u_component_of_wind", "10m_v_component_of_wind", "boundary_layer_height"],
            "year": ["2021"], "month": ["12"], "day": [f"{d:02d}" for d in range(1, 13)],
            "time": [f"{h:02d}:00" for h in range(24)], "data_format": "netcdf", "download_format": "unarchived",
            "area": city_area(city_cells("pgh"))}


def load_std_check_blh(glat, glon):
    """BLH (feet) at one grid point from the standard-dataset check file."""
    import xarray as xr
    ds = xr.open_dataset(std_check_path())
    s = ds["blh"].sel(latitude=glat, longitude=glon).to_series().astype(float)
    ds.close()
    s.index = pd.to_datetime(s.index)
    return s * W.M_TO_FT


def cell_daily(h, start_date, end_date, tz, where):
    """Daily model features for local days start..end from a cell's hourly inputs (weather_history recipe),
    plus wind_direction_circmean, the unweighted circular mean the original training scripts used.
    Any missing hour or a day without 23-25 hours raises MissingWeatherError."""
    d = W.daily_from_hourly(h[W.HOURLY_VARS], start_date, end_date, tz, where=where).drop(columns="era5_filled_hours")
    dates = h.index.tz_localize("UTC").tz_convert(tz).strftime("%Y-%m-%d")
    rad = np.radians(h.wind_direction_10m)
    uv = pd.DataFrame({"u": np.sin(rad), "v": np.cos(rad), "date": dates}, index=h.index)
    uv = uv[(uv.date >= start_date) & (uv.date <= end_date)].groupby("date")[["u", "v"]].mean()
    d["wind_direction_circmean"] = np.degrees(np.arctan2(uv.u, uv.v)).reindex(d.index) % 360
    return d


# ---------------------------------------------------------------- source check

# Open-Meteo's ERA5 value -> the Copernicus ERA5 series it should equal. Open-Meteo rounds what it stores
# (0.1 F, 0.1 hPa, 0.1 mph, whole %, whole degrees, whole W/m2, 5 m BLH, 0.1 mm rain), so small differences
# are expected; a systematic offset, a ratio away from 1 or a best lag other than 0 means they are not the
# same quantity. Rain (the panel's only Open-Meteo variable) excludes snow, so it should equal ERA5 total
# precipitation except when snow falls.
SOURCE_PAIRS = {f"om_{v}": v for v in W.HOURLY_VARS if v != "rain"}
SOURCE_PAIRS["om_rain"] = "cds_total_precipitation"


def compare_series(a, b, circular=False):
    """Agreement of two hourly series on the hours both have, and the lag (h) of b that best matches a."""
    def diff(x, y):
        d = x - y
        return ((d + 180) % 360) - 180 if circular else d
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = diff(both.a, both.b)
    lag_err = {lag: diff(a, b.shift(lag)).abs().mean() for lag in (-2, -1, 0, 1, 2)}
    return {"hours_compared": len(both), "mean_diff": float(d.mean()), "mean_abs_diff": float(d.abs().mean()),
            "p99_abs_diff": float(d.abs().quantile(0.99)) if len(d) else float("nan"),
            "max_abs_diff": float(d.abs().max()) if len(d) else float("nan"),
            "ratio_of_means": float("nan") if circular else float(both.a.mean() / both.b.mean()),
            "corr": float("nan") if circular else float(both.a.corr(both.b)),
            "best_lag_h": int(min(lag_err, key=lambda k: lag_err[k]))}


def check_sources(h):
    """One row per panel variable: Open-Meteo ERA5 vs Copernicus ERA5 at the same cell and hours."""
    rows = []
    for om, cds in SOURCE_PAIRS.items():
        r = compare_series(h[om], h[cds], circular=om == "om_wind_direction_10m")
        rows.append({"variable": om.replace("om_", ""), "panel_source": PANEL_SOURCES[om.replace("om_", "")],
                     "open_meteo": om, "copernicus": cds,
                     "open_meteo_missing_hours": int(h[om].isna().sum()), **r})
    return rows


# ---------------------------------------------------------------- CLI

def _past_stop(stop_at):
    now = dt.datetime.now().time()
    stop = dt.time(*map(int, stop_at.split(":")))
    return now >= stop or now < dt.time(7, 0)


def fetch_open_meteo_all(stop_at="22:45", pause_s=8.0):
    jobs = [(c, y) for city in config.CITIES for c in city_cells(city) for y in config.CITIES[city]["fetch_years"]]
    jobs = list(dict.fromkeys(jobs))
    todo = [j for j in jobs if not os.path.exists(om_path(*j[0], j[1]))]
    print(f"open-meteo: {len(jobs)} cell-years, {len(todo)} to fetch", flush=True)
    for i, ((glat, glon), y) in enumerate(todo, 1):
        if _past_stop(stop_at):
            print(f"stopping before {cell_id(glat, glon)} {y}: past {stop_at} (HYSPLIT download window). "
                  f"{len(todo) - i + 1} left; rerun after 07:00.", flush=True)
            return 1
        fetch_open_meteo_cell(glat, glon, y)
        print(f"  [{i}/{len(todo)}] {cell_id(glat, glon)} {y} ok", flush=True)
        time.sleep(pause_s)
    print("open-meteo: done", flush=True)
    return 0


def fetch_cds_all(pause_s=5.0):
    cells = list(dict.fromkeys(c for city in config.CITIES for c in city_cells(city)))
    for i, (glat, glon) in enumerate(cells, 1):
        t0 = time.time()
        made = fetch_cds_cell(glat, glon)
        print(f"  [{i}/{len(cells)}] cds {cell_id(glat, glon)}: {'fetched' if made else 'cached'} ({time.time() - t0:.0f}s)", flush=True)
        if made:
            time.sleep(pause_s)
    print("cds: done", flush=True)
    return 0


def fetch_std_check():
    made = fetch_cds(dec2021_check_request(), dec2021_check_path(), STD_DATASET)
    print(f"standard-dataset Dec 2021 check: {'fetched' if made else 'cached'}", flush=True)
    made = fetch_cds(std_check_request(), std_check_path(), STD_DATASET)
    print(f"standard-dataset 2018 BLH check: {'fetched' if made else 'cached'}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fetch-open-meteo", "fetch-cds", "fetch-cds-standard-check"])
    ap.add_argument("--stop-at", default="22:45")
    a = ap.parse_args()
    run = {"fetch-open-meteo": lambda: fetch_open_meteo_all(a.stop_at), "fetch-cds": fetch_cds_all,
           "fetch-cds-standard-check": fetch_std_check}
    sys.exit(run[a.what]())
