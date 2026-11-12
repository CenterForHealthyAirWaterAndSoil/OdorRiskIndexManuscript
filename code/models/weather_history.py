"""
Standard source for past weather in this project's analyses: validation backtests and local-report
checks. The ORI refit's training panel uses training/ instead (one ERA5 source per variable). See
DATA_SOURCES.md for the full source map.

Source: Open-Meteo historical archive (archive-api.open-meteo.com), imperial units.
  1. By default, Open-Meteo's default archive model, as the old training data used (Pittsburgh/Louisville
     fetch_weather_for_zips*.py). That default is ECMWF IFS (~9 km) for every variable IFS has, on every date,
     and ERA5 for the rest (boundary-layer height until IFS gained it around 2024-09); checked 2026-10-23.
     models="era5" asks for ERA5 for every variable instead.
  2. Any hour Open-Meteo leaves empty is filled from Copernicus ERA5 directly (the same reanalysis),
     through the Copernicus Climate Data Store (needs ~/.cdsapirc). Fills are cached under
     data/era5_fill_cache/ so each month is requested once.
     Known gap: boundary_layer_height is empty from 2024-01-01 to 2024-06-29 at every location.
  3. Anything still missing raises MissingWeatherError. Nothing is defaulted or silently dropped.

Daily features use the same recipe as training and the live forecast: hours are fetched in UTC,
converted to the location's local time, and grouped by local calendar day. Daily mean temperature,
humidity, pressure, wind speed, shortwave radiation and boundary-layer height; daily rain sum;
diurnal range = max - min temperature; wind direction = speed-weighted vector mean.
Units: degrees F, mph, inches, hPa, W/m2, feet (boundary-layer height).
"""
import os
import time

import numpy as np
import pandas as pd

from odor_forecast_core import fetch_open_meteo_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
               "wind_direction_10m", "rain", "shortwave_radiation", "boundary_layer_height"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "era5_fill_cache")
M_TO_FT = 3.28084

# Open-Meteo hourly variable -> (Copernicus ERA5 variable, NetCDF short name, converter to Open-Meteo units).
# Only variables with a known Open-Meteo gap are mapped; add others here if a new gap appears.
ERA5_FILL = {
    "boundary_layer_height": ("boundary_layer_height", "blh", lambda m: m * M_TO_FT),
}


class MissingWeatherError(RuntimeError):
    """Weather data is missing and could not be filled from ERA5."""


def _grid_point(lat, lon):
    """Nearest point on the 0.25-degree ERA5 grid."""
    return round(lat * 4) / 4, round(lon * 4) / 4


def _cache_path(var, glat, glon):
    return os.path.join(CACHE_DIR, f"{var}_{glat:.2f}_{glon:.2f}.csv")


def _load_cache(var, glat, glon):
    p = _cache_path(var, glat, glon)
    if not os.path.exists(p):
        return pd.Series(dtype=float)
    c = pd.read_csv(p, parse_dates=["time_utc"])
    return c.set_index("time_utc")["value"].sort_index()


def _save_cache(var, glat, glon, series):
    os.makedirs(CACHE_DIR, exist_ok=True)
    old = _load_cache(var, glat, glon)
    merged = pd.concat([old, series])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.rename("value").rename_axis("time_utc").reset_index().to_csv(_cache_path(var, glat, glon), index=False)


def seed_cache_from_netcdf(nc_path, var):
    """Add every grid point in an ERA5 NetCDF already downloaded from Copernicus to the cache."""
    import xarray as xr
    era5_var, short, conv = ERA5_FILL[var]
    ds = xr.open_dataset(nc_path)
    tdim = "valid_time" if "valid_time" in ds.coords else "time"
    for glat in ds.latitude.values:
        for glon in ds.longitude.values:
            s = ds[short].sel(latitude=glat, longitude=glon).to_series()
            s.index = pd.to_datetime(s.index)
            _save_cache(var, float(glat), float(glon), conv(s.astype(float)))
    return ds


def _fetch_era5(var, glat, glon, months):
    """Request whole months of one ERA5 variable at one grid point from Copernicus."""
    import cdsapi
    import xarray as xr
    era5_var, short, conv = ERA5_FILL[var]
    os.makedirs(CACHE_DIR, exist_ok=True)
    for year in sorted({m[:4] for m in months}):
        mlist = sorted(m[5:7] for m in months if m.startswith(year))
        target = os.path.join(CACHE_DIR, f"_req_{var}_{glat:.2f}_{glon:.2f}_{year}.nc")
        cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": [era5_var], "year": [year], "month": mlist,
            "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf", "download_format": "unarchived",
            "area": [glat, glon, glat, glon]}, target)
        ds = xr.open_dataset(target)
        s = ds[short].sel(latitude=glat, longitude=glon, method="nearest").to_series()
        s.index = pd.to_datetime(s.index)
        _save_cache(var, glat, glon, conv(s.astype(float)))
        ds.close()
        os.remove(target)


def _fill_from_era5(hourly, lat, lon):
    """Fill empty hours of mapped variables from Copernicus ERA5. Returns (hourly, filled_counts)."""
    glat, glon = _grid_point(lat, lon)
    filled = {}
    hourly["era5_filled"] = False
    for var in ERA5_FILL:
        miss = hourly[var].isna()
        if not miss.any():
            continue
        cache = _load_cache(var, glat, glon)
        need = hourly.index[miss]
        months = sorted({t.strftime("%Y-%m") for t in need if t not in cache.index})
        if months:
            try:
                _fetch_era5(var, glat, glon, months)
            except Exception as e:
                raise MissingWeatherError(
                    f"{var} is missing for {miss.sum()} hours at ({lat}, {lon}) and the Copernicus ERA5 "
                    f"fill failed ({e}). Check ~/.cdsapirc and the ERA5 licence.") from e
            cache = _load_cache(var, glat, glon)
        hourly.loc[miss, var] = cache.reindex(need).values
        hourly.loc[miss & hourly[var].notna(), "era5_filled"] = True
        filled[var] = int(miss.sum() - hourly[var].isna().sum())
    return hourly, filled


def archive_url(lat, lon, start_date, end_date, models=None, grid_cell=False):
    """Open-Meteo archive request for HOURLY_VARS, in UTC and imperial units.

    models: None keeps Open-Meteo's default model (mostly ECMWF IFS, see the module docstring);
    "era5" requests ERA5 for every variable and date.
    grid_cell: True returns the nearest grid cell's own values (cell_selection=nearest, elevation=nan),
    without Open-Meteo's downscaling of temperature and pressure to the point's elevation.
    """
    url = (f"{ARCHIVE_URL}?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
           f"&hourly={','.join(HOURLY_VARS)}&temperature_unit=fahrenheit&wind_speed_unit=mph"
           f"&precipitation_unit=inch&timezone=GMT")
    if models:
        url += f"&models={models}"
    if grid_cell:
        url += "&cell_selection=nearest&elevation=nan"
    return url


def parse_archive(data):
    """Open-Meteo archive JSON -> hourly DataFrame indexed by UTC time, boundary-layer height in feet."""
    units = data.get("hourly_units", {})
    h = pd.DataFrame(data["hourly"])
    h.index = pd.to_datetime(h.pop("time"))
    h = h.apply(pd.to_numeric, errors="coerce")
    # Imperial requests return boundary-layer height in feet; convert only if Open-Meteo says metres.
    if str(units.get("boundary_layer_height", "ft")).lower().startswith("m"):
        h["boundary_layer_height"] *= M_TO_FT
    return h


def hourly_weather(lat, lon, start_date, end_date, models=None):
    """Hourly weather in UTC (index) with gaps filled; raises MissingWeatherError if any remain."""
    # Pad one day each side so every local calendar day in [start, end] has all of its UTC hours.
    s = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    e = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    h = parse_archive(fetch_open_meteo_json(archive_url(lat, lon, s, e, models=models)))
    h, filled = _fill_from_era5(h, lat, lon)
    h.attrs["era5_filled_hours"] = filled
    return h


def daily_features(lat, lon, start_date, end_date, tz, models=None):
    """Daily model features for local calendar days start_date..end_date (inclusive).

    tz is required (e.g. "America/Chicago" for Calvert, "America/New_York" for Pittsburgh): it decides
    which UTC hours make up each local day. models=None keeps Open-Meteo's default archive model;
    models="era5" requests ERA5 explicitly.

    Returns a DataFrame indexed by local date (YYYY-MM-DD) with the model's weather columns plus
    'era5_filled_hours' (how many hours of each local day came from the Copernicus fill).
    Raises MissingWeatherError if any day is incomplete or any value is still missing.
    """
    h = hourly_weather(lat, lon, start_date, end_date, models=models)
    g = daily_from_hourly(h, start_date, end_date, tz, where=f"({lat}, {lon})")
    g.attrs["era5_filled_hours"] = h.attrs.get("era5_filled_hours", {})
    return g


def daily_from_hourly(h, start_date, end_date, tz, where=""):
    """Aggregate hourly UTC weather (HOURLY_VARS columns) to local calendar days start_date..end_date.

    Raises MissingWeatherError on any missing hourly value or any day without 23-25 hours.
    """
    h = h.copy()
    if "era5_filled" not in h.columns:
        h["era5_filled"] = False
    local = h.index.tz_localize("UTC").tz_convert(tz)
    h["date"] = local.strftime("%Y-%m-%d")
    h["wind_u"] = h.wind_speed_10m * np.sin(np.radians(h.wind_direction_10m))
    h["wind_v"] = h.wind_speed_10m * np.cos(np.radians(h.wind_direction_10m))
    h = h[(h.date >= start_date) & (h.date <= end_date)]
    # Check hours, not days: a daily mean silently skips a missing hour.
    missing = h[HOURLY_VARS].isna().sum()
    if missing.any():
        raise MissingWeatherError(f"missing hourly values at {where} with no ERA5 fill mapped: "
                                  f"{missing[missing > 0].to_dict()}")
    g = h.groupby("date").agg(
        temperature=("temperature_2m", "mean"), temp_min=("temperature_2m", "min"),
        temp_max=("temperature_2m", "max"), precipitation=("rain", "sum"),
        wind_speed=("wind_speed_10m", "mean"), wind_u=("wind_u", "mean"), wind_v=("wind_v", "mean"),
        relative_humidity=("relative_humidity_2m", "mean"), atmospheric_pressure=("surface_pressure", "mean"),
        solar_radiation=("shortwave_radiation", "mean"), boundary_layer_height=("boundary_layer_height", "mean"),
        n_hours=("temperature_2m", "size"), era5_filled_hours=("era5_filled", "sum"))
    g["wind_direction"] = np.degrees(np.arctan2(g.wind_u, g.wind_v)) % 360
    g["diurnal_temperature_range"] = g.temp_max - g.temp_min
    g["temperature_squared"] = g.temperature ** 2
    g = g.drop(columns=["wind_u", "wind_v"])
    expected = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
    short = [d for d in expected if d not in g.index or not 23 <= g.loc[d, "n_hours"] <= 25]
    if short:
        raise MissingWeatherError(f"{len(short)} local days at {where} lack a full set of hours, e.g. {short[:3]}")
    if g.isna().any().any():
        bad = g.columns[g.isna().any()].tolist()
        raise MissingWeatherError(f"missing values remain in {bad} at {where}")
    g["era5_filled_hours"] = g.era5_filled_hours.astype(int)
    return g
