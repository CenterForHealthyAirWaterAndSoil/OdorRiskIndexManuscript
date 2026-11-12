"""Inputs for scoring the ORI refit at the Calvert monitors (REFIT_PLAN.md s8; DEVIATIONS 23 M1-M3, M15, M20).

Builds and saves every input to refit_inputs/ so the files can be hashed and logged in DEVIATIONS before anything is
scored. Nothing here reads a monitor measurement.

  era5_cell_hourly.parquet   Calvert ERA5 cell 37.00_-88.25, UTC hours 2018-01-01..2026-06-06, panel sources (Copernicus
                             ERA5 time series; Open-Meteo ERA5 rain), via training/era5_cells.py exactly as the panel.
  era5_cell_daily.csv        local America/Chicago days 2018-01-01..2026-06-05 (the panel's daily recipe) with
                             pressure_anomaly = surface pressure minus the cell's mean over those days (plan s7).
  era5_comparator_hourly.parquet  UTC hours 2022-12-31..2026-01-01 for the report-free comparators (M3): the cell's
                             Copernicus 10 m wind, BLH, 2 m temperature, surface pressure and Open-Meteo ERA5 rain from
                             the panel files above, Open-Meteo ERA5 100 m wind at the cell, and Copernicus ERA5
                             pressure-level 925/850 hPa temperature and 500 hPa wind at the same grid point.
  deployed_inputs_daily.csv  the deployed model's inputs (weather_history.daily_features default) at each monitor,
                             2023-2025: for the refit-on-deployed-inputs sensitivity (s8 model-by-inputs).
  ifs_lead0_daily.csv        archived ECMWF IFS forecasts (Open-Meteo historical-forecast API, models=ecmwf_ifs,
                             requested like the live site at the cell's elevation) at each monitor, 2024-09-01..
                             2025-12-31, complete local days only (M15 secondary 8). Leads >= 1 are not built: the
                             previous-runs archive has no boundary-layer height for ecmwf_ifs at any lead.
  local_reports_2026.csv     Smell MyCity reports from 420xx zip codes within 20 miles of the complex, 2026 (M20).
  era5_cell_daily_2026.csv   Calvert cell ERA5 local days 2026-01-01..2026-10-15 for M20 (same sources).
Missing values are never filled: an incomplete hour or day stops the build or, for the archived forecasts and the
2026 descriptive line only, leaves that day without a value.

Run: ~/weather-varaible-analysis/.venv/bin/python refit_calvert_inputs.py
"""
import glob
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests
import xarray as xr

REPO = "/path/to/weather-varaible-analysis"
sys.path.insert(0, REPO)
import odor_forecast_core as M  # noqa: E402
import weather_history as W  # noqa: E402
from training import config, era5_cells as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "refit_inputs")
CELL = (37.0, -88.25)
TZ = "America/Chicago"
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879),
         "lwd": (37.047914, -88.338323)}
UNITS = "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=GMT"
M20_START, M20_END = "2026-01-01", "2026-10-15"
M20_RADIUS_MI = 20.0


def save(df, name, **kw):
    path = os.path.join(OUT, name)
    (df.to_parquet if name.endswith(".parquet") else df.to_csv)(path, **kw)
    print("wrote", name, len(df), flush=True)


def complete_days(h, tz, cols):
    """Local days with every hour present (23-25) and no missing value in cols. Never fills."""
    local = h.index.tz_localize("UTC").tz_convert(tz)
    d = pd.Series(local.strftime("%Y-%m-%d"), index=h.index)
    ok = h[cols].notna().all(axis=1).groupby(d).all()
    n = d.groupby(d).size()
    return sorted(set(ok[ok].index) & set(n[(n >= 23) & (n <= 25)].index))


def daily_complete_only(h, tz, where):
    """The weather_history daily recipe applied to complete local days only (archived forecasts, M20)."""
    rows = []
    for day in complete_days(h, tz, W.HOURLY_VARS):
        rows.append(W.daily_from_hourly(h, day, day, tz, where=where))
    return pd.concat(rows) if rows else pd.DataFrame()


def cell_panel_style():
    """Calvert cell hourly and daily ERA5 exactly as training/build_training_panel.py builds a cell."""
    glat, glon = CELL
    E.fetch_cds_cell(glat, glon)
    years = range(int(config.START_DATE[:4]), int(config.FETCH_END[:4]) + 1)
    for y in years:
        if E.fetch_open_meteo_cell(glat, glon, y):
            time.sleep(8)
    start_utc = f"{config.START_DATE} 00:00"
    end_utc = (pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 23:00")
    h, meta = E.cell_hourly("calvert", glat, glon, years, start_utc, end_utc)
    d = E.cell_daily(h, config.START_DATE, config.END_DATE, TZ, where=f"calvert cell {E.cell_id(glat, glon)}")
    d["pressure_anomaly"] = d.atmospheric_pressure - d.atmospheric_pressure.mean()
    d.attrs["pressure_mean_hpa"] = float(d.atmospheric_pressure.mean())
    return h, d, meta


def pressure_levels():
    """Copernicus ERA5 925/850 hPa temperature and 500 hPa wind at 37.00 N 88.25 W (the cached files of
    fetch_era5_upper.py, the same grid point as the cell)."""
    def load(pattern):
        parts = [xr.open_dataset(f).load() for f in sorted(glob.glob(os.path.join(HERE, pattern)))]
        for p in parts:
            la, lo = float(np.atleast_1d(p.latitude.values)[0]), float(np.atleast_1d(p.longitude.values)[0])
            if abs(la - CELL[0]) > 1e-3 or abs(lo - CELL[1]) > 1e-3:
                raise RuntimeError(f"pressure-level file at ({la}, {lo}), not the Calvert cell")
        tdim = "valid_time" if "valid_time" in parts[0].dims else "time"
        return xr.concat(parts, dim=tdim).sortby(tdim)
    t, w = load("era5_pl_temp_*.nc"), load("era5_pl_wind500_*.nc")
    lev = "pressure_level" if "pressure_level" in t.dims else "level"
    df = pd.DataFrame({"t925": t["t"].sel({lev: 925}).squeeze().to_series(),
                       "t850": t["t"].sel({lev: 850}).squeeze().to_series()})
    u = w["u"].sel({lev: 500}).squeeze().to_series()
    v = w["v"].sel({lev: 500}).squeeze().to_series()
    df["wind500"] = np.hypot(u, v).reindex(df.index)
    df.index = pd.to_datetime(df.index)
    return df


def w100_era5():
    glat, glon = CELL
    url = (f"{W.ARCHIVE_URL}?latitude={glat}&longitude={glon}&start_date=2022-12-31&end_date=2026-01-01"
           f"&hourly=wind_speed_100m&wind_speed_unit=ms&timezone=GMT&models=era5&cell_selection=nearest&elevation=nan")
    d = M.fetch_open_meteo_json(url)
    if abs(d["latitude"] - glat) > 0.01 or abs(d["longitude"] - glon) > 0.01:
        raise RuntimeError(f"Open-Meteo returned ({d['latitude']}, {d['longitude']}) for the 100 m wind")
    return pd.Series(d["hourly"]["wind_speed_100m"], index=pd.to_datetime(d["hourly"]["time"]), dtype=float)


def ifs_lead0(lat, lon, elev):
    url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&elevation={elev}&start_date=2024-08-31&end_date=2026-01-01&hourly={','.join(W.HOURLY_VARS)}{UNITS}"
           f"&models=ecmwf_ifs")
    h = W.parse_archive(M.fetch_open_meteo_json(url))
    g = daily_complete_only(h, TZ, where=f"ifs ({lat}, {lon})")
    return g[(g.index >= "2024-09-01") & (g.index <= "2025-12-31")]


def local_reports():
    zips = ",".join(str(z) for z in range(42001, 42100))
    t0 = int(pd.Timestamp(M20_START, tz=TZ).timestamp())
    t1 = int((pd.Timestamp(M20_END, tz=TZ) + pd.Timedelta(days=1)).timestamp())
    r = requests.get("https://api.smellpittsburgh.org/api/v2/smell_reports",
                     params={"zipcodes": zips, "start_time": t0, "end_time": t1}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["dist_mi"] = [M._pair_distance_miles(a, b, M.IND_LAT, M.IND_LON) for a, b in zip(df.latitude, df.longitude)]
    df = df[df.dist_mi <= M20_RADIUS_MI].copy()
    df["date"] = pd.to_datetime(df.observed_at, unit="s", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    return df[["date", "observed_at", "zipcode", "latitude", "longitude", "dist_mi", "smell_value"]]


def cell_2026():
    """Calvert cell ERA5 for the M20 window, same sources as the panel, complete local days only."""
    import cdsapi
    glat, glon = CELL
    zpath = os.path.join(OUT, "cds_era5_2026_timeseries.zip")
    if not os.path.exists(zpath):
        req = {"variable": E.TS_FIELDS, "location": {"longitude": glon, "latitude": glat},
               "date": [f"2025-12-31/{(pd.Timestamp(M20_END) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"],
               "data_format": "netcdf"}
        cdsapi.Client(quiet=True).retrieve(E.TS_DATASET, req, zpath + ".part")
        os.replace(zpath + ".part", zpath)
    ddir = zpath[:-4]
    if not os.path.isdir(ddir):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ddir)
    parts = []
    for f in sorted(glob.glob(os.path.join(ddir, "*.nc"))):
        ds = xr.open_dataset(f)
        if abs(float(ds.latitude.values) - glat) > 1e-3 or abs(float(ds.longitude.values) - glon) > 1e-3:
            raise RuntimeError(f"{f} is not at the Calvert cell")
        parts.append(ds.drop_vars([v for v in ("latitude", "longitude", "number", "expver") if v in ds.variables])
                     .to_dataframe())
        ds.close()
    c = pd.concat(parts, axis=1).sort_index().astype(float)
    c.index = pd.to_datetime(c.index)
    c = c[~c.index.duplicated()]
    t, td = c.t2m - 273.15, c.d2m - 273.15
    h = pd.DataFrame({"temperature_2m": t * 9 / 5 + 32,
                      "relative_humidity_2m": E._rh_magnus(t, td).clip(upper=100.0),
                      "surface_pressure": c.sp / 100, "wind_speed_10m": np.hypot(c.u10, c.v10) * E.MS_TO_MPH,
                      "wind_direction_10m": np.degrees(np.arctan2(-c.u10, -c.v10)) % 360,
                      "shortwave_radiation": c.ssrd / 3600, "boundary_layer_height": c.blh * W.M_TO_FT}, index=c.index)
    url = W.archive_url(glat, glon, "2025-12-31", (pd.Timestamp(M20_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        models="era5", grid_cell=True)
    om = W.parse_archive(M.fetch_open_meteo_json(url))
    h = h.join(om[["rain"]], how="left")[W.HOURLY_VARS]
    d = daily_complete_only(h, TZ, where="calvert cell 2026")
    return d[(d.index >= M20_START) & (d.index <= M20_END)]


def main():
    os.makedirs(OUT, exist_ok=True)
    h, d, meta = cell_panel_style()
    save(h[W.HOURLY_VARS].rename_axis("time_utc"), "era5_cell_hourly.parquet")
    save(d.rename_axis("date"), "era5_cell_daily.csv")
    json.dump({"cell": CELL, "pressure_mean_hpa_2018_2026": d.attrs["pressure_mean_hpa"], "provenance": meta},
              open(os.path.join(OUT, "era5_cell_provenance.json"), "w"), indent=1)

    comp = h.loc["2022-12-31":"2026-01-01 23:00", W.HOURLY_VARS].join(w100_era5().rename("w100_ms")).join(pressure_levels())
    save(comp.rename_axis("time_utc"), "era5_comparator_hourly.parquet")

    dep = []
    for s, (lat, lon) in SITES.items():
        f = W.daily_features(lat, lon, "2023-01-01", "2025-12-31", tz=TZ)
        dep.append(f.assign(site=s).rename_axis("date").reset_index())
        time.sleep(2)
    save(pd.concat(dep, ignore_index=True), "deployed_inputs_daily.csv", index=False)

    ifs = []
    elev = M.era5_cell_elevation(*CELL)
    for s, (lat, lon) in SITES.items():
        if s == "lwd":
            continue
        ifs.append(ifs_lead0(lat, lon, elev).assign(site=s).rename_axis("date").reset_index())
        time.sleep(2)
    save(pd.concat(ifs, ignore_index=True), "ifs_lead0_daily.csv", index=False)

    save(local_reports(), "local_reports_2026.csv", index=False)
    save(cell_2026().rename_axis("date"), "era5_cell_daily_2026.csv")


if __name__ == "__main__":
    main()
