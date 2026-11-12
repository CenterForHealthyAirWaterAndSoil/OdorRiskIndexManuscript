"""ECMWF IFS variant of the training panel, for the pre-registered comparison in refit/IFS_COMPARISON_PLAN.md
(DEVIATIONS item 28).

One source per variable for every date: Open-Meteo's ECMWF IFS archive (models=ecmwf_ifs, each zip's 9 km IFS grid
cell, the cell's own values: cell_selection=nearest, elevation=nan) for temperature, relative humidity, surface
pressure, wind speed and direction, rain and shortwave radiation; Copernicus ERA5 boundary-layer height from the
zip's ERA5 cell, exactly as in the main panel. Everything else (reports, outcome, proximity) comes from the main
panel unchanged. Missing values are never filled: a zip-day with any missing hour is left out and counted.

    python -m training.ifs_panel discover   # each zip's IFS grid cell (one-day requests)
    python -m training.ifs_panel fetch      # every IFS cell-year, paced; stops cleanly on a rate limit (rerun later)
    python -m training.ifs_panel build      # training_panel_v2/ifs_variant/zip_day_panel_ifs.parquet
"""
import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import odor_forecast_core as core
import weather_history as W
from training import config

IFS_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
            "rain", "shortwave_radiation"]
IFS_DIR = os.path.join(config.RAW_DIR, "open_meteo_ifs")
OUT_DIR = os.path.join(config.OUT_DIR, "ifs_variant")
LOOKUP = os.path.join(OUT_DIR, "ifs_cell_lookup.csv")
PANEL = os.path.join(config.OUT_DIR, "zip_day_panel.parquet")
HOURLY_ERA5 = os.path.join(config.OUT_DIR, "weather_cell_hourly.parquet")
YEARS = range(int(config.START_DATE[:4]), int(config.FETCH_END[:4]) + 1)
UNITS = {"temperature_2m": "°F", "relative_humidity_2m": "%", "surface_pressure": "hPa", "wind_speed_10m": "mp/h",
         "wind_direction_10m": "°", "rain": "inch", "shortwave_radiation": "W/m²"}
WEATHER_COLS = ["temperature", "temp_min", "temp_max", "temperature_squared", "diurnal_temperature_range",
                "relative_humidity", "wind_speed", "wind_direction", "precipitation", "solar_radiation",
                "boundary_layer_height", "atmospheric_pressure", "pressure_anomaly", "wind_align_weighted"]


class IFSError(RuntimeError):
    """A response is not what the comparison assumes (wrong cell, units, or a rate limit)."""


def cell_key(glat, glon):
    return f"{glat:.4f}_{glon:.4f}"


def url(lat, lon, start, end, hourly=IFS_VARS):
    return (f"{W.ARCHIVE_URL}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
            f"&hourly={','.join(hourly)}&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
            f"&timezone=GMT&models=ecmwf_ifs&cell_selection=nearest&elevation=nan")


def _get(u, attempts=4):
    """One request; a rate limit stops the run, transient failures (empty body, 5xx, network) are retried."""
    for i in range(1, attempts + 1):
        try:
            r = requests.get(u, timeout=180)
            if r.status_code == 429:
                raise IFSError(f"Open-Meteo rate limit: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except IFSError:
            raise
        except (requests.exceptions.RequestException, ValueError) as e:
            if i == attempts:
                raise
            print(f"transient failure ({e!r}); retry {i}/{attempts - 1} in {30 * i} s", flush=True)
            time.sleep(30 * i)


def discover(pause_s=0.2):
    """Each zip's IFS grid cell, from a one-day request at its centroid."""
    p = pd.read_parquet(PANEL, columns=["city", "zipcode", "zip_lat", "zip_lon", "cell_id"]).drop_duplicates("zipcode")
    rows = []
    for r in p.itertuples():
        j = _get(url(r.zip_lat, r.zip_lon, "2020-06-01", "2020-06-01", ["temperature_2m"]))
        rows.append({"city": r.city, "zipcode": r.zipcode, "zip_lat": r.zip_lat, "zip_lon": r.zip_lon,
                     "era5_cell_id": r.cell_id, "ifs_lat": round(j["latitude"], 4), "ifs_lon": round(j["longitude"], 4),
                     "ifs_elevation_m": j.get("elevation")})
        time.sleep(pause_s)
    out = pd.DataFrame(rows)
    out["ifs_cell"] = [cell_key(a, b) for a, b in zip(out.ifs_lat, out.ifs_lon)]
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(LOOKUP, index=False)
    return out


def path(cell, year):
    return os.path.join(IFS_DIR, f"{cell}_{year}.json.gz")


def check(data, glat, glon):
    if "hourly" not in data:
        raise IFSError(f"no hourly data for IFS cell {cell_key(glat, glon)}: {str(data)[:200]}")
    if abs(data["latitude"] - glat) > 1e-3 or abs(data["longitude"] - glon) > 1e-3:
        raise IFSError(f"asked for IFS cell {cell_key(glat, glon)}, got ({data['latitude']}, {data['longitude']})")
    bad = {v: data.get("hourly_units", {}).get(v) for v, u in UNITS.items()
           if data.get("hourly_units", {}).get(v) != u}
    if bad:
        raise IFSError(f"unexpected units for IFS cell {cell_key(glat, glon)}: {bad}")


def fetch(pause_s=25.0):
    """Every IFS cell-year (cached). A rate limit stops the run; rerun later to continue."""
    lk = pd.read_csv(LOOKUP, dtype={"zipcode": str})
    cells = lk.drop_duplicates("ifs_cell")[["ifs_cell", "ifs_lat", "ifs_lon"]]
    os.makedirs(IFS_DIR, exist_ok=True)
    todo = [(c, la, lo, y) for c, la, lo in cells.itertuples(index=False) for y in YEARS if not os.path.exists(path(c, y))]
    print(f"{len(todo)} cell-years to fetch", flush=True)
    for i, (c, la, lo, y) in enumerate(todo, 1):
        s, e = f"{y}-01-01", min(f"{y}-12-31", config.FETCH_END)
        u = url(la, lo, s, e)
        data = _get(u)
        check(data, la, lo)
        tmp = path(c, y) + ".part"
        with gzip.open(tmp, "wt") as f:
            json.dump({"url": u, "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "data": data}, f)
        os.replace(tmp, path(c, y))
        print(f"{i}/{len(todo)} {c} {y}", flush=True)
        time.sleep(pause_s)


def load_cell(cell):
    parts = []
    for y in YEARS:
        with gzip.open(path(cell, y), "rt") as f:
            rec = json.load(f)
        h = pd.DataFrame(rec["data"]["hourly"])
        h.index = pd.to_datetime(h.pop("time"))
        parts.append(h.apply(pd.to_numeric, errors="coerce"))
    h = pd.concat(parts).sort_index()
    if h.index.duplicated().any():
        raise IFSError(f"duplicate hours for IFS cell {cell}")
    return h[IFS_VARS]


def complete_blocks(h, start, end, tz):
    """Runs of consecutive local days whose hours are all present (23-25) with no missing value."""
    local = h.index.tz_localize("UTC").tz_convert(tz)
    day = pd.Series(local.strftime("%Y-%m-%d"), index=h.index)
    keep = (day >= start) & (day <= end)
    ok = h.loc[keep, W.HOURLY_VARS].notna().all(axis=1).groupby(day[keep]).all()
    n = day[keep].groupby(day[keep]).size()
    good = sorted(set(ok[ok].index) & set(n[(n >= 23) & (n <= 25)].index))
    blocks, run = [], []
    for d in good:
        if run and (pd.Timestamp(d) - pd.Timestamp(run[-1])).days != 1:
            blocks.append((run[0], run[-1]))
            run = []
        run.append(d)
    if run:
        blocks.append((run[0], run[-1]))
    return blocks


def daily_complete_only(h, start, end, tz, where):
    """The main panel's daily recipe on complete local days only; incomplete days are left out, never filled."""
    parts = [W.daily_from_hourly(h, a, b, tz, where=where) for a, b in complete_blocks(h, start, end, tz)]
    return pd.concat(parts) if parts else pd.DataFrame()


def build():
    lk = pd.read_csv(LOOKUP, dtype={"zipcode": str})
    era5 = pd.read_parquet(HOURLY_ERA5, columns=["cell_id", "time_utc", "boundary_layer_height"])
    era5["time_utc"] = pd.to_datetime(era5.time_utc)
    panel = pd.read_parquet(PANEL)
    start_utc = pd.Timestamp(f"{config.START_DATE} 00:00")
    end_utc = pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1, hours=23)
    idx = pd.date_range(start_utc, end_utc, freq="h")
    cache, daily = {}, []
    for r in lk.itertuples():
        tz = config.CITIES[r.city]["tz"]
        if r.ifs_cell not in cache:
            cache[r.ifs_cell] = load_cell(r.ifs_cell)
        h = cache[r.ifs_cell].reindex(idx)
        blh = era5[era5.cell_id == r.era5_cell_id].set_index("time_utc").boundary_layer_height
        h["boundary_layer_height"] = blh.reindex(idx)            # Copernicus ERA5 BLH, the main panel's source
        d = daily_complete_only(h, config.START_DATE, config.END_DATE, tz, where=f"zip {r.zipcode}")
        d["pressure_anomaly"] = d.atmospheric_pressure - d.atmospheric_pressure.mean()
        sources = config.CITIES[r.city]["sources"]
        d["wind_align_weighted"] = [core.compute_multisource_terms(r.zip_lat, r.zip_lon, w, sources=sources)[1]
                                    for w in d.wind_direction]
        daily.append(d[WEATHER_COLS].assign(zipcode=r.zipcode).rename_axis("date").reset_index())
    daily = pd.concat(daily, ignore_index=True)
    base = panel.drop(columns=[c for c in WEATHER_COLS if c in panel.columns])
    out = base.merge(daily, on=["zipcode", "date"], how="inner", validate="one_to_one")
    left_out = len(panel) - len(out)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(os.path.join(OUT_DIR, "zip_day_panel_ifs.parquet"), index=False)
    prov = {"built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": len(out), "main_panel_rows": len(panel), "zip_days_left_out_incomplete_ifs": int(left_out),
            "ifs_cells": {c: int(n) for c, n in lk.groupby("city").ifs_cell.nunique().items()},
            "sources": {"ifs": "Open-Meteo archive models=ecmwf_ifs, cell_selection=nearest, elevation=nan",
                        "boundary_layer_height": "Copernicus ERA5 (main panel weather_cell_hourly.parquet)"}}
    json.dump(prov, open(os.path.join(OUT_DIR, "provenance.json"), "w"), indent=1)
    print(json.dumps(prov, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["discover", "fetch", "build"])
    ap.add_argument("--pause", type=float, default=25.0)
    a = ap.parse_args(argv)
    if a.step == "discover":
        print(discover().groupby("city").ifs_cell.nunique())
    elif a.step == "fetch":
        try:
            fetch(a.pause)
        except IFSError as e:
            print(f"stopped: {e}. Cached files are kept; rerun later to continue.", file=sys.stderr)
            sys.exit(3)
    else:
        build()


if __name__ == "__main__":
    main()
