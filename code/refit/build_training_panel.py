"""Rebuild the Pittsburgh + Louisville zip-day training panels for the ORI refit, from the raw files only.

Inputs: the raw Smell PGH / Smell MyCity exports and cached ERA5 weather (training/era5_cells.py fetches it).
It never reads the old merged panels (open-meteo-smell-merged.csv) and never imputes: any missing hour, or a
local day without 23-25 hours, raises.

    python -m training.era5_cells fetch-open-meteo && python -m training.era5_cells fetch-cds
    python -m training.build_training_panel

Outputs in training_panel_v2/: zip_day_panel.{parquet,csv}, cell_lookup.csv, provenance.json +
provenance_requests.csv, DATA_DICTIONARY.md, and review intermediates (weather_cell_hourly.parquet,
reports_zip_hour.parquet, reports_kept.parquet).
"""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import config, era5_cells as E, panel as P, reports as R  # noqa: E402

YEARS = range(int(config.START_DATE[:4]), int(config.FETCH_END[:4]) + 1)
START_UTC = f"{config.START_DATE} 00:00"
END_UTC = (pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 23:00")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def city_zips(city, reports):
    """All zips with coordinates, their grid cell, and the original-set flag."""
    z = R.zip_centroids(reports).reset_index()
    cells = [E.grid_cell(a, b) for a, b in zip(z.lat, z.lon)]
    z["cell_lat"], z["cell_lon"] = [c[0] for c in cells], [c[1] for c in cells]
    z["cell_id"] = [E.cell_id(*c) for c in cells]
    z["in_original_set"] = z.zipcode.isin(config.PGH_ORIGINAL_ZIPS) if city == "pgh" else True
    if city == "pgh" and z.in_original_set.sum() != len(config.PGH_ORIGINAL_ZIPS):
        raise P.PanelError("some of the original 61 Pittsburgh zips are missing from the raw reports")
    return z


def build_city(city):
    cfg = config.CITIES[city]
    dates = pd.date_range(config.START_DATE, config.END_DATE).strftime("%Y-%m-%d").tolist()
    reports = R.load_reports(cfg["reports"], cfg["tz"])
    zips = city_zips(city, reports)
    kept, dropped = R.select_reports(reports, set(zips.zipcode), config.START_DATE, config.END_DATE)
    zh = R.zip_hour_table(kept)
    targets = R.zip_day_targets(zh)

    daily, hourly, meta, checks = [], [], [], []
    for cid, glat, glon in zips[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates().itertuples(index=False):
        h, m = E.cell_hourly(city, glat, glon, YEARS, START_UTC, END_UTC)
        checks += [{"city": city, "cell_id": cid, **r} for r in E.check_sources(h)]
        d = E.cell_daily(h, config.START_DATE, config.END_DATE, cfg["tz"], where=f"{city} cell {cid}")
        daily.append(d.rename_axis("date").reset_index().assign(cell_id=cid))
        hh = h.rename_axis("time_utc").reset_index()
        hh["time_local"] = hh.time_utc.dt.tz_localize("UTC").dt.tz_convert(cfg["tz"]).dt.strftime("%Y-%m-%dT%H:%M%z")
        hourly.append(hh.assign(city=city, cell_id=cid))
        elev = {r["elevation_m"] for r in m if r.get("elevation_m") is not None}
        if len(elev) != 1:
            raise E.WeatherSourceError(f"{city} cell {cid}: Open-Meteo reported elevations {elev}")
        zips.loc[zips.cell_id == cid, "cell_elevation_m"] = elev.pop()
        meta += [dict(r, city=city) for r in m]
    cell_daily = pd.concat(daily, ignore_index=True)

    p = P.assemble_city_panel(city, zips, cell_daily, targets, cfg["sources"], dates,
                              sparse_before=config.LOU_SPARSE_BEFORE if city == "lou" else None)
    p, thresholds = P.add_events(p, reference=p.zip_in_original_set)
    info = {"raw_reports": {"path": os.path.relpath(cfg["reports"], config.REPO), "rows": len(reports),
                            "sha256": _sha256(cfg["reports"]), "first_local": str(reports.time_local.min()),
                            "last_local": str(reports.time_local.max())},
            "reports_kept": len(kept), "reports_dropped": dropped,
            "reports_kept_original_zips": int(kept.zipcode.isin(zips.zipcode[zips.in_original_set]).sum()),
            "zips": len(zips), "zips_original_set": int(zips.in_original_set.sum()), "cells": int(zips.cell_id.nunique()),
            "zip_days": len(p), "event_thresholds_original_zips": thresholds, "tz": cfg["tz"]}
    kept_out = kept.assign(city=city, time_local=kept.time_local.dt.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return p, zips.assign(city=city), pd.concat(hourly), zh.assign(city=city), kept_out, meta, checks, info


def write_data_dictionary(path, thresholds):
    lines = ["# Training panel v2: data dictionary", "",
             "One row per zip x local calendar day, 2018-01-01 to 2026-06-05, for every zip with coordinates in the raw",
             "reports (Pittsburgh 107, Louisville 36). Built by `training/build_training_panel.py`; see provenance.json.", "",
             "**Weather sources, one per variable for every date.** Copernicus ERA5 (`reanalysis-era5-single-levels-timeseries`,",
             "equal to the standard gridded ERA5): temperature, RH (from dew point), surface pressure, wind speed and",
             "direction (from u/v), solar radiation and boundary-layer height (metres x 3.28084). Open-Meteo's ERA5",
             "archive (`models=era5`, at the 0.25-degree cell with `cell_selection=nearest&elevation=nan`): rain only.",
             "Zips in one grid cell share one weather series; nothing is downscaled to a zip's elevation.", "",
             "**Days** are local calendar days built from UTC hours (23, 24 or 25 hours). Nothing is imputed.", "",
             "| column | unit | meaning |", "|---|---|---|"]
    lines += [f"| `{c}` | {u} | {m} |" for c, u, m in P.COLUMNS]
    lines += ["", "**Event thresholds** (mean burden over each city's `zip_in_original_set` rows, all dates):", ""]
    for city, thr in thresholds.items():
        lines.append(f"- {city}: " + ", ".join(f"`{k}` > {v:.4f}" for k, v in thr.items()))
    lines += ["", "The fit can recompute events on another population (e.g. report days only, or all zips) from",
              "the burden columns.", "",
              "## Intermediates", "",
              "- `weather_cell_hourly.parquet`: one row per city x grid cell x UTC hour: the panel's hourly inputs",
              "  (weather_history names and units, BLH in feet), Open-Meteo's ERA5 value of every variable as `om_*`",
              "  and ERA5 total precipitation as `cds_total_precipitation` (both for the source check only), and `time_local`.",
              "- `reports_zip_hour.parquet`: one row per zip x local wall-clock hour with reports (`hour_local`),",
              "  with count, mean, max, sum, value-1 count and the mean without value-1 reports.",
              "- `reports_kept.parquet`: every raw report that entered the panel, with its local time and date.",
              "- `cell_lookup.csv`: zip -> grid cell, with centroid, report count and the original-set flag."]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(config.OUT_DIR, exist_ok=True)
    parts = {c: build_city(c) for c in config.CITIES}
    panel = P.order_columns(pd.concat([parts[c][0] for c in parts], ignore_index=True))
    if panel.duplicated(["city", "zipcode", "date"]).any():
        raise P.PanelError("duplicate city x zip x date rows")
    out = lambda name: os.path.join(config.OUT_DIR, name)
    panel.to_parquet(out("zip_day_panel.parquet"), index=False)
    panel.to_csv(out("zip_day_panel.csv"), index=False)

    lookup = pd.concat([parts[c][1] for c in parts], ignore_index=True)
    lookup = lookup[["city", "zipcode", "lat", "lon", "n_reports_all", "in_original_set", "cell_id", "cell_lat",
                     "cell_lon", "cell_elevation_m"]].rename(columns={"lat": "zip_lat", "lon": "zip_lon"})
    lookup.to_csv(out("cell_lookup.csv"), index=False)
    pd.concat([parts[c][2] for c in parts], ignore_index=True).to_parquet(out("weather_cell_hourly.parquet"), index=False)
    pd.concat([parts[c][3] for c in parts], ignore_index=True).to_parquet(out("reports_zip_hour.parquet"), index=False)
    pd.concat([parts[c][4] for c in parts], ignore_index=True).to_parquet(out("reports_kept.parquet"), index=False)

    requests = pd.DataFrame([r for c in parts for r in parts[c][5]])
    requests.to_csv(out("provenance_requests.csv"), index=False)
    checks = pd.DataFrame([r for c in parts for r in parts[c][6]])
    checks.to_csv(out("source_check.csv"), index=False)
    commit = subprocess.run(["git", "-C", config.REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", config.REPO, "status", "--porcelain", "--", "training", "weather_history.py"],
                           capture_output=True, text=True).stdout.strip()
    prov = {"built_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "script": "training/build_training_panel.py", "git_commit": commit, "uncommitted_changes": bool(dirty),
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "window_local_days": [config.START_DATE, config.END_DATE],
            "weather_sources": {v: ("Copernicus CDS reanalysis-era5-single-levels-timeseries" if src == "copernicus"
                                    else "Open-Meteo archive models=era5, cell_selection=nearest, elevation=nan")
                                for v, src in E.PANEL_SOURCES.items()},
            "requests": {"open_meteo": int((requests.source.str.startswith("open-meteo")).sum()),
                         "copernicus": int(requests[requests.source.str.startswith("copernicus")].cell_id.nunique())},
            "rows": {"zip_day_panel": len(panel), **{c: parts[c][7]["zip_days"] for c in parts}},
            "cities": {c: parts[c][7] for c in parts},
            "source_check": "source_check.csv"}
    json.dump(prov, open(out("provenance.json"), "w"), indent=1, default=str)
    write_data_dictionary(out("DATA_DICTIONARY.md"), {c: parts[c][7]["event_thresholds_original_zips"] for c in parts})
    print(f"panel: {len(panel):,} zip-days -> {out('zip_day_panel.parquet')}")
    for c in parts:
        i = parts[c][7]
        print(f"  {c}: {i['zips']} zips ({i['zips_original_set']} original), {i['cells']} cells, {i['zip_days']:,} zip-days, "
              f"{i['reports_kept']:,} reports kept, dropped {i['reports_dropped']}")


if __name__ == "__main__":
    main()
