"""Mean ERA5 surface pressure of each live-site grid cell, the reference for the refit's pressure anomaly.

The refit was fit on pressure_anomaly = surface pressure minus each location's own ERA5-cell mean over the training
window (REFIT_PLAN.md s7). The live site needs the same reference for the seven 0.25-degree cells its 32 tracts
fall in. Each mean is taken over complete local days (America/Chicago), 2018-01-01 to 2026-06-05, from the same
Copernicus ERA5 time series the training panel uses, then written to training_panel_v2/live_cell_pressure.json.

Run: .venv/bin/python -m training.live_cell_pressure
"""
import json
import os

import pandas as pd

import odor_forecast_core as core
from training import config
from training import era5_cells as E

TZ = "America/Chicago"
OUT = os.path.join(config.OUT_DIR, "live_cell_pressure.json")


def live_cells():
    return sorted({(round(lat * 4) / 4, round(lon * 4) / 4) for lat, lon in (v["coords"] for v in core.LOCATIONS.values())})


def cell_mean_pressure(glat, glon):
    E.fetch_cds_cell(glat, glon)
    h, _ = E.load_cds_cell(glat, glon)
    h = h[["surface_pressure"]].dropna()
    local = h.index.tz_localize("UTC").tz_convert(TZ)
    day = pd.Series(local.strftime("%Y-%m-%d"), index=h.index)
    keep = (day >= config.START_DATE) & (day <= config.END_DATE)
    h, day = h[keep], day[keep]
    n = day.groupby(day).size()
    full = n[(n >= 23) & (n <= 25)].index
    daily = h.groupby(day).surface_pressure.mean().loc[full]
    return float(daily.mean()), int(len(daily))


def main():
    out = {}
    for glat, glon in live_cells():
        mean, days = cell_mean_pressure(glat, glon)
        out[E.cell_id(glat, glon)] = {"lat": glat, "lon": glon, "mean_surface_pressure_hpa": round(mean, 3),
                                      "complete_days": days}
        print(E.cell_id(glat, glon), round(mean, 3), days, flush=True)
    with open(OUT, "w") as f:
        json.dump({"window": [config.START_DATE, config.END_DATE], "tz": TZ,
                   "source": "Copernicus reanalysis-era5-single-levels-timeseries", "cells": out}, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
