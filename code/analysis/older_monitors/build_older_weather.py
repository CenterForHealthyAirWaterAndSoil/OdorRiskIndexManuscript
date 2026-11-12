"""NOV2026_PLAN.md part C, weather: the Calvert ERA5 cell (37.00, -88.25; every older monitor is in it) for local days
2011-01-01..2022-12-31, built exactly as refit_calvert_inputs.cell_panel_style() builds refit_inputs/era5_cell_daily.csv.

* 2011-01-01..2017-12-31: new retrievals (fetch_era5_older.py; same two sources: Copernicus ERA5 time series for all
  variables except rain, Open-Meteo models=era5 for rain), through training/era5_cells.cell_hourly / cell_daily.
* 2018-01-01..2022-12-31: the existing refit_inputs files (same sources), unchanged.
* The two retrievals are checked against each other on the overlap 2018-01-01..2018-03-31 (hourly and daily).
* pressure_anomaly = surface pressure minus 1002.1278805398465 hPa, the cell's 2018-01-01..2026-06-05 mean used in
  current scoring (refit_inputs/era5_cell_provenance.json), so the models see the same scale.
Outputs: older_weather_hourly.parquet (UTC hours, W.HOURLY_VARS), older_weather_daily.csv, weather_check.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
REPO = "/path/to/weather-varaible-analysis"
sys.path.insert(0, REPO)
import weather_history as W  # noqa: E402
from training import config, era5_cells as E  # noqa: E402

E.CDS_DIR = os.path.join(HERE, "raw", "cds_era5")
E.OM_DIR = os.path.join(HERE, "raw", "open_meteo_era5")
TZ = "America/Chicago"
GLAT, GLON = 37.0, -88.25
P_MEAN = json.load(open(os.path.join(EXT, "refit_inputs", "era5_cell_provenance.json")))["pressure_mean_hpa_2018_2026"]


def main():
    years = range(2010, 2019)
    h, meta = E.cell_hourly("calvert", GLAT, GLON, years, "2010-12-31 00:00", "2018-04-01 23:00")
    d = E.cell_daily(h, "2011-01-01", "2018-03-31", TZ, where="calvert cell 2011-2018 (older monitors)")
    d["pressure_anomaly"] = d.atmospheric_pressure - P_MEAN

    old_h = pd.read_parquet(os.path.join(EXT, "refit_inputs", "era5_cell_hourly.parquet"))
    old_h.index = pd.to_datetime(old_h.index)
    old_d = pd.read_csv(os.path.join(EXT, "refit_inputs", "era5_cell_daily.csv"), index_col="date")
    # ---- overlap check: the new retrieval against the existing panel-style files
    ov_h = slice("2018-01-01 00:00", "2018-04-01 23:00")
    a, b = h.loc[ov_h, W.HOURLY_VARS], old_h.loc[ov_h, W.HOURLY_VARS]
    chk = {"hourly_overlap_hours": int(len(a)),
           "hourly_max_abs_diff": {v: float((a[v] - b[v]).abs().max()) for v in W.HOURLY_VARS}}
    ov = [x for x in d.index if "2018-01-01" <= x <= "2018-03-31"]
    cols = [c for c in old_d.columns if c in d.columns]
    chk["daily_overlap_days"] = len(ov)
    chk["daily_max_abs_diff"] = {c: float((d.loc[ov, c] - old_d.loc[ov, c]).abs().max()) for c in cols}
    chk["pressure_mean_used_hpa"] = P_MEAN
    worst = max(max(chk["hourly_max_abs_diff"].values()), max(chk["daily_max_abs_diff"].values()))
    chk["exact"] = bool(worst < 1e-9)
    print(json.dumps(chk, indent=1))
    if worst > 1e-6:
        raise SystemExit("new retrieval does not reproduce refit_inputs on the overlap")

    # ---- combine: new before 2018, existing from 2018
    hh = pd.concat([h.loc[: "2017-12-31 23:00", W.HOURLY_VARS], old_h.loc["2018-01-01 00:00": "2023-01-01 12:00", W.HOURLY_VARS]])
    assert not hh.index.duplicated().any()
    dd = pd.concat([d.loc[[x for x in d.index if x < "2018-01-01"]],
                    old_d.loc[[x for x in old_d.index if "2018-01-01" <= x <= "2022-12-31"]]])
    dd["pressure_anomaly"] = dd.atmospheric_pressure - P_MEAN   # same constant on every date
    gap = pd.date_range("2011-01-01", "2022-12-31").strftime("%Y-%m-%d").difference(dd.index)
    chk["missing_local_days_2011_2022"] = list(gap)
    chk["days"] = int(len(dd))
    chk["provenance_new"] = meta
    hh.rename_axis("time_utc").to_parquet(os.path.join(HERE, "older_weather_hourly.parquet"))
    dd.rename_axis("date").to_csv(os.path.join(HERE, "older_weather_daily.csv"))
    json.dump(chk, open(os.path.join(HERE, "weather_check.json"), "w"), indent=1, default=str)
    print("days", len(dd), "missing", len(gap))


if __name__ == "__main__":
    main()
