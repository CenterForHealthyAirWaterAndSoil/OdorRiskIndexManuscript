"""NOV2026_PLAN.md part C, scores (input building only: no measurement is read here except the list of sample dates).

For every older monitor and every date in the union of their sample dates (forecasts at every monitor on every date,
as the plan's joint date permutation needs), from older_weather_daily.csv / older_weather_hourly.parquet:
  f_deployed_on_era5  deployed model through predict_ori (score_refit.deployed_pct), primary score
  lp_refit, f_refit   refit linear predictor and probability with the Calvert constant (score_refit.main())
  TERMS               the refit's 11 inputs (temp_c, temp_c2 built with the refit's temperature centre)
  hours_downwind      share of local-day hours with ERA5 10 m wind within 30 degrees of the bearing from the complex
                      centre (M.IND_LAT, M.IND_LON) to the monitor (score_refit.comparator_indices definition)
The same columns for the three current monitors, 2023-2025, from refit_inputs/ (current_scores_era5.csv), checked
against refit_calvert_monitor_days.csv (score against score).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
sys.path.insert(0, EXT)
import score_refit as S  # noqa: E402
M = S.M
TZ = "America/Chicago"


def hours_downwind(h, lat, lon):
    """score_refit.comparator_indices' 'hours downwind of complex (wind-direction analysis)', alone."""
    loc = h.index.tz_localize("UTC").tz_convert(TZ)
    date = pd.Series(loc.strftime("%Y-%m-%d"), index=h.index)
    brg = M._source_bearing(M.IND_LAT, M.IND_LON, lat, lon)
    diff = ((h.wind_direction_10m - (brg + 180) % 360 + 180) % 360) - 180
    dw = (diff.abs() <= 30).astype(float)
    n = date.groupby(date).size()
    ok = h.wind_direction_10m.notna().groupby(date).all() & (n >= 23) & (n <= 25)
    out = dw.groupby(date).mean()
    return out[ok[ok].index]


def score_block(daily, hourly, lat, lon, b, t0, const, fits):
    g = S.with_source_terms(daily.copy(), lat, lon)
    g["temp_c"] = g.temperature - t0
    g["temp_c2"] = (g.temperature - t0) ** 2
    g["lp_refit"] = S.refit_lp(g, b, t0)
    g["f_refit"] = 100 * expit(const + g.lp_refit)
    g["lp_pgh_only"] = S.refit_lp(g, fits["city_pgh_only"], t0)
    g["lp_lou_only"] = S.refit_lp(g, fits["city_lou_only"], t0)
    g["f_deployed_on_era5"] = S.deployed_pct(g)
    hd = hours_downwind(hourly, lat, lon)
    g["hours_downwind"] = hd.reindex(g.date).values
    return g


def main():
    fits, t0, const, _ = S.load_refit()
    b = fits["main"]
    cols = ["date", "f_deployed_on_era5", "lp_refit", "f_refit", "lp_pgh_only", "lp_lou_only", "hours_downwind"] + S.TERMS

    # ---- older monitors
    inv = pd.read_csv(os.path.join(HERE, "monitor_inventory.csv"), dtype={"site_id": str})
    md = pd.read_csv(os.path.join(HERE, "older_monitor_days.csv"), usecols=["date", "site_id"])
    dates = sorted(md.date.unique())
    wd = pd.read_csv(os.path.join(HERE, "older_weather_daily.csv"))
    wh = pd.read_parquet(os.path.join(HERE, "older_weather_hourly.parquet"))
    wh.index = pd.to_datetime(wh.index)
    missing = sorted(set(dates) - set(wd.date))
    if missing:
        raise SystemExit(f"no complete ERA5 day for sample dates {missing[:10]} ({len(missing)})")
    day = wd[wd.date.isin(dates)]
    rows = []
    for _, r in inv.iterrows():
        g = score_block(day, wh, r.lat, r.lon, b, t0, const, fits)
        rows.append(g[cols].assign(site_id=r.site_id, name=r["name"]))
    out = pd.concat(rows)
    out["sampled"] = out.set_index(["date", "site_id"]).index.isin(md.set_index(["date", "site_id"]).index)
    if out[["f_deployed_on_era5", "f_refit", "hours_downwind"]].isna().any().any():
        raise SystemExit("a score is missing")
    out[["date", "site_id", "name", "sampled"] + cols[1:]].to_csv(os.path.join(HERE, "older_scores.csv"), index=False)
    print("older_scores", out.shape, "sampled", int(out.sampled.sum()))

    # ---- current monitors 2023-2025 (check against refit_calvert_monitor_days.csv)
    cd = pd.read_csv(os.path.join(EXT, "refit_inputs", "era5_cell_daily.csv"))
    cd = cd[(cd.date >= "2023-01-01") & (cd.date <= "2025-12-31")]
    ch = pd.read_parquet(os.path.join(EXT, "refit_inputs", "era5_cell_hourly.parquet"))
    ch.index = pd.to_datetime(ch.index)
    comp = pd.read_parquet(os.path.join(EXT, "refit_inputs", "era5_comparator_hourly.parquet"))
    comp.index = pd.to_datetime(comp.index)
    common = ch.index.intersection(comp.index)
    wdir_diff = float((ch.loc[common, "wind_direction_10m"] - comp.loc[common, "wind_direction_10m"]).abs().max())
    rows = []
    for s, (lat, lon) in S.SITES.items():
        g = score_block(cd, ch, lat, lon, b, t0, const, fits)
        rows.append(g[cols].assign(site=s))
    cur = pd.concat(rows)
    cur[["date", "site"] + cols[1:]].to_csv(os.path.join(HERE, "current_scores_era5.csv"), index=False)
    ref = pd.read_csv(os.path.join(EXT, "refit_calvert_monitor_days.csv"))
    m = ref.merge(cur, on=["date", "site"], suffixes=("_ref", ""))
    chk = {"monitor_days_compared": int(len(m)),
           "wind_direction_cell_vs_comparator_hourly_max_abs_diff_deg": wdir_diff,
           "hours_downwind_max_abs_diff": float((m["hours downwind of complex (wind-direction analysis)"] - m.hours_downwind).abs().max()),
           "f_refit_max_abs_diff": float((m.f_refit_ref - m.f_refit).abs().max()),
           "lp_refit_max_abs_diff": float((m.lp_refit_ref - m.lp_refit).abs().max()),
           "f_deployed_on_era5_max_abs_diff": float((m.f_deployed_on_era5_ref - m.f_deployed_on_era5).abs().max()),
           "multi_source_exposure_max_abs_diff": float((m.multi_source_exposure_ref - m.multi_source_exposure).abs().max())}
    print(json.dumps(chk, indent=1))
    json.dump(chk, open(os.path.join(HERE, "scores_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
