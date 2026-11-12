"""Figure 2 data: calibration of the ORI in the training cities, by decile of out-of-fold prediction.

Since 2026-11-11 (DEVIATIONS 40) the paper's ORI is the 2026 refit. This script reads its out-of-fold predictions
(training_panel_v2/fit/oof_predictions_main.parquet: each calendar month predicted by a fit without that month, day of
week held at its average effect) and the refit's training panel (reports per zip-day), and writes, for each city, the
event rate and mean reports per zip-day by equal-count decile of the out-of-fold ORI. Unlike the earlier figure
(the deployed model on Pittsburgh citywide days, in-sample), nothing here was seen by the fit that scored it.

The earlier version of this script is kept as fig3_ori_calibration_deployed_2026-11-02.py.bak.

Output: 02_analysis_datasets/ori_oof_deciles.csv (read by 04_code/poster_figures/fig2_calibration_deciles.py)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle  # noqa: E402

import pandas as pd  # noqa: E402

WS = "/path/to/weather-varaible-analysis/training_panel_v2"
OUT = os.path.join(figstyle.PKG, "02_analysis_datasets", "ori_oof_deciles.csv")


def main():
    oof = pd.read_parquet(os.path.join(WS, "fit", "oof_predictions_main.parquet"))
    pan = pd.read_parquet(os.path.join(WS, "zip_day_panel.parquet"), columns=["city", "zipcode", "date", "n_reports"])
    d = oof.merge(pan, on=["city", "zipcode", "date"], how="left", validate="one_to_one")
    assert d.n_reports.notna().all() and len(d) == 424890, len(d)
    res = json.load(open(os.path.join(WS, "fit", "results.json")))
    conc = {r["city"]: r["zip_strat_c"] for r in res["main"]["oof"]} if isinstance(res["main"].get("oof"), list) else {}
    rows = []
    for city, g in d.groupby("city"):
        g = g.copy()
        g["decile"] = pd.qcut(g.p_oof.rank(method="first"), 10, labels=False) + 1
        t = g.groupby("decile").agg(n_zip_days=("event", "size"), pred_min_pct=("p_oof", "min"),
                                    pred_max_pct=("p_oof", "max"), pred_mean_pct=("p_oof", "mean"),
                                    mean_reports=("n_reports", "mean"), event_pct=("event", "mean")).reset_index()
        for c in ("pred_min_pct", "pred_max_pct", "pred_mean_pct", "event_pct"):
            t[c] = (100 * t[c]).round(2)
        t["mean_reports"] = t.mean_reports.round(3)
        t.insert(0, "city", {"pgh": "Pittsburgh", "lou": "Louisville"}[city])
        t["event_pct_all"] = round(100 * g.event.mean(), 2)
        t["n_zip_days_total"] = len(g)
        rows.append(t)
        print(f"{city}: n {len(g):,}, events {100 * g.event.mean():.1f}%; event % lowest decile {t.event_pct.iloc[0]}, "
              f"highest {t.event_pct.iloc[-1]}; predicted {t.pred_mean_pct.iloc[0]} to {t.pred_mean_pct.iloc[-1]}")
    out = pd.concat(rows)
    out.to_csv(OUT, index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
