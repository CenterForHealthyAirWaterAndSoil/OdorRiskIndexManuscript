"""Prior-year emissions sensitivity (decision D5, 2026-10-26; DEVIATIONS item 20).

What a forecaster could have used: each date's emissions are the previous year's TRI (TRI 2022 for 2023 dates,
2023 for 2024 dates, 2024 for 2025 dates). Same method as sensitivity_tri_yearmatched.py: each facility x release
x species series in a run's con2stn.txt is rescaled to the prior-year TRI rate, with no HYSPLIT re-run.
A rescale can only scale sources that were run (their 2024 TRI rate is nonzero); a source with air releases in an
earlier year but none in 2024 (Arkema trichloroethylene, 830 lb in 2022) cannot be added this way.

Writes predicted_tri_prioryear/<date>.csv (+ .DONE) for every date in predicted_clipped/. With --score, scores the
dates in DATES_FILE with run_plume.py's own code, unchanged except the input folder and output names
-> plume_results_tri_prioryear.json. Score only once the date set is final (see DEVIATIONS 16 and 20).
"""
import glob, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensitivity_tri_yearmatched as YM

OUT = os.path.join(HERE, "predicted_tri_prioryear")


def main(score=False):
    os.makedirs(OUT, exist_ok=True)
    rates = {y: YM.tri_rates(y) for y in (2022, 2023, 2024)}
    dates = sorted(os.path.basename(p)[:10] for p in glob.glob(os.path.join(YM.PRED, "*.DONE")))
    worst = 0.0
    for d in dates:
        orig = pd.read_csv(os.path.join(YM.PRED, f"{d}.csv"))
        same = YM.predict(d, rates[2024], use_factor=False).merge(orig, on=["date", "site", "species"], how="right", suffixes=("", "_orig"))
        worst = max(worst, float((same.pred_mean_ugm3.fillna(0) - same.pred_mean_ugm3_orig).abs().max()))
        py = YM.predict(d, rates[int(d[:4]) - 1])
        py = orig[["date", "site", "species"]].merge(py, on=["date", "site", "species"], how="left").fillna({"pred_mean_ugm3": 0.0})
        py.to_csv(os.path.join(OUT, f"{d}.csv"), index=False)
        open(os.path.join(OUT, f"{d}.DONE"), "w").close()
    print(f"{len(dates)} dates rescaled; reproduction check (factor 1) max |diff| = {worst:.2e} ug/m3")
    assert worst < 1e-6, "rescaling with factor 1 does not reproduce predicted_clipped"
    if not score:
        return
    src = open(os.path.join(HERE, "run_plume.py")).read()
    src = src.replace('PRED_DIR = "/path/to/plume-validation/validation/results/predicted_clipped"', f'PRED_DIR = "{OUT}"')
    src = src.replace('"plume_results.json"', '"plume_results_tri_prioryear.json"').replace('"plume_monitor_days.csv"', '"plume_monitor_days_tri_prioryear.csv"')
    assert f'PRED_DIR = "{OUT}"' in src and "plume_results_tri_prioryear.json" in src
    exec(compile(src, "run_plume.py (TRI prior-year inputs)", "exec"), {"__file__": os.path.join(HERE, "run_plume.py"), "__name__": "__main__"})


if __name__ == "__main__":
    main(score="--score" in sys.argv)
