"""All 2023-2025 dates with the 2026 refit as the paper's ORI (DEVIATIONS 40).

Same sample and statistics as score_all_dates.py (DEVIATIONS 39): the 141 dates on which both community monitors have
valid vinyl chloride and benzene (282 monitor-days), mean within-monitor Spearman rho, joint date-permutation p, and
paired date-bootstrap CIs from score_refit.paired(). The only change is which ORI is "the ORI": the refit on ERA5
inputs (f_refit), and ORI + hours downwind is rebuilt from it. The earlier deployed model is kept as a comparator
(the paper's Table S1). Seed 20261111.

Outputs: all_dates_refit_results.json, ALL_DATES_REFIT_RESULTS.md.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import score_refit as S

HERE = os.path.dirname(os.path.abspath(__file__))
S.rng = np.random.default_rng(20261111)
HYS = {"vc": "p_vinyl_chloride", "benzene": "p_benzene", "edc": "p_edc", "butadiene": "p_butadiene", "total_voc": "pred_sum7"}
POLL = ["vc", "benzene", "total_voc"]
ALL = ["vc", "edc", "benzene", "butadiene", "total_voc"]


def exact(d, col, y):
    g = d.dropna(subset=[col, y])
    return float(np.mean([spearmanr(x[col], x[y])[0] for _, x in g.groupby("site")]))


def main():
    s = pd.read_csv(os.path.join(HERE, "all_dates_monitor_days.csv"))     # written by score_all_dates.py
    assert s.date.nunique() == 141 and len(s) == 282
    s["combo_deployed"] = s.combo
    s["combo"] = s.groupby("site")[["f_refit", "hd"]].rank(pct=True).mean(axis=1)
    # Calvert-fitted benzene and total VOC out-of-fold columns are rebuilt exactly as score_all_dates.py does
    import score_nov2026_ab as AB
    fits, t0, _, _ = S.load_refit()
    full = pd.read_csv(os.path.join(HERE, "refit_calvert_monitor_days.csv"))
    full = full[full.site.isin(S.RES)].copy()
    tf = pd.read_csv(os.path.join(HERE, "total_voc_fixed.csv"))[["date", "site", "total_voc_fixed"]]
    full = full.merge(tf, on=["date", "site"], how="left", validate="one_to_one")
    full["total_voc"] = full.total_voc_fixed
    full["cal_year"] = full.date.str[:4]
    X = AB.features(t0)
    full = full.drop(columns=[c for c in AB.TERMS if c in full.columns]).merge(X, on=["date", "site"], how="left",
                                                                                validate="one_to_one")
    for y in POLL:
        full[f"cf_{y}"] = np.nan
        for yr in AB.YEARS:
            sc, m, _ = AB.fit(full[full.cal_year != yr], y, AB.TERMS)
            te = full.cal_year == yr
            full.loc[te, f"cf_{y}"] = AB.predict(sc, m, full[te], AB.TERMS)
    s = s.drop(columns=[c for c in s.columns if c.startswith("cf_")]).merge(
        full[["date", "site"] + [f"cf_{y}" for y in POLL]], on=["date", "site"], how="left", validate="one_to_one")
    assert abs(exact(s, "cf_vc", "vc") - 0.6747488220699305) < 1e-9

    score_col = {"ori_refit": lambda y: "f_refit", "ori_earlier_version": lambda y: "f_corrected",
                 "hysplit": lambda y: HYS[y], "hours_downwind": lambda y: "hd", "combo": lambda y: "combo",
                 "calvert_fitted": lambda y: f"cf_{y}", "aqi_observed_paducah": lambda y: "a1_aqi"}
    out = {"sample": "141 dates, 282 community monitor-days (all 2023-2025 dates with both monitors)", "seed": 20261111,
           "ori": "2026 refit on ERA5 inputs (f_refit); combo = equal-weight within-monitor ranks of f_refit and hours downwind",
           "scores": {}, "exact_rho": {}, "paired": {}}
    for k, f in score_col.items():
        out["scores"][k] = {y: S.joint_date_perm(s.assign(forecast=s[f(y)]), y, 5000) for y in POLL}
        print(k, {y: out["scores"][k][y]["rho"] for y in POLL}, flush=True)
    for y in ALL:
        out["exact_rho"][y] = {c: exact(s, c, y) for c in ["f_refit", "f_corrected", HYS[y], "hd", "combo", "a1_aqi"]
                               + ([f"cf_{y}"] if y in POLL else [])}
    pairs = [(y, "f_refit", HYS[y], "ORI minus HYSPLIT") for y in ALL]
    pairs += [(y, "combo", HYS[y], "ORI + hours downwind minus HYSPLIT") for y in ["vc", "benzene"]]
    pairs += [(y, HYS[y], "hd", "HYSPLIT minus hours downwind") for y in ["vc", "benzene"]]
    pairs += [(y, "combo", "f_refit", "ORI + hours downwind minus ORI") for y in ["vc", "benzene"]]
    pairs += [(y, "combo", "hd", "ORI + hours downwind minus hours downwind") for y in ["vc", "benzene"]]
    pairs += [(y, f"cf_{y}", "f_refit", "Calvert-fitted minus ORI") for y in ["vc", "benzene"]]
    pairs += [(y, f"cf_{y}", "hd", "Calvert-fitted minus hours downwind") for y in ["vc", "benzene"]]
    pairs += [(y, "f_refit", "a1_aqi", "ORI minus daily AQI (Paducah)") for y in ["vc", "benzene", "total_voc"]]
    pairs += [(y, "f_refit", "f_corrected", "ORI refit minus earlier version") for y in ["vc", "benzene"]]
    for y, a, b, lab in pairs:
        r = S.paired(s, y, a, b)
        out["paired"][f"{lab} | {y}"] = {"a": a, "b": b, "a_rho": r["refit_rho"], "b_rho": r["comparator_rho"],
                                         "diff": r["diff"], "ci95": r["ci95"], "p_swap": r["p_swap"], "dates": r["shared_dates"]}
        print(lab, y, r["refit_rho"], r["comparator_rho"], r["diff"], r["ci95"], flush=True)
    json.dump(out, open(os.path.join(HERE, "all_dates_refit_results.json"), "w"), indent=1, default=float)

    L = ["# All 2023-2025 dates with the 2026 refit as the ORI (DEVIATIONS 40)", "",
         "141 dates, 282 community monitor-days. Mean within-monitor Spearman rho; joint date-permutation p (5,000).", "",
         "| score | vinyl chloride | benzene | total VOC |", "|---|---|---|---|"]
    for k in score_col:
        r = out["scores"][k]
        L.append(f"| {k} | " + " | ".join(f"{r[y]['rho']:.3f} (p {r[y]['p']:.4f})" for y in POLL) + " |")
    L += ["", "| comparison | pollutant | a | b | difference | 95% CI | p (swap) |", "|---|---|---|---|---|---|---|"]
    for key, r in out["paired"].items():
        lab, y = key.split(" | ")
        L.append(f"| {lab} | {y} | {r['a_rho']:.3f} | {r['b_rho']:.3f} | {r['diff']:+.3f} | "
                 f"{r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f} | {r['p_swap']:.4f} |")
    open(os.path.join(HERE, "ALL_DATES_REFIT_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
