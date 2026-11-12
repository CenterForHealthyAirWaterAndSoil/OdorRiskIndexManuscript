"""All 2023-2025 dates at the two residential monitors, including the poster's Jan-Jun 2025 days (DEVIATIONS 39).

Same statistics as items 26-37 (score_refit.py), corrected total VOC (item 38), seed 20261104.
Outputs: all_dates_results.json, ALL_DATES_RESULTS.md, all_dates_monitor_days.csv.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import score_refit as S
import score_nov2026_ab as AB

HERE = os.path.dirname(os.path.abspath(__file__))
S.rng = np.random.default_rng(20261104)
RES = S.RES
HD = "hours downwind of complex (wind-direction analysis)"
POLL = ["vc", "benzene", "total_voc"]
HYS = {"vc": "p_vinyl_chloride", "benzene": "p_benzene", "edc": "p_edc", "butadiene": "p_butadiene", "total_voc": "pred_sum7"}


def exact(d, col, y):
    g = d.dropna(subset=[col, y])
    return float(np.mean([spearmanr(x[col], x[y])[0] for _, x in g.groupby("site")]))


def main():
    fits, t0, _, _ = S.load_refit()
    d = pd.read_csv(os.path.join(HERE, "refit_calvert_monitor_days.csv"))
    d = d[d.site.isin(RES)].copy()
    tf = pd.read_csv(os.path.join(HERE, "total_voc_fixed.csv"))[["date", "site", "total_voc_fixed"]]
    d = d.merge(tf, on=["date", "site"], how="left", validate="one_to_one")
    d["total_voc"] = d.total_voc_fixed
    hp = pd.read_csv(os.path.join(HERE, "plume_full", "plume_monitor_days_all189.csv"))
    d = d.merge(hp[["date", "site"] + list(HYS.values())], on=["date", "site"], how="left", validate="one_to_one")
    aq = pd.read_csv(os.path.join(HERE, "aqi_inputs", "aqi_daily.csv"))
    d = d.merge(aq[["date", "site", "a1_aqi"]], on=["date", "site"], how="left", validate="one_to_one")
    d["hd"] = d[HD]
    d["cal_year"] = d.date.str[:4]

    # Calvert-fitted model, leave one calendar year out (item 34 procedure), total VOC on the corrected definition
    X = AB.features(t0)
    d = d.drop(columns=[c for c in AB.TERMS if c in d.columns]).merge(X, on=["date", "site"], how="left", validate="one_to_one")
    for y in POLL:
        d[f"cf_{y}"] = np.nan
        for yr in AB.YEARS:
            sc, m, _ = AB.fit(d[d.cal_year != yr], y, AB.TERMS)
            te = d.cal_year == yr
            d.loc[te, f"cf_{y}"] = AB.predict(sc, m, d[te], AB.TERMS)

    ok = d.dropna(subset=["vc", "benzene"]).groupby("date").site.nunique()
    s = d[d.date.isin(ok[ok == 2].index)].copy()
    assert s.date.nunique() == 141 and len(s) == 282, (s.date.nunique(), len(s))
    assert s[["f_corrected", "f_asrun", "hd", "p_vinyl_chloride", "a1_aqi"]].notna().all().all()
    s["combo"] = AB.rank_combo(s, "f_corrected", "hd")
    s.to_csv(os.path.join(HERE, "all_dates_monitor_days.csv"), index=False)

    score_col = {"ori_corrected": lambda y: "f_corrected", "ori_as_run": lambda y: "f_asrun",
                 "hysplit": lambda y: HYS[y], "hours_downwind": lambda y: "hd", "combo": lambda y: "combo",
                 "calvert_fitted": lambda y: f"cf_{y}", "aqi_observed_paducah": lambda y: "a1_aqi"}
    out = {"sample": "141 dates, 282 residential monitor-days (all 2023-2025 dates with both monitors)", "seed": 20261104,
           "scores": {}, "exact_rho": {}, "paired": {}}
    for k, f in score_col.items():
        out["scores"][k] = {}
        for y in POLL:
            r = S.joint_date_perm(s.assign(forecast=s[f(y)]), y, 5000)
            out["scores"][k][y] = r
        print(k, {y: out["scores"][k][y]["rho"] for y in POLL}, flush=True)
    for y in ["vc", "edc", "benzene", "butadiene", "total_voc"]:
        out["exact_rho"][y] = {c: exact(s, c, y) for c in ["f_corrected", HYS[y], "hd", "combo", "a1_aqi"]
                               + ([f"cf_{y}"] if y in POLL else [])}
    pairs = [(y, "f_corrected", HYS[y], "ORI minus HYSPLIT") for y in ["vc", "edc", "benzene", "butadiene", "total_voc"]]
    pairs += [(y, "combo", HYS[y], "ORI + hours downwind minus HYSPLIT") for y in ["vc", "benzene"]]
    pairs += [(y, HYS[y], "hd", "HYSPLIT minus hours downwind") for y in ["vc", "benzene"]]
    pairs += [(y, "combo", "f_corrected", "ORI + hours downwind minus ORI") for y in ["vc", "benzene"]]
    pairs += [(y, "combo", "hd", "ORI + hours downwind minus hours downwind") for y in ["vc", "benzene"]]
    pairs += [(y, f"cf_{y}", "f_corrected", "Calvert-fitted minus ORI") for y in ["vc", "benzene"]]
    pairs += [(y, f"cf_{y}", "hd", "Calvert-fitted minus hours downwind") for y in ["vc", "benzene"]]
    pairs += [(y, "f_corrected", "a1_aqi", "ORI minus daily AQI (Paducah)") for y in ["vc", "benzene", "total_voc"]]
    for y, a, b, lab in pairs:
        r = S.paired(s, y, a, b)
        out["paired"][f"{lab} | {y}"] = {"a": a, "b": b, "a_rho": r["refit_rho"], "b_rho": r["comparator_rho"],
                                         "diff": r["diff"], "ci95": r["ci95"], "p_swap": r["p_swap"], "dates": r["shared_dates"]}
        print(lab, y, r["refit_rho"], r["comparator_rho"], r["diff"], r["ci95"], flush=True)
    json.dump(out, open(os.path.join(HERE, "all_dates_results.json"), "w"), indent=1, default=float)

    L = ["# All 2023-2025 dates, including the poster's Jan-Jun 2025 days (DEVIATIONS 39, post hoc)", "",
         "141 dates, 282 residential monitor-days. Mean within-monitor Spearman rho; joint date-permutation p (5,000).", "",
         "| score | vinyl chloride | benzene | total VOC |", "|---|---|---|---|"]
    for k in score_col:
        r = out["scores"][k]
        L.append(f"| {k} | " + " | ".join(f"{r[y]['rho']:.3f} (p {r[y]['p']:.4f})" for y in POLL) + " |")
    L += ["", "| comparison | pollutant | a | b | difference | 95% CI | p (swap) |", "|---|---|---|---|---|---|---|"]
    for key, r in out["paired"].items():
        lab, y = key.split(" | ")
        L.append(f"| {lab} | {y} | {r['a_rho']:.3f} | {r['b_rho']:.3f} | {r['diff']:+.3f} | "
                 f"{r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f} | {r['p_swap']:.4f} |")
    open(os.path.join(HERE, "ALL_DATES_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
