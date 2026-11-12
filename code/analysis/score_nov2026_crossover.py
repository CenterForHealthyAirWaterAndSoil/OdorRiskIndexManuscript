"""POST HOC head-to-head (DEVIATIONS 37): which score ranks which pollutant best, on identical monitor-days.

(a) ORI (deployed, corrected) minus HYSPLIT (species-matched; 7-species sum for total VOC), five pollutants grouped in
    advance: made at the complex = vinyl chloride, EDC; many sources = benzene, 1,3-butadiene, total VOC.
(b) ORI + hours downwind (part A combo) minus HYSPLIT, vinyl chloride and benzene.
(c) HYSPLIT minus hours downwind, vinyl chloride and benzene.
Sample: the 115 dates / 230 residential monitor-days of items 26, 30, 32 and 34. Statistic: score_refit.paired()
(mean within-monitor rho difference, date-bootstrap 95% CI, within-date swap p; B = 10,000). Post hoc and unadjusted:
read the CIs, not the p-values, as the evidence. Seed 20261103.
Outputs: nov2026_crossover_results.json, NOV2026_CROSSOVER_RESULTS.md.
"""
import json
import os

import numpy as np
import pandas as pd

import score_refit as S

HERE = os.path.dirname(os.path.abspath(__file__))
S.rng = np.random.default_rng(20261103)
POLL = {"vc": ("vinyl chloride", "p_vinyl_chloride", "made at the complex"),
        "edc": ("EDC (1,2-dichloroethane)", "p_edc", "made at the complex"),
        "benzene": ("benzene", "p_benzene", "many sources"),
        "butadiene": ("1,3-butadiene", "p_butadiene", "many sources"),
        "total_voc": ("total VOC", "pred_sum7", "many sources")}


def main():
    a = pd.read_csv(os.path.join(HERE, "nov2026_ab_monitor_days.csv"))
    p = pd.read_csv(os.path.join(HERE, "plume_full", "plume_monitor_days_new156.csv"))
    d = a.merge(p[["date", "site", "p_vinyl_chloride", "p_benzene", "p_edc", "p_butadiene", "pred_sum7"]],
                on=["date", "site"], how="left", validate="one_to_one")
    assert len(d) == 230 and d.date.nunique() == 115 and d.p_vinyl_chloride.notna().all()
    # corrected total VOC (DEVIATIONS 38a): species reported in every year, canisters averaged
    tf = pd.read_csv(os.path.join(HERE, "total_voc_fixed.csv"))[["date", "site", "total_voc_fixed"]]
    d = d.merge(tf, on=["date", "site"], how="left", validate="one_to_one")
    assert d.total_voc_fixed.notna().all()
    d["total_voc"] = d.total_voc_fixed
    out = {"sample": "115 dates, 230 residential monitor-days", "seed": 20261103, "post_hoc": True}
    out["a_ori_minus_hysplit"] = {}
    for y, (name, hcol, grp) in POLL.items():
        r = S.paired(d, y, "f_corrected", hcol)
        r.update(pollutant=name, group=grp, hysplit_score=hcol,
                 detected_share={s: round(float((g[y] > 0).mean()), 2) for s, g in d.groupby("site")})
        out["a_ori_minus_hysplit"][y] = r
        print("a", y, r["refit_rho"], r["comparator_rho"], r["diff"], r["ci95"], flush=True)
    out["b_combo_minus_hysplit"] = {y: S.paired(d, y, "combo", POLL[y][1]) for y in ["vc", "benzene"]}
    out["c_hysplit_minus_hours_downwind"] = {y: S.paired(d, y, POLL[y][1], "hd") for y in ["vc", "benzene"]}
    out["hours_downwind_and_combo_same_days"] = {y: {"hours_downwind": S.describe(d, "hd", 2000)[y]["rho"],
                                                     "combo": S.describe(d, "combo", 2000)[y]["rho"]}
                                                 for y in ["vc", "benzene"]}
    from scipy.stats import spearmanr
    out["exact_rho_for_display"] = {
        y: {c: float(np.mean([spearmanr(g[c], g[y])[0] for _, g in d.groupby("site")]))
            for c in ["f_corrected", "hd", "combo", POLL[y][1]]} for y in POLL}
    out["total_voc_definition"] = "56 species reported in every year 2023-2025, canisters averaged (DEVIATIONS 38a)"
    json.dump(out, open(os.path.join(HERE, "nov2026_crossover_results.json"), "w"), indent=1, default=float)

    def row(lab, r):
        return (f"| {lab} | {r['refit_rho']:.2f} | {r['comparator_rho']:.2f} | {r['diff']:+.2f} | "
                f"{r['ci95'][0]:+.2f} to {r['ci95'][1]:+.2f} | {r['p_swap']:.4f} |")
    L = ["# Which score ranks which pollutant (post hoc, DEVIATIONS 37)", "",
         "115 dates, 230 residential monitor-days; mean within-monitor Spearman rho; paired difference with date-bootstrap "
         "95% CI and within-date swap p (unadjusted).", "",
         "## (a) ORI minus HYSPLIT", "", "| pollutant (group) | ORI | HYSPLIT | difference | 95% CI | p |", "|---|---|---|---|---|---|"]
    for y, r in out["a_ori_minus_hysplit"].items():
        L.append(row(f"{r['pollutant']} ({r['group']})", r))
    L += ["", "## (b) ORI + hours downwind minus HYSPLIT", "", "| pollutant | combo | HYSPLIT | difference | 95% CI | p |", "|---|---|---|---|---|---|"]
    for y, r in out["b_combo_minus_hysplit"].items():
        L.append(row(POLL[y][0], r))
    L += ["", "## (c) HYSPLIT minus hours downwind", "", "| pollutant | HYSPLIT | hours downwind | difference | 95% CI | p |", "|---|---|---|---|---|---|"]
    for y, r in out["c_hysplit_minus_hours_downwind"].items():
        L.append(row(POLL[y][0], r))
    open(os.path.join(HERE, "NOV2026_CROSSOVER_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
