"""Does dropping Louisville (or just its sewer reports) help the ORI at Calvert City? (DEVIATIONS 41, post hoc, exploratory)

Uses fits already made by training/fit_refit.py under REFIT_PLAN (city_pgh_only, city_lou_only, sewer_removed); nothing
is refit here. Same sample and statistics as score_all_dates_refit.py (141 dates, 282 community monitor-days; paired
date bootstrap). Seed 20261111. Output: drop_louisville_results.json."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

EXT = "/path/to/school/R!Louisville/Re-analysis/extended/"
sys.path.insert(0, EXT)
os.chdir(EXT)
import score_refit as S  # noqa: E402

S.rng = np.random.default_rng(20261111)
coef = pd.read_csv("/path/to/weather-varaible-analysis/training_panel_v2/fit/coefficients.csv")
B = {f: coef[coef.fit == f].set_index("term").coef for f in ["main", "city_pgh_only", "city_lou_only", "sewer_removed"]}
d = pd.read_csv(EXT + "all_dates_monitor_days.csv")
for f in B:
    d["lp_" + f] = sum(B[f][t] * d[t] for t in S.TERMS)
assert np.allclose(d.lp_main, d.lp_refit) and np.allclose(d.lp_city_pgh_only, d.lp_pgh_only)
P = ["vc", "benzene", "edc", "butadiene", "total_voc"]


def wr(col, g=d, ys=P):
    return {y: round(float(np.mean([stats.spearmanr(x[col], x[y])[0] for _, x in g.groupby("site")])), 3) for y in ys}


print("== 141 dates, within-monitor rho")
for f, lab in [("main", "pooled (paper)"), ("city_pgh_only", "Pittsburgh only"), ("sewer_removed", "Louisville sewer reports removed"), ("city_lou_only", "Louisville only")]:
    print(f"{lab:34s}", wr("lp_" + f))
d["hd"] = d["hd"]
for f in ["main", "city_pgh_only"]:
    d["combo_" + f] = d.groupby("site")[["lp_" + f, "hd"]].rank(pct=True).mean(axis=1)
    print(f"combo with {f:15s}", wr("combo_" + f, ys=["vc", "benzene"]))
out = {"rho_141": {}, "paired_vs_pooled": {}, "older_monitors": {}}
for f in ["main", "city_pgh_only", "sewer_removed", "city_lou_only"]:
    out["rho_141"][f] = wr("lp_" + f)
print("\n== paired (minus the pooled model), 10,000 date resamples")
for f in ["city_pgh_only", "sewer_removed"]:
    for y in ["vc", "benzene", "edc", "total_voc"]:
        r = S.paired(d, y, "lp_" + f, "lp_main")
        out["paired_vs_pooled"][f"{f} | {y}"] = {"a_rho": r["refit_rho"], "b_rho": r["comparator_rho"], "diff": r["diff"], "ci95": r["ci95"]}
        print(f"{f:15s} {y:9s} {r['refit_rho']:.3f} vs {r['comparator_rho']:.3f}  diff {r['diff']:+.3f}  CI {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}")
# no-alignment versions: how much trapping signal each fit carries
for f in ["main", "city_pgh_only"]:
    d["na_" + f] = d["lp_" + f] - B[f]["wind_align_weighted"] * d.wind_align_weighted
    print(f"without wind term, {f:15s}", wr("na_" + f, ys=["vc", "benzene"]))

print("\n== older monitors 2011-2022 (community monitors)")
sc = pd.read_csv(EXT + "older_monitors/older_scores.csv")
om = pd.read_csv("/path/to/school/Calvert_Publication_Package/02_analysis_datasets/older_monitor_days_2011_2022.csv")
om = om[~om.set_aside_in_plant.astype(bool)]
m = om.merge(sc[["date", "site_id", "lp_refit", "lp_pgh_only", "lp_lou_only", "hours_downwind"] + S.TERMS],
             left_on=["date", "aqs_site_id"], right_on=["date", "site_id"], how="left", validate="one_to_one")
assert m.lp_refit.notna().all()
m["lp_sewer_removed"] = sum(B["sewer_removed"][t] * m[t] for t in S.TERMS)
for col, lab in [("lp_refit", "pooled (paper)"), ("lp_pgh_only", "Pittsburgh only"), ("lp_sewer_removed", "sewer removed"), ("lp_lou_only", "Louisville only")]:
    v = {y: round(float(np.mean([stats.spearmanr(g[col], g[c2], nan_policy="omit")[0] for _, g in m.groupby("aqs_site_id")])), 3)
         for y, c2 in [("vc", "vinyl_chloride_ppbC"), ("benzene", "benzene_ppbC")]}
    print(f"{lab:34s}", v, "n", len(m))
    out["older_monitors"][col] = v
json.dump(out, open(EXT + "drop_louisville_results.json", "w"), indent=1, default=float)
print("wrote drop_louisville_results.json")
