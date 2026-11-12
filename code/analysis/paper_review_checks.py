"""Checks added during the first author's review of the paper draft (DEVIATIONS 42, post hoc, exploratory).

All on the item 39/40 sample (141 dates, 282 community monitor-days) unless stated, with the statistics of
score_all_dates_refit.py (within-monitor Spearman rho; paired date bootstrap from score_refit.paired). Seed 20261111.
 A. ORI (refit) against each of the 15 stagnation and inversion measures and their composite (re-ranked on the
    141 dates), and against hours downwind, for benzene and vinyl chloride.
 B. ORI + hours downwind against composite + hours downwind (equal-weight within-monitor ranks), five pollutants.
 C. Days over the public alert line (ORI > 14.79%) against days under it, per monitor.
 D. HYSPLIT predicted / measured mean concentration per monitor, four species, on the 328 scored monitor-days.
 E. Pittsburgh complaint-weather correlations on calendar-month and year-month means (how Kim et al. 2026 computed
    theirs) beside the daily ones.
Output: paper_review_checks.json."""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import score_refit as S

HERE = os.path.dirname(os.path.abspath(__file__))
S.rng = np.random.default_rng(20261111)
d = pd.read_csv(os.path.join(HERE, "all_dates_monitor_days.csv"))
assert d.date.nunique() == 141 and len(d) == 282
TRAP = [c for c in d.columns[20:37] if "downwind" not in c]
assert len(TRAP) == 15
d["composite"] = d.groupby("site")[TRAP].rank(pct=True).mean(axis=1)
d["ori_hd"] = d.groupby("site")[["f_refit", "hd"]].rank(pct=True).mean(axis=1)
d["composite_hd"] = d.groupby("site")[["composite", "hd"]].rank(pct=True).mean(axis=1)
out = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}


def pr(y, a, b):
    r = S.paired(d, y, a, b)
    return {"a_rho": r["refit_rho"], "b_rho": r["comparator_rho"], "diff": r["diff"], "ci95": r["ci95"], "p_swap": r["p_swap"]}


for y in ["benzene", "vc"]:
    for c in TRAP + ["composite", "hd"]:
        out["A"][f"ORI minus {c} | {y}"] = pr(y, "f_refit", c)
        print(y, c, out["A"][f"ORI minus {c} | {y}"], flush=True)
for y in ["vc", "benzene", "edc", "butadiene", "total_voc"]:
    out["B"][y] = pr(y, "ori_hd", "composite_hd")
    print("B", y, out["B"][y], flush=True)

d["over"] = d.f_refit > 14.79
out["C"]["share_over"] = {s: float(g.over.mean()) for s, g in d.groupby("site")} | {"all": float(d.over.mean())}
for y in ["vc", "benzene", "edc", "total_voc"]:
    out["C"][y] = {s: {"over": float(g[g.over][y].mean()), "under": float(g[~g.over][y].mean()),
                       "ratio": float(g[g.over][y].mean() / g[~g.over][y].mean()),
                       "p_mannwhitney": float(stats.mannwhitneyu(g[g.over][y], g[~g.over][y]).pvalue)}
                   for s, g in d.groupby("site")}
print("C", json.dumps(out["C"], default=float)[:900])

p = pd.read_csv(os.path.join(HERE, "plume_full", "plume_monitor_days_all189.csv"))
p = p[p.site.isin(["elementary", "johnson_riley"])]
CONV = {"vc": (2, 62.50, "p_vinyl_chloride"), "benzene": (6, 78.11, "p_benzene"), "edc": (2, 98.96, "p_edc"),
        "butadiene": (4, 54.09, "p_butadiene")}
out["D"]["monitor_days"] = int(len(p)); out["D"]["dates"] = int(p.date.nunique())
for y, (nc, mw, pc) in CONV.items():
    out["D"][y] = {s: float(g[pc].mean() / (g[y] / nc * mw / 24.45).mean()) for s, g in p.dropna(subset=[y, pc]).groupby("site")}
print("D", out["D"])

panel = pd.read_parquet("/path/to/weather-varaible-analysis/training_panel_v2/zip_day_panel.parquet")
pg = panel[panel.city == "pgh"]
V = ["diurnal_temperature_range", "boundary_layer_height", "wind_speed", "temperature", "solar_radiation",
     "atmospheric_pressure", "relative_humidity", "precipitation"]
day = pg.groupby("date").agg(n=("n_reports", "sum"), **{v: (v, "mean") for v in V}).reset_index()
day["date"] = pd.to_datetime(day.date)
cal = day.groupby(day.date.dt.month).mean(numeric_only=True)
ym = day.groupby(day.date.dt.to_period("M")).mean(numeric_only=True)
out["E"] = {"days": int(len(day)), "share_days_without_reports": float((day.n == 0).mean()), "median_reports_per_day": float(day.n.median()),
            "daily_spearman": {v: float(stats.spearmanr(day.n, day[v])[0]) for v in V},
            "calendar_month_pearson_n12": {v: float(stats.pearsonr(cal.n, cal[v])[0]) for v in V},
            "year_month_pearson": {v: float(stats.pearsonr(ym.n, ym[v])[0]) for v in V}, "year_months": int(len(ym))}
print("E", out["E"])
json.dump(out, open(os.path.join(HERE, "paper_review_checks.json"), "w"), indent=1, default=float)
print("wrote paper_review_checks.json")
