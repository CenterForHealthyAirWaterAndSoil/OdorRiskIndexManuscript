"""Extended plume (HYSPLIT) validation, following PLUME_PLAN.md (frozen 2026-10-06, sha256 in PLUME_PLAN_HASH.txt)
and DEVIATIONS.md items 5-14. Statistics mirror run_extended.py with the HYSPLIT predicted daily mean as the score."""
import os, sys, json, hashlib, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
sys.path.insert(0, "/path/to/plume-validation")
from validation import config as C
HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(20261005)
# DEVIATIONS 32: optional --dates-file and --tag (defaults reproduce the 78-date half-sample run and its file names)
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dates-file", default=os.path.join(HERE, "plume_dates_subset.txt"))
_ap.add_argument("--tag", default=None)
ARGS = _ap.parse_known_args()[0]   # tolerate the sensitivity scripts' own flags (e.g. --score)
OUT_JSON = os.path.join(HERE, "plume_results.json") if ARGS.tag is None else os.path.join(HERE, "plume_full", f"plume_results_{ARGS.tag}.json")
OUT_CSV = os.path.join(HERE, "plume_monitor_days.csv") if ARGS.tag is None else os.path.join(HERE, "plume_full", f"plume_monitor_days_{ARGS.tag}.csv")
PRED_DIR = "/path/to/plume-validation/validation/results/predicted_clipped"
RES = ["elementary", "johnson_riley"]

# ---------- plan integrity ----------
plan_sha = hashlib.sha256(open(os.path.join(HERE, "PLUME_PLAN.md"), "rb").read()).hexdigest()
frozen_sha = open(os.path.join(HERE, "PLUME_PLAN_HASH.txt")).read().split()[0]
assert plan_sha == frozen_sha, "PLUME_PLAN.md changed since it was frozen"

# ---------- predictions (78-date half-sample, item 11) ----------
dates = [l.strip() for l in open(ARGS.dates_file) if l.strip()]
missing = [d for d in dates if not os.path.exists(os.path.join(PRED_DIR, f"{d}.DONE"))]
assert not missing, f"dates without a DONE marker: {missing}"
pr = pd.concat([pd.read_csv(os.path.join(PRED_DIR, f"{d}.csv")) for d in dates])
pr = pr[pr.site.isin(RES)]
pw = pr.pivot_table(index=["date", "site"], columns="species", values="pred_mean_ugm3").reset_index()
pw = pw.rename(columns={sp: f"p_{sp}" for sp in C.TARGET_SPECIES})   # measured columns share these names
pw["pred_sum7"] = pw[[f"p_{sp}" for sp in C.TARGET_SPECIES]].sum(axis=1)

# ---------- measurements (same monitor-day table as run_extended.py) ----------
md = pd.read_csv(os.path.join(HERE, "monitor_days.csv"))
md = md[md.site.isin(RES) & md.date.isin(dates)]
fs = pd.read_csv(os.path.join(HERE, "forecast_standard_source.csv"))
data = md.merge(pw, on=["date", "site"], how="inner").merge(fs[["date", "site", "forecast_std"]], on=["date", "site"], how="left")
assert len(data) == len(md), "a monitor-day has no plume prediction"
data["year"] = data.date.str[:4] + np.where(data.date >= "2025-07-01", "H2", "")

# ---------- statistics (as in run_extended.py; rank form of the same Spearman) ----------
def within_rho(d, x, y):
    vals = [spearmanr(g[x], g[y])[0] for _, g in d.groupby("site") if g[y].nunique() > 1]
    return float(np.mean(vals)) if vals else np.nan

def shared(d, y):
    d = d.dropna(subset=[y]); both = d.groupby("date").site.nunique()
    return d[d.date.isin(both[both == 2].index)]

def _rank_corr(xr, yr):
    xr = xr - xr.mean(axis=-1, keepdims=True); yr = yr - yr.mean()
    return (xr @ yr) / np.sqrt((xr ** 2).sum(axis=-1) * (yr ** 2).sum())

def joint_date_perm(d, x, y, B=10000):
    """Shuffle the score's dates jointly across both monitors (same permutation for both)."""
    d = shared(d, y); obs = within_rho(d, x, y)
    piv_x = d.pivot(index="date", columns="site", values=x); piv_y = d.pivot(index="date", columns="site", values=y)
    perms = np.array([rng.permutation(len(piv_x)) for _ in range(B)])
    null = np.zeros(B); k = 0
    for s in RES:
        if piv_y[s].nunique() <= 1: continue
        xr = rankdata(piv_x[s].values); yr = rankdata(piv_y[s].values)
        null += _rank_corr(xr[perms], yr); k += 1
    null /= k
    c = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    return {"rho": round(obs, 3), "p": round((c + 1) / (B + 1), 4), "shared_dates": int(len(piv_x)), "monitor_days": int(len(d))}

def site_perm(g, x, y, B=5000):
    g = g.dropna(subset=[y]); obs = spearmanr(g[x], g[y])[0]
    xr = rankdata(g[x].values); yr = rankdata(g[y].values)
    null = _rank_corr(xr[np.array([rng.permutation(len(g)) for _ in range(B)])], yr)
    c = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    return {"rho": round(float(obs), 3), "p": round((c + 1) / (B + 1), 4), "n": int(len(g))}

def hi_days(d, y):
    q = d.groupby("site")[y].transform(lambda v: v.quantile(0.75))
    return ((d[y] >= q) & (d[y] > 0) if y == "vc" else d[y] >= q).astype(int)

def boot_auc(d, B=2000):
    ds = d.date.unique(); g = {k: v for k, v in d.groupby("date")}; vals = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(ds, len(ds))])
        if 0 < s.hi.sum() < len(s): vals.append(roc_auc_score(s.hi, s.fc))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

SCORE = {"vc": "p_vinyl_chloride", "benzene": "p_benzene"}   # species-matched score for each measured pollutant
out = {"plume_plan_sha256": plan_sha, "dates": len(dates), "failed_dates_excluded": [],
       "monitor_days": data.groupby("site").size().to_dict()}

# ---------- PRIMARY (Q1 vinyl chloride, Q2 benzene) ----------
prim = {k: joint_date_perm(data, SCORE[k], k) for k in ["vc", "benzene"]}
_, padj, _, _ = multipletests([prim["vc"]["p"], prim["benzene"]["p"]], method="holm")
for k, pa in zip(["vc", "benzene"], padj): prim[k]["p_holm"] = round(float(pa), 4)
ok = [prim[k]["rho"] > 0 and prim[k]["p_holm"] < 0.05 for k in ["vc", "benzene"]]
out["primary"] = prim; out["decision"] = "SUPPORTED" if all(ok) else ("PARTIALLY SUPPORTED" if any(ok) else "NOT SUPPORTED")
print("PRIMARY", json.dumps(prim), out["decision"], flush=True)

# ---------- SECONDARY ----------
sec = {}
# 1. poster metric: 7-species predicted sum
sec["sum7"] = {k: joint_date_perm(data, "pred_sum7", k) for k in ["vc", "benzene", "total_voc"]}
# 2. each endpoint per monitor
sec["per_monitor"] = {s: {k: site_perm(data[data.site == s], SCORE[k], k) for k in ["vc", "benzene"]} for s in RES}
# 3. site-adjusted AUC for top-quartile days (VC also requires detection)
sec["site_adjusted_auc"] = {}
for k, x in [("vc", "p_vinyl_chloride"), ("benzene", "p_benzene"), ("total_voc", "pred_sum7")]:
    d = data.dropna(subset=[k]).copy(); d["hi"] = hi_days(d, k)
    d["fc"] = d[x] - d.groupby("site")[x].transform("mean")
    sec["site_adjusted_auc"][k] = {"score": x, "auc": round(roc_auc_score(d.hi, d.fc), 3), "ci": boot_auc(d),
                                   "events": int(d.hi.sum()), "n": len(d)}
# 4. spatial: predicted JR-minus-Elementary vs measured JR-minus-Elementary
sec["spatial"] = {}
for k in ["vc", "benzene"]:
    d = shared(data, k); px = d.pivot(index="date", columns="site", values=SCORE[k]); py = d.pivot(index="date", columns="site", values=k)
    dx = (px.johnson_riley - px.elementary).values
    dy = (py.johnson_riley - py.elementary).round(6).values   # round off float noise so equal measured differences tie
    obs = spearmanr(dx, dy)[0]; B = 10000
    null = _rank_corr(rankdata(dx)[np.array([rng.permutation(len(dx)) for _ in range(B)])], rankdata(dy))
    sec["spatial"][k] = {"rho": round(float(obs), 3), "p": round((int((np.abs(null) >= abs(obs) - 1e-12).sum()) + 1) / (B + 1), 4),
                         "shared_dates": int(len(dx)), "pred_jr_higher": f"{int((dx > 0).sum())}/{len(dx)}",
                         "meas_jr_higher": f"{int((dy > 0).sum())}/{len(dy)}"}
# 5. added value over the (corrected) odor forecast
sec["added_value"] = {}
for k, x in [("vc", "p_vinyl_chloride"), ("benzene", "p_benzene"), ("total_voc", "pred_sum7")]:
    d = data.dropna(subset=[k]).copy(); d["hi"] = hi_days(d, k)
    z = lambda c: (d[c] - d.groupby("site")[c].transform("mean")) / d.groupby("site")[c].transform("std")
    d["z_plume"], d["z_odor"] = z(x), z("forecast_std"); d["jr"] = (d.site == "johnson_riley").astype(int)   # monitor fixed effect
    groups = pd.factorize(d.date)[0]
    res = {}
    for name, f in [("both", "hi ~ z_plume + z_odor + jr"), ("odor_only", "hi ~ z_odor + jr"), ("plume_only", "hi ~ z_plume + jr")]:
        m = smf.logit(f, d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        res[name] = {t: {"or_per_sd": round(float(np.exp(m.params[t])), 2), "p": round(float(m.pvalues[t]), 4)}
                     for t in ["z_plume", "z_odor"] if t in m.params}
        res[name]["llf"] = round(float(m.llf), 2)
    res["events"] = int(d.hi.sum()); res["n"] = len(d)
    sec["added_value"][k] = res
# 6. by year (all-dates-including-poster deferred, DEVIATIONS item 14)
sec["by_year"] = {yr: {k: joint_date_perm(data[data.year == yr], SCORE[k], k, B=3000) for k in ["vc", "benzene"]}
                  for yr in sorted(data.year.unique())}
sec["all_dates_incl_poster"] = ("deferred: poster dates not re-run with the 150k particle cap (DEVIATIONS items 11, 14)"
                                if ARGS.tag is None else "scored separately as tag all189 (DEVIATIONS 16d, 32)")
# 7. magnitude: mean predicted / mean measured (ug/m3), per species and monitor
a = pd.read_csv(os.path.join(HERE, "calvert_aqs_extract.csv"), dtype=str)
a = a[(a.file == "HAPS") & a["Site Num"].str.zfill(4).isin(["0018", "0020"])]
a["site"] = a["Site Num"].str.zfill(4).map({"0018": "elementary", "0020": "johnson_riley"}); a["date"] = a["Date Local"]
a = a[a.date.isin(dates)]; a["ppbc"] = pd.to_numeric(a["Arithmetic Mean"]).clip(lower=0)
mag = {}
for sp, info in C.TARGET_SPECIES.items():
    m = a[a["Parameter Name"] == info["monitor_name"]].groupby(["date", "site"]).ppbc.mean().reset_index()
    m["meas_ugm3"] = m.ppbc / info["n_carbons"] * info["mol_weight"] / C.MOLAR_VOLUME_L
    m = m.merge(pw[["date", "site", f"p_{sp}"]], on=["date", "site"]).rename(columns={f"p_{sp}": sp})
    for s, g in m.groupby("site"):
        mag.setdefault(sp, {})[s] = {"pred_mean": round(float(g[sp].mean()), 4), "meas_mean": round(float(g.meas_ugm3.mean()), 4),
                                     "ratio": round(float(g[sp].mean() / g.meas_ugm3.mean()), 2) if g.meas_ugm3.mean() > 0 else None,
                                     "detect_frac": round(float((g.ppbc > 0).mean()), 2), "n": int(len(g))}
sec["magnitude"] = mag
out["secondary"] = sec
if ARGS.tag is not None:
    out["tag"] = ARGS.tag; out["dates_file"] = os.path.basename(ARGS.dates_file)
json.dump(out, open(OUT_JSON, "w"), indent=1, default=float)
data.to_csv(OUT_CSV, index=False)
print(json.dumps(out, indent=1, default=float))
