"""Extended Calvert validation, following ANALYSIS_PLAN.md (frozen 2026-10-05, sha256 in PLAN_HASH.txt)."""
import os, sys, json, time, glob, numpy as np, pandas as pd, requests
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable
sys.path.insert(0, "/path/to/weather-varaible-analysis"); sys.path.insert(0, "/path/to/plume-validation")
import odor_forecast_core as M
from validation.odor_backtest import fetch_daily_features
HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(20261005)
RAW = "/path/to/aqs_daily_files/"  # EPA AQS daily_HAPS_YYYY.zip and daily_VOCS_YYYY.zip (aqs.epa.gov/aqsweb/airdata)
SITES = {"0018": ("elementary", 37.026757, -88.343757), "0020": ("johnson_riley", 37.041202, -88.351879), "0021": ("lwd", 37.047914, -88.338323)}
POLL = {"Vinyl chloride": "vc", "Benzene": "benzene", "1,3-Butadiene": "butadiene", "Ethylene dichloride": "edc"}

# ---------- 1. Calvert extracts from EPA files ----------
ext = os.path.join(HERE, "calvert_aqs_extract.csv")
if not os.path.exists(ext):
    rows = []
    for kind in ["HAPS", "VOCS"]:
        for y in [2023, 2024, 2025]:
            d = pd.read_csv(f"{RAW}daily_{kind}_{y}.csv", dtype=str, low_memory=False)
            d = d[(d["State Code"] == "21") & (d["County Code"] == "157") & d["Site Num"].isin(SITES) & (d["Sample Duration"] == "24 HOUR")]
            d["file"] = kind; rows.append(d)
    pd.concat(rows).to_csv(ext, index=False)
a = pd.read_csv(ext, dtype=str); a["val"] = pd.to_numeric(a["Arithmetic Mean"])
a["site"] = a["Site Num"].str.zfill(4).map(lambda s: SITES[s][0]); a["date"] = a["Date Local"]
h = a[(a.file == "HAPS") & a["Parameter Name"].isin(POLL)]
wide = h.pivot_table(index=["date", "site"], columns="Parameter Name", values="val", aggfunc="mean").rename(columns=POLL).reset_index()
v = a[(a.file == "VOCS") & (a["Units of Measure"] == "Parts per billion Carbon") & ~a["Parameter Name"].str.match(r"^(Total|Sum)")]
tv = v.groupby(["date", "site"]).agg(total_voc=("val", "sum"), n_species=("Parameter Name", "nunique")).reset_index()
data = wide.merge(tv, on=["date", "site"], how="left")

# ---------- 2. Forecast from ERA5 at each monitor ----------
fc_path = os.path.join(HERE, "forecast_era5.csv")
if not os.path.exists(fc_path):
    parts = []
    for num, (name, lat, lon) in SITES.items():
        for att in range(5):
            try: f = fetch_daily_features(lat, lon, "2023-01-01", "2025-12-31"); break
            except Exception: time.sleep(5 * (att + 1))
        for _, r in f.iterrows():
            mse, msa = M.compute_multisource_terms(lat, lon, r.wind_direction)
            row = r.to_dict(); row.update(mse=mse, msa=msa)
            parts.append({"date": r["date"] if "date" in f.columns else r.name, "site": name,
                          "forecast": M.predict_ori(row, M.COEFFS_POOLED_PROXIMITY, use_wind_filter=False)})
    pd.DataFrame(parts).to_csv(fc_path, index=False)
fc = pd.read_csv(fc_path)
data = data.merge(fc, on=["date", "site"], how="inner")
data["poster_window"] = (data.date >= "2025-01-01") & (data.date <= "2025-06-30")
data["year"] = data.date.str[:4] + np.where((data.date >= "2025-07-01"), "H2", "")
data.to_csv(os.path.join(HERE, "monitor_days.csv"), index=False)

RES = ["elementary", "johnson_riley"]
def within_rho(d, y):
    vals = [spearmanr(g.forecast, g[y])[0] for _, g in d.groupby("site") if g[y].nunique() > 1]
    return float(np.mean(vals)) if vals else np.nan

def joint_date_perm(d, y, B=10000):
    d = d[d.site.isin(RES)].dropna(subset=[y])
    both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)]
    piv = d.pivot(index="date", columns="site", values="forecast"); dates = piv.index.values
    obs = within_rho(d, y); c = 0
    fcv = {s: piv[s].values for s in RES}; idx = {dt: i for i, dt in enumerate(dates)}
    di = d.date.map(idx).values; sv = d.site.values
    for _ in range(B):
        p = rng.permutation(len(dates))
        t = d.assign(forecast=[fcv[s][p[i]] for s, i in zip(sv, di)])
        c += abs(within_rho(t, y)) >= abs(obs)
    return {"rho": round(obs, 3), "p": round((c + 1) / (B + 1), 4), "shared_dates": int(len(dates)), "monitor_days": int(len(d))}

def site_perm(g, y, B=5000):
    g = g.dropna(subset=[y]); obs = spearmanr(g.forecast, g[y])[0]; c = 0
    for _ in range(B): c += abs(spearmanr(rng.permutation(g.forecast.values), g[y])[0]) >= abs(obs)
    return {"rho": round(float(obs), 3), "p": round((c + 1) / (B + 1), 4), "n": int(len(g))}

out = {"plan_sha256": open(os.path.join(HERE, "PLAN_HASH.txt")).read().split()[0]}
new = data[~data.poster_window]
out["counts"] = new[new.site.isin(RES)].groupby("site").size().to_dict()

# ---------- PRIMARY ----------
prim = {k: joint_date_perm(new, k) for k in ["vc", "benzene"]}
rej, padj, _, _ = multipletests([prim["vc"]["p"], prim["benzene"]["p"]], method="holm")
for k, pa in zip(["vc", "benzene"], padj): prim[k]["p_holm"] = round(float(pa), 4)
ok = [prim[k]["rho"] > 0 and prim[k]["p_holm"] < 0.05 for k in ["vc", "benzene"]]
out["primary"] = prim; out["decision"] = "SUPPORTED" if all(ok) else ("PARTIALLY SUPPORTED" if any(ok) else "NOT SUPPORTED")
print("PRIMARY", json.dumps(prim), out["decision"], flush=True)

# ---------- SECONDARY ----------
sec = {}
sec["per_monitor"] = {s: {k: site_perm(new[new.site == s], k) for k in ["vc", "benzene"]} for s in RES}
def boot_auc(d, B=2000):
    ds = d.date.unique(); g = {k: v for k, v in d.groupby("date")}; vals = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(ds, len(ds))])
        if 0 < s.hi.sum() < len(s): vals.append(roc_auc_score(s.hi, s.fc))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]
sec["site_adjusted_auc"] = {}
for k in ["vc", "benzene", "total_voc"]:
    d = new[new.site.isin(RES)].dropna(subset=[k]).copy()
    d["hi"] = (d[k] >= d.groupby("site")[k].transform(lambda x: x.quantile(0.75))).astype(int)
    if k == "vc": d["hi"] = ((d[k] >= d.groupby("site")[k].transform(lambda x: x.quantile(0.75))) & (d[k] > 0)).astype(int)
    d["fc"] = d.forecast - d.groupby("site").forecast.transform("mean")
    sec["site_adjusted_auc"][k] = {"auc": round(roc_auc_score(d.hi, d.fc), 3), "ci": boot_auc(d), "events": int(d.hi.sum()), "n": len(d)}
sec["other_pollutants"] = {k: joint_date_perm(new, k, B=5000) for k in ["total_voc", "butadiene", "edc"]}
sec["all_dates_incl_poster"] = {k: joint_date_perm(data, k, B=5000) for k in ["vc", "benzene"]}
sec["by_year"] = {yr: {k: joint_date_perm(new[new.year == yr], k, B=3000) for k in ["vc", "benzene"]} for yr in sorted(new.year.unique())}
d = new[new.site.isin(RES)].dropna(subset=["vc"]).copy()
d["hi_fc"] = d.forecast > d.groupby("site").forecast.transform("median"); d["det"] = d.vc > 0
tabs = [np.array([[(g.hi_fc & g.det).sum(), (g.hi_fc & ~g.det).sum()], [(~g.hi_fc & g.det).sum(), (~g.hi_fc & ~g.det).sum()]]) + 0.5 for _, g in d.groupby("site")]
sec["vc_detection"] = {s: {"hi": f"{int((g.hi_fc & g.det).sum())}/{int(g.hi_fc.sum())}", "lo": f"{int((~g.hi_fc & g.det).sum())}/{int((~g.hi_fc).sum())}"} for s, g in d.groupby("site")}
sec["vc_detection"]["cmh_p"] = round(float(StratifiedTable(tabs).test_null_odds().pvalue), 4)
sec["vc_detection"]["cmh_or"] = round(float(StratifiedTable(tabs).oddsratio_pooled), 2)
sec["lwd"] = {k: site_perm(new[new.site == "lwd"], k) for k in ["vc", "benzene"]}
out["secondary"] = sec
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
