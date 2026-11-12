"""POST HOC. (1) Fill the Jan-Jun 2024 BLH gap from Copernicus ERA5 (the reanalysis Open-Meteo serves) and recompute forecasts; same-month-mean fill kept for comparison.
(2) Re-run the two primary endpoints on corrected forecasts (sensitivity, not confirmatory).
(3) Exploratory sweep of every chemical in the EPA files, Benjamini-Hochberg across chemicals."""
import sys, json, numpy as np, pandas as pd, xarray as xr
from scipy.stats import spearmanr
sys.path.insert(0, "/path/to/weather-varaible-analysis"); sys.path.insert(0, "/path/to/plume-validation")
import odor_forecast_core as M
from validation.odor_backtest import fetch_daily_features
rng = np.random.default_rng(77)
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879), "lwd": (37.047914, -88.338323)}
RES = ["elementary", "johnson_riley"]

# ---- 1. corrected forecasts (Copernicus key failed; fill missing BLH with the site's same-month mean) ----
rows, check = [], {}
def score(f, lat, lon, blh_col):
    out = []
    for d, r in f.iterrows():
        mse, msa = M.compute_multisource_terms(lat, lon, r.wind_direction); row = r.to_dict()
        row.update(mse=mse, msa=msa, boundary_layer_height=r[blh_col])
        out.append(M.predict_ori(row, M.COEFFS_POOLED_PROXIMITY, use_wind_filter=False) if pd.notna(r[blh_col]) else np.nan)
    return np.array(out)
ds = xr.open_dataset("era5_blh_calvert_2024h1.nc")
clim_rows = []
for s, (lat, lon) in SITES.items():
    f = fetch_daily_features(lat, lon, "2023-01-01", "2025-12-31"); f["date"] = f["date"] if "date" in f.columns else f.index
    f = f.set_index("date"); f["month"] = pd.to_datetime(f.index).month
    ser = ds["blh"].sel(latitude=lat, longitude=lon, method="nearest").to_series()
    ser.index = pd.to_datetime(ser.index).tz_localize("UTC").tz_convert("America/Chicago")
    cds_daily = (ser * 3.28084).groupby(ser.index.strftime("%Y-%m-%d")).mean()
    cds_daily = cds_daily[ser.groupby(ser.index.strftime("%Y-%m-%d")).size() == 24]   # complete local days only
    ok = f.boundary_layer_height.notna()
    both = [d for d in f.index[ok] if d in cds_daily.index]
    check[s] = {"overlap_days_openmeteo_vs_cds": len(both),
                "spearman_openmeteo_vs_cds": round(float(spearmanr(f.loc[both, "boundary_layer_height"], cds_daily.loc[both])[0]), 4),
                "mean_ratio_cds_over_openmeteo": round(float(cds_daily.loc[both].mean() / f.loc[both, "boundary_layer_height"].mean()), 3),
                "filled_days": int((~ok).sum())}
    f["blh_cds"] = f.boundary_layer_height.fillna(cds_daily.reindex(f.index))
    check[s]["still_missing_after_cds_fill"] = int(f.blh_cds.isna().sum())
    f["blh_clim"] = f.boundary_layer_height.fillna(f.month.map(f.boundary_layer_height.groupby(f.month).mean()))
    for d, p, q in zip(f.index, score(f, lat, lon, "blh_cds"), score(f, lat, lon, "blh_clim")):
        rows.append({"date": d, "site": s, "forecast_fixed": p}); clim_rows.append({"date": d, "site": s, "forecast_clim": q})
pd.DataFrame(clim_rows).to_csv("forecast_era5_blh_climfill.csv", index=False)
fx = pd.DataFrame(rows); fx.to_csv("forecast_era5_blh_cdsfill.csv", index=False)
m = pd.read_csv("monitor_days.csv").merge(fx, on=["date", "site"], how="left")
m["changed"] = (m.forecast - m.forecast_fixed).abs() > 0.05
out = {"blh_check": check, "monitor_days_changed": int(m[m.site.isin(RES)].changed.sum())}

def wrho(d, y, fc):
    return float(np.mean([spearmanr(g[fc], g[y])[0] for _, g in d.groupby("site") if g[y].nunique() > 1 and g[fc].nunique() > 1]))
def joint_perm(d, y, fc, B):
    d = d[d.site.isin(RES)].dropna(subset=[y, fc]); both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)]
    if d[y].nunique() < 3: return None
    piv = d.pivot(index="date", columns="site", values=fc); dates = piv.index.values; idx = {k: i for i, k in enumerate(dates)}
    obs = wrho(d, y, fc); c = 0; di = d.date.map(idx).values; sv = d.site.values; arr = {s: piv[s].values for s in RES}
    for _ in range(B):
        p = rng.permutation(len(dates)); c += abs(wrho(d.assign(**{fc: [arr[s][p[i]] for s, i in zip(sv, di)]}), y, fc)) >= abs(obs)
    return {"rho": round(obs, 3), "p": round((c + 1) / (B + 1), 4), "monitor_days": int(len(d)), "detected_share": round(float((d[y] > 0).mean()), 2)}

new = m[~m.poster_window]
out["primary_as_specified"] = {y: joint_perm(new, y, "forecast", 10000) for y in ["vc", "benzene"]}
out["primary_blh_corrected"] = {y: joint_perm(new, y, "forecast_fixed", 10000) for y in ["vc", "benzene"]}
out["by_year_corrected"] = {yr: {y: joint_perm(new[new.date.str[:4] == yr], y, "forecast_fixed", 3000) for y in ["vc", "benzene"]}
                            for yr in ["2023", "2024", "2025"]}
out["total_voc_corrected"] = joint_perm(new, "total_voc", "forecast_fixed", 5000)

# ---- 3. every chemical (exploratory) ----
a = pd.read_csv("calvert_aqs_extract.csv", dtype=str); a["val"] = pd.to_numeric(a["Arithmetic Mean"])
a["site"] = a["Site Num"].str.zfill(4).map({"0018": "elementary", "0020": "johnson_riley", "0021": "lwd"}); a["date"] = a["Date Local"]
a = a[a.site.isin(RES) & ~a["Parameter Name"].str.match(r"^(Total|Sum)")]
a["src"] = np.where(a.file == "HAPS", 0, 1)  # prefer the HAPS value when a species is in both files
a = a.sort_values("src").drop_duplicates(["date", "site", "Parameter Name"])
wide = a.pivot_table(index=["date", "site"], columns="Parameter Name", values="val").reset_index()
sp = new[["date", "site", "forecast_fixed"]].merge(wide, on=["date", "site"])
res = []
for c in [c for c in wide.columns if c not in ("date", "site")]:
    if (sp[c] > 0).mean() < 0.10 or sp[c].notna().sum() < 100: continue
    r = joint_perm(sp, c, "forecast_fixed", 2000)
    if r: res.append({"chemical": c, **r})
t = pd.DataFrame(res).sort_values("p").reset_index(drop=True); k = len(t)
t["q_bh"] = (t.p * k / (t.index + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1).round(4)
t.to_csv("species_sweep_corrected.csv", index=False)
clim = pd.read_csv("forecast_era5_blh_climfill.csv"); cm = m.merge(clim, on=["date", "site"])
out["cds_vs_clim_fill_spearman_filled_days"] = round(float(spearmanr(cm[cm.date < "2024-07-01"].forecast_fixed, cm[cm.date < "2024-07-01"].forecast_clim)[0]), 4)
out["species_sweep"] = {"n_tested": k, "n_q_lt_0.05": int((t.q_bh < 0.05).sum()),
                        "n_positive": int((t.rho > 0).sum()), "table_file": "species_sweep_corrected.csv"}
json.dump(out, open("corrected_and_species_cds.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)); print(t.to_string(index=False))
