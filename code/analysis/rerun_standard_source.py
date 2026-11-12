"""Re-run the extended Calvert analysis with weather from the project standard (weather_history.daily_features):
ERA5 via Open-Meteo, Copernicus gap fill, DST-correct local days. Primary endpoints as in ANALYSIS_PLAN.md."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "/path/to/weather-varaible-analysis")
import odor_forecast_core as M, weather_history as W
rng = np.random.default_rng(99)
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879), "lwd": (37.047914, -88.338323)}
RES = ["elementary", "johnson_riley"]
rows = []
for s, (lat, lon) in SITES.items():
    f = W.daily_features(lat, lon, "2023-01-01", "2025-12-31", tz="America/Chicago")
    for d, r in f.iterrows():
        mse, msa = M.compute_multisource_terms(lat, lon, r.wind_direction); row = r.to_dict(); row.update(mse=mse, msa=msa)
        rows.append({"date": d, "site": s, "forecast_std": M.predict_ori(row, M.COEFFS_POOLED_PROXIMITY, use_wind_filter=False),
                     "era5_filled_hours": int(r.era5_filled_hours)})
fc = pd.DataFrame(rows); fc.to_csv("forecast_standard_source.csv", index=False)
m = pd.read_csv("monitor_days.csv").merge(fc, on=["date", "site"])
def wrho(d, y): return float(np.mean([spearmanr(g.forecast_std, g[y])[0] for _, g in d.groupby("site") if g[y].nunique() > 1]))
def perm(d, y, B):
    d = d[d.site.isin(RES)].dropna(subset=[y]); both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)]
    piv = d.pivot(index="date", columns="site", values="forecast_std"); dates = piv.index.values; idx = {k: i for i, k in enumerate(dates)}
    obs = wrho(d, y); c = 0; di = d.date.map(idx).values; sv = d.site.values; arr = {s: piv[s].values for s in RES}
    for _ in range(B):
        p = rng.permutation(len(dates)); c += abs(wrho(d.assign(forecast_std=[arr[s][p[i]] for s, i in zip(sv, di)]), y)) >= abs(obs)
    return {"rho": round(obs, 3), "p": round((c + 1) / (B + 1), 4), "monitor_days": int(len(d))}
new = m[~m.poster_window]
out = {"source": "weather_history.daily_features (ERA5 archive + Copernicus fill, DST-aware local days)",
       "primary": {y: perm(new, y, 10000) for y in ["vc", "benzene"]},
       "total_voc": perm(new, "total_voc", 5000),
       "by_year": {yr: {y: perm(new[new.date.str[:4] == yr], y, 3000) for y in ["vc", "benzene"]} for yr in ["2023", "2024", "2025"]},
       "forecast_mean_by_year": new[new.site.isin(RES)].groupby(new.date.str[:4]).forecast_std.mean().round(1).to_dict()}
json.dump(out, open("standard_source_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
