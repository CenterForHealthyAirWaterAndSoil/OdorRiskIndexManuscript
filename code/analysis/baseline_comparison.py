"""EXPLORATORY: is the odor-trained forecast better than simple trapping indices or a model trained on the monitors?
Same 271 new monitor-days (2023, 2024, Jul-Dec 2025), weather from weather_history (standard source)."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
sys.path.insert(0, "/path/to/weather-varaible-analysis")
import weather_history as W
rng = np.random.default_rng(5)
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879)}
feats = []
for s, (lat, lon) in SITES.items():
    f = W.daily_features(lat, lon, "2023-01-01", "2025-12-31", tz="America/Chicago").reset_index(); f["site"] = s; feats.append(f)
wx = pd.concat(feats)
m = pd.read_csv("monitor_days.csv").merge(pd.read_csv("forecast_standard_source.csv"), on=["date", "site"]).merge(wx, on=["date", "site"])
m = m[m.site.isin(SITES) & ~m.poster_window].copy()
m["blh_m"] = m.boundary_layer_height / 3.28084; m["wind_ms"] = m.wind_speed * 0.44704
m["ventilation_index"] = m.blh_m * m.wind_ms                    # m2/s; low = trapped
m["yr"] = m.date.str[:4]
# Scores where HIGHER = more trapping / more expected pollution
cand = {"odor forecast (ours)": m.forecast_std, "low wind speed": -m.wind_speed, "low mixing height": -m.boundary_layer_height,
        "low ventilation index": -m.ventilation_index, "large day-night temp swing": m.diurnal_temperature_range}
# Monitor-trained model: ridge on weather features, trained within each monitor on the other years (leave-one-year-out)
X = ["temperature", "temperature_squared", "solar_radiation", "relative_humidity", "wind_speed", "precipitation",
     "diurnal_temperature_range", "boundary_layer_height", "atmospheric_pressure", "wind_u_proxy", "wind_v_proxy"]
m["wind_u_proxy"] = np.sin(np.radians(m.wind_direction)); m["wind_v_proxy"] = np.cos(np.radians(m.wind_direction))
def loyo(y):
    pred = pd.Series(np.nan, index=m.index)
    for s in SITES:
        for yr in m.yr.unique():
            tr = (m.site == s) & (m.yr != yr) & m[y].notna(); te = (m.site == s) & (m.yr == yr)
            Z = m.loc[tr, X]; mu, sd = Z.mean(), Z.std().replace(0, 1)
            mod = Ridge(alpha=10).fit((Z - mu) / sd, m.loc[tr, y].rank(pct=True))
            pred[te] = mod.predict((m.loc[te, X] - mu) / sd)
    return pred
def wrho(score, y):
    d = m.assign(_s=score).dropna(subset=[y, "_s"])
    return float(np.mean([spearmanr(g._s, g[y])[0] for _, g in d.groupby("site")]))
def boot_ci(score, y, B=2000):
    d = m.assign(_s=score).dropna(subset=[y, "_s"]); dates = d.date.unique(); g = {k: v for k, v in d.groupby("date")}; vals = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(dates, len(dates))])
        vals.append(np.mean([spearmanr(x._s, x[y])[0] for _, x in s.groupby("site")]))
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]
out = {}
for y in ["vc", "benzene", "total_voc"]:
    c = dict(cand); c["model trained on Calvert monitors (held-out year)"] = loyo(y)
    out[y] = {k: {"rho": round(wrho(v, y), 3), "ci95": boot_ci(v, y)} for k, v in c.items()}
json.dump(out, open("baseline_comparison.json", "w"), indent=1)
for y, r in out.items():
    print(f"\n{y}:"); [print(f"  {k:52s} rho {v['rho']:+.3f}  {v['ci95']}") for k, v in sorted(r.items(), key=lambda kv: -kv[1]["rho"])]
