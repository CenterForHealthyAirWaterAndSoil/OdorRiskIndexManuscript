"""EXPLORATORY: fair comparison of the odor forecast with standard trapping/stagnation indices.
Hourly weather: weather_history (standard ERA5 source) + Open-Meteo ERA5 100 m wind (transport wind, UTC request);
upper air: Copernicus ERA5 pressure levels at the Calvert grid point (37.00 N, 88.25 W). Days are Central local days.
Every index is oriented so higher = more trapping (or, for the wind-direction measures, more time downwind). Same 271 new monitor-days as the primary analysis."""
import sys, glob, json, numpy as np, pandas as pd, xarray as xr
from scipy.stats import spearmanr
sys.path.insert(0, "/path/to/weather-varaible-analysis")
import weather_history as W
import odor_forecast_core as M
from odor_forecast_core import fetch_open_meteo_json
rng = np.random.default_rng(11)
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879)}
TZ = "America/Chicago"

def era5_pl():
    def load(pattern):
        parts = [xr.open_dataset(f).load() for f in sorted(glob.glob(pattern))]
        tdim = "valid_time" if "valid_time" in parts[0].dims else "time"
        return xr.concat(parts, dim=tdim).sortby(tdim)
    t, w = load("era5_pl_temp_*.nc"), load("era5_pl_wind500_*.nc")
    lev = "pressure_level" if "pressure_level" in t.dims else "level"
    df = pd.DataFrame({"t925": t["t"].sel({lev: 925}).squeeze().to_series(), "t850": t["t"].sel({lev: 850}).squeeze().to_series()})
    u = w["u"].sel({lev: 500}).squeeze().to_series(); v = w["v"].sel({lev: 500}).squeeze().to_series()
    df["wind500"] = np.hypot(u, v).reindex(df.index)
    df.index = pd.to_datetime(df.index); return df
pl = era5_pl()

rows = []
for s, (lat, lon) in SITES.items():
    h = W.hourly_weather(lat, lon, "2023-01-01", "2025-12-31")
    d = fetch_open_meteo_json(f"{W.ARCHIVE_URL}?latitude={lat}&longitude={lon}&start_date=2022-12-31&end_date=2026-01-01"
                              f"&hourly=wind_speed_100m&wind_speed_unit=ms&timezone=GMT")
    w100 = pd.Series(d["hourly"]["wind_speed_100m"], index=pd.to_datetime(d["hourly"]["time"]), dtype=float)
    h = h.join(w100.rename("w100_ms")).join(pl)
    loc = h.index.tz_localize("UTC").tz_convert(TZ)
    h["date"] = loc.strftime("%Y-%m-%d"); h["hr"] = loc.hour
    h["w10_ms"] = h.wind_speed_10m * 0.44704; h["blh_m"] = h.boundary_layer_height / 3.28084
    h["t2k"] = (h.temperature_2m - 32) * 5 / 9 + 273.15; h["p_sfc"] = h.surface_pressure
    # wind blows FROM wind_direction; the monitor is downwind of the complex when wind comes from (bearing - 180)
    brg = M._source_bearing(M.IND_LAT, M.IND_LON, lat, lon)
    diff = ((h.wind_direction_10m - (brg + 180) % 360 + 180) % 360) - 180
    h["downwind"] = (diff.abs() <= 30).astype(float)
    h["downwind_calm"] = ((diff.abs() <= 30) & (h.wind_speed_10m * 0.44704 < 3.0)).astype(float)
    h["inv925"] = h.t925 - h.t2k; h["inv850"] = h.t850 - h.t2k
    h["lts"] = h.t850 * (1000 / 850) ** 0.286 - h.t2k * (1000 / h.p_sfc) ** 0.286
    night, aft = h[h.hr < 6], h[(h.hr >= 12) & (h.hr <= 17)]
    g, gn, ga = h.groupby("date"), night.groupby("date"), aft.groupby("date")
    six = h[h.hr == 6].set_index("date")
    day = pd.DataFrame({
        "low wind (daily mean)": -g.w10_ms.mean(), "low wind (overnight)": -gn.w10_ms.mean(),
        "low mixing height (daily mean)": -g.blh_m.mean(), "low mixing height (overnight min)": -g.blh_m.min(),
        "low mixing height (afternoon max)": -ga.blh_m.max(),
        "low ventilation index (afternoon, NWS-style)": -(ga.blh_m.max() * ga.w100_ms.mean()),
        "low ventilation index (overnight)": -(gn.blh_m.mean() * gn.w100_ms.mean()),
        "low ventilation index (daily mean)": -(g.blh_m.mean() * g.w10_ms.mean()),
        "inversion 925 hPa at 6 AM": six.inv925, "inversion 850 hPa at 6 AM": six.inv850,
        "strongest overnight inversion (925 hPa)": h[h.hr <= 8].groupby("date").inv925.max(),
        "lower-tropospheric stability": g.lts.mean(),
        "large day-night temp swing": g.temperature_2m.max() - g.temperature_2m.min(),
        "hours downwind of complex (wind-direction analysis)": g.downwind.mean(),
        "hours downwind of complex in calm wind": g.downwind_calm.mean()})
    stag = pd.DataFrame({"w": g.w10_ms.mean() < 3.2, "u": g.wind500.mean() < 13.0, "p": g.rain.sum() < 1 / 25.4})
    day["NOAA stagnation day (Horton 2012)"] = stag.all(axis=1).astype(int)
    day["stagnation criteria met (0-3)"] = stag.sum(axis=1)
    day["site"] = s; rows.append(day.reset_index())
idx = pd.concat(rows)
m = (pd.read_csv("monitor_days.csv").merge(pd.read_csv("forecast_standard_source.csv"), on=["date", "site"])
     .merge(idx, on=["date", "site"]))
m = m[m.site.isin(SITES) & ~m.poster_window].copy()
names = [c for c in idx.columns if c not in ("date", "site")]
# composite: mean within-monitor percentile rank of every index (best-effort combination, no training)
m["composite of all indices"] = m.groupby("site")[names].rank(pct=True).mean(axis=1)
m["odor forecast (ours)"] = m.forecast_std
cands = ["odor forecast (ours)", "composite of all indices"] + names

# ---- models that need local monitor history (trained within each monitor, tested on a held-out year) ----
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
m["yr"] = m.date.str[:4]; doy = pd.to_datetime(m.date).dt.dayofyear
m["doy_sin"], m["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
m["weekday"] = pd.to_datetime(m.date).dt.weekday; m["month"] = pd.to_datetime(m.date).dt.month
WXF = ["low wind (daily mean)", "low mixing height (daily mean)", "large day-night temp swing", "lower-tropospheric stability",
       "inversion 925 hPa at 6 AM", "hours downwind of complex (wind-direction analysis)"]
def trained(y, kind):
    pred = pd.Series(np.nan, index=m.index)
    for s in SITES:
        for yr in m.yr.unique():
            tr = (m.site == s) & (m.yr != yr) & m[y].notna(); te = (m.site == s) & (m.yr == yr)
            if kind == "seasonal":
                means = m.loc[tr].groupby("month")[y].mean(); pred[te] = m.loc[te, "month"].map(means).values; continue
            feats = WXF + (["doy_sin", "doy_cos", "weekday"] if kind == "rf" else [])
            Xtr, Xte = m.loc[tr, feats], m.loc[te, feats]; ytr = m.loc[tr, y].rank(pct=True)
            if kind == "rf":
                mod = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0).fit(Xtr, ytr)
                pred[te] = mod.predict(Xte)
            else:
                mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
                pred[te] = Ridge(alpha=10).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)
    return pred

def wrho(d, col, y):
    vals = []
    for _, g in d.groupby("site"):
        g = g.dropna(subset=[col, y])
        if g[col].nunique() > 1 and g[y].nunique() > 1:
            vals.append(spearmanr(g[col], g[y])[0])
    return float(np.mean(vals))
out = {"n_monitor_days": int(len(m)), "missing_index_values": int(m[names].isna().sum().sum())}
dates = m.date.unique(); boot_dates = [rng.choice(dates, len(dates)) for _ in range(1000)]
TRAINED = {"seasonal pattern only (same month, other years)": "seasonal", "ridge trained on Calvert monitors": "ridge",
           "random forest trained on Calvert monitors (rmweather-style)": "rf"}
for y in ["vc", "benzene", "total_voc"]:
    for name, kind in TRAINED.items(): m[name] = trained(y, kind)
    byd = {k: v for k, v in m.groupby("date")}; boots = [pd.concat([byd[k] for k in bd]) for bd in boot_dates]
    d = m.dropna(subset=[y]); allc = cands + list(TRAINED)
    r = {c: round(wrho(d, c, y), 3) for c in allc}
    best = max((c for c in cands if c != "odor forecast (ours)"), key=lambda c: r[c])
    diffs = [wrho(b.dropna(subset=[y]), "odor forecast (ours)", y) - wrho(b.dropna(subset=[y]), best, y) for b in boots]
    best_tr = max(TRAINED, key=lambda c: r[c])
    dtr = [wrho(b.dropna(subset=[y]), "odor forecast (ours)", y) - wrho(b.dropna(subset=[y]), best_tr, y) for b in boots]
    out[y + "_vs_trained"] = {"best_trained": best_tr, "ours_minus_best_trained": round(r["odor forecast (ours)"] - r[best_tr], 3),
                              "ci95": [round(float(np.percentile(dtr, 2.5)), 3), round(float(np.percentile(dtr, 97.5)), 3)]}
    out[y] = {"rho": dict(sorted(r.items(), key=lambda kv: -kv[1])), "best_index": best,
              "ours_minus_best": round(r["odor forecast (ours)"] - r[best], 3),
              "ours_minus_best_ci95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]}
# Trapping-only measures (no wind-direction information): ours vs the best single one, and vs their composite
TRAP = [c for c in names if "downwind" not in c]
m["composite of trapping measures only"] = m.groupby("site")[TRAP].rank(pct=True).mean(axis=1)
byd = {k: v for k, v in m.groupby("date")}; boots = [pd.concat([byd[k] for k in bd]) for bd in boot_dates]
for y in ["vc", "benzene", "total_voc"]:
    d = m.dropna(subset=[y]); rr = {c: wrho(d, c, y) for c in TRAP + ["composite of trapping measures only", "odor forecast (ours)"]}
    bt = max(TRAP, key=lambda c: rr[c])
    res = {}
    for comp in [bt, "composite of trapping measures only"]:
        dd = [wrho(b.dropna(subset=[y]), "odor forecast (ours)", y) - wrho(b.dropna(subset=[y]), comp, y) for b in boots]
        res[comp] = {"rho_comp": round(rr[comp], 3), "ours_minus": round(rr["odor forecast (ours)"] - rr[comp], 3),
                     "ci95": [round(float(np.percentile(dd, 2.5)), 3), round(float(np.percentile(dd, 97.5)), 3)]}
    out[y + "_vs_trapping_only"] = {"best_single_trapping_measure": bt, **res}
    print(y, json.dumps(out[y + "_vs_trapping_only"]))
m.to_csv("fair_baselines_monitor_days.csv", index=False)
json.dump(out, open("fair_baselines.json", "w"), indent=1)
print("monitor-days", out["n_monitor_days"], "| missing index values", out["missing_index_values"])
for y in ["vc", "benzene", "total_voc"]:
    print(f"\n{y}: best index = {out[y]['best_index']}  ours-minus-best = {out[y]['ours_minus_best']} CI {out[y]['ours_minus_best_ci95']}")
    print(f"   vs trained: {out[y + '_vs_trained']}")
    for k, v in out[y]["rho"].items(): print(f"   {v:+.3f}  {k}")
