"""Plan secondary item 8: forecast-lead sensitivity with archived forecasts (BLH from the lead-0 run)."""
import sys, json, time, numpy as np, pandas as pd, requests
sys.path.insert(0, "/path/to/weather-varaible-analysis"); import odor_forecast_core as M
from scipy.stats import spearmanr
rng = np.random.default_rng(8)
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879)}
VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "rain", "shortwave_radiation", "boundary_layer_height"]
md = pd.read_csv("monitor_days.csv"); md = md[md.site.isin(SITES) & ~md.poster_window]
def get(url, p):
    for a in range(5):
        try:
            r = requests.get(url, params=p, timeout=120); r.raise_for_status(); return r.json()["hourly"]
        except Exception: time.sleep(5 * (a + 1))
def fetch(lat, lon, lead):
    base = {"latitude": lat, "longitude": lon, "start_date": "2023-01-01", "end_date": "2025-12-31",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "America/Chicago"}
    h0 = get("https://historical-forecast-api.open-meteo.com/v1/forecast", dict(base, hourly=",".join(VARS)))
    if lead == 0: h = h0
    else:
        names = [f"{v}_previous_day{lead}" for v in VARS if v != "boundary_layer_height"]
        hl = get("https://previous-runs-api.open-meteo.com/v1/forecast", dict(base, hourly=",".join(names)))
        h = {v: hl[f"{v}_previous_day{lead}"] for v in VARS if v != "boundary_layer_height"}
        h["boundary_layer_height"] = h0["boundary_layer_height"]; h["time"] = hl["time"]
    df = pd.DataFrame({v: h[v] for v in VARS}); df["time"] = pd.to_datetime(h["time"]); df["date"] = df.time.dt.strftime("%Y-%m-%d")
    df["blh_ft"] = df.boundary_layer_height * 3.28084; df["precip_in"] = df.rain / 25.4
    df["u"] = df.wind_speed_10m * np.sin(np.radians(df.wind_direction_10m)); df["v"] = df.wind_speed_10m * np.cos(np.radians(df.wind_direction_10m))
    g = df.groupby("date").agg(temperature=("temperature_2m", "mean"), tmin=("temperature_2m", "min"), tmax=("temperature_2m", "max"),
        precipitation=("precip_in", "sum"), wind_speed=("wind_speed_10m", "mean"), u=("u", "mean"), v=("v", "mean"),
        relative_humidity=("relative_humidity_2m", "mean"), atmospheric_pressure=("surface_pressure", "mean"),
        solar_radiation=("shortwave_radiation", "mean"), boundary_layer_height=("blh_ft", "mean"),
        complete=("temperature_2m", lambda x: x.notna().all()))
    g["temperature_squared"] = g.temperature ** 2; g["diurnal_temperature_range"] = g.tmax - g.tmin
    g["wind_direction"] = np.degrees(np.arctan2(g.u, g.v)) % 360
    return g
def within_rho(d, y): return float(np.mean([spearmanr(g.f, g[y])[0] for _, g in d.groupby("site")]))
res = {}
for lead in [0, 1, 3, 5]:
    parts = []
    for s, (lat, lon) in SITES.items():
        g = fetch(lat, lon, lead); sub = md[md.site == s].copy()
        def sc(r):
            if r.date not in g.index or not g.loc[r.date, "complete"] or np.isnan(g.loc[r.date, "boundary_layer_height"]): return np.nan
            row = g.loc[r.date].to_dict(); mse, msa = M.compute_multisource_terms(lat, lon, row["wind_direction"]); row.update(mse=mse, msa=msa)
            return M.predict_ori(row, M.COEFFS_POOLED_PROXIMITY, use_wind_filter=False)
        sub["f"] = sub.apply(sc, axis=1); parts.append(sub)
    d = pd.concat(parts).dropna(subset=["f"])
    both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)]
    r = {"monitor_days": len(d), "first_date": d.date.min(), "corr_with_era5": round(float(np.corrcoef(d.f, d.forecast)[0, 1]), 3)}
    piv = d.pivot(index="date", columns="site", values="f"); dates = piv.index.values; idx = {dt: i for i, dt in enumerate(dates)}
    for y in ["vc", "benzene"]:
        obs = within_rho(d, y); c = 0
        for _ in range(3000):
            p = rng.permutation(len(dates)); t = d.assign(f=[piv[s].values[p[idx[dt]]] for s, dt in zip(d.site, d.date)])
            c += abs(within_rho(t, y)) >= abs(obs)
        r[y] = {"rho": round(obs, 3), "p": round((c + 1) / 3001, 4)}
    res[f"lead_{lead}"] = r; print(lead, r, flush=True)
res["note"] = "Plan secondary item 8. Archived forecasts (Open-Meteo historical-forecast and previous-runs APIs); boundary-layer height from the lead-0 run at leads >= 1."
json.dump(res, open("lead_sensitivity.json", "w"), indent=1)
