"""Score the ORI refit at the Calvert monitors: refit/REFIT_PLAN.md s8 (frozen, SHA-256 a01ca278...), DEVIATIONS 23-26.

Inputs (logged in DEVIATIONS before this runs): refit_inputs/ (built by refit_calvert_inputs.py), monitor_days.csv and
forecast_standard_source.csv (the deployed model's existing results), and the refit itself
(~/weather-varaible-analysis/training_panel_v2/fit/). The primary and secondary statistics are run_extended.py's own
code, copied unchanged below and applied to a `forecast` column; the deployed model's results are not recomputed.

Outputs: refit_calvert_results.json, REFIT_CALVERT_RESULTS.md, refit_calvert_monitor_days.csv.
Run: ~/weather-varaible-analysis/.venv/bin/python score_refit.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "/path/to/weather-varaible-analysis")
import odor_forecast_core as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "refit_inputs")
FIT = "/path/to/weather-varaible-analysis/training_panel_v2/fit"
SEED = 20261028
rng = np.random.default_rng(SEED)
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879),
         "lwd": (37.047914, -88.338323)}
RES = ["elementary", "johnson_riley"]
TZ = "America/Chicago"
B_PRIMARY, B_PAIRED = 10000, 10000
TERMS = ["temp_c", "temp_c2", "diurnal_temperature_range", "relative_humidity", "wind_speed", "boundary_layer_height",
         "solar_radiation", "precipitation", "pressure_anomaly", "multi_source_exposure", "wind_align_weighted"]


# ------------------------------------------------------------------ the refit

def load_refit():
    coefs = pd.read_csv(os.path.join(FIT, "coefficients.csv"))
    manifest = json.load(open(os.path.join(FIT, "run_manifest.json")))
    res = json.load(open(os.path.join(FIT, "results.json")))
    fits = {f: coefs[coefs.fit == f].set_index("term").coef for f in ("main", "city_pgh_only", "city_lou_only")}
    df = res["deployment_form"]
    const = df["calvert_intercept_two_city_2025_mean"] + df["average_day_of_week_and_holiday_effect"]
    return fits, manifest["temperature_centre_F"], const, manifest


def with_source_terms(df, lat, lon):
    """Calvert proximity index and wind alignment for a location (the deployed code's own function, s8)."""
    t = [M.compute_multisource_terms(lat, lon, w) for w in df.wind_direction]
    return df.assign(multi_source_exposure=[x[0] for x in t], wind_align_weighted=[x[1] for x in t])


def refit_lp(df, b, t0):
    """Linear predictor without the constant (intercept and average weekday effect are one number at Calvert)."""
    x = {"temp_c": df.temperature - t0, "temp_c2": (df.temperature - t0) ** 2}
    lp = 0.0
    for t in TERMS:
        v = x[t] if t in x else df[t]
        if np.any(pd.isna(v)):
            raise ValueError(f"missing {t}: refusing to score")
        lp = lp + b[t] * v
    return np.asarray(lp, float)


def deployed_pct(df):
    """The deployed model through its own code (predict_ori, pressure transfer on), as in run_extended.py."""
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        row.update(mse=r.multi_source_exposure, msa=r.wind_align_weighted)
        out.append(M.predict_ori(row, M.COEFFS_POOLED_PROXIMITY, use_wind_filter=False))
    return np.asarray(out, float)


# ------------------------------------------------------------------ report-free comparators (M3)
# Definitions copied from fair_baselines.py (2026-10-06); inputs are the ERA5 hourly comparator file.

def comparator_indices(h, lat, lon):
    h = h.copy()
    loc = h.index.tz_localize("UTC").tz_convert(TZ)
    h["date"] = loc.strftime("%Y-%m-%d")
    h["hr"] = loc.hour
    h["w10_ms"] = h.wind_speed_10m * 0.44704
    h["blh_m"] = h.boundary_layer_height / 3.28084
    h["t2k"] = (h.temperature_2m - 32) * 5 / 9 + 273.15
    h["p_sfc"] = h.surface_pressure
    brg = M._source_bearing(M.IND_LAT, M.IND_LON, lat, lon)
    diff = ((h.wind_direction_10m - (brg + 180) % 360 + 180) % 360) - 180
    h["downwind"] = (diff.abs() <= 30).astype(float)
    h["downwind_calm"] = ((diff.abs() <= 30) & (h.wind_speed_10m * 0.44704 < 3.0)).astype(float)
    h["inv925"] = h.t925 - h.t2k
    h["inv850"] = h.t850 - h.t2k
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
    return day


def complete_comparator_days(h):
    cols = ["temperature_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "rain",
            "boundary_layer_height", "w100_ms", "t925", "t850", "wind500"]
    loc = h.index.tz_localize("UTC").tz_convert(TZ)
    d = pd.Series(loc.strftime("%Y-%m-%d"), index=h.index)
    ok = h[cols].notna().all(axis=1).groupby(d).all()
    n = d.groupby(d).size()
    return set(ok[ok].index) & set(n[(n >= 23) & (n <= 25)].index)


# ------------------------------------------------------------------ run_extended.py statistics, unchanged

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
    return {"rho": round(obs, 3), "p": round((c + 1) / (B + 1), 4), "shared_dates": int(len(dates)),
            "monitor_days": int(len(d))}


def site_perm(g, y, B=5000):
    g = g.dropna(subset=[y]); obs = spearmanr(g.forecast, g[y])[0]; c = 0
    for _ in range(B): c += abs(spearmanr(rng.permutation(g.forecast.values), g[y])[0]) >= abs(obs)
    return {"rho": round(float(obs), 3), "p": round((c + 1) / (B + 1), 4), "n": int(len(g))}


def boot_auc(d, B=2000):
    ds = d.date.unique(); g = {k: v for k, v in d.groupby("date")}; vals = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(ds, len(ds))])
        if 0 < s.hi.sum() < len(s): vals.append(roc_auc_score(s.hi, s.fc))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def analysis_plan(data, new):
    """ANALYSIS_PLAN.md primaries and secondaries 1-7 for the `forecast` column (run_extended.py code)."""
    out = {"counts": new[new.site.isin(RES)].groupby("site").size().to_dict()}
    prim = {k: joint_date_perm(new, k, B_PRIMARY) for k in ["vc", "benzene"]}
    rej, padj, _, _ = multipletests([prim["vc"]["p"], prim["benzene"]["p"]], method="holm")
    for k, pa in zip(["vc", "benzene"], padj): prim[k]["p_holm"] = round(float(pa), 4)
    ok = [prim[k]["rho"] > 0 and prim[k]["p_holm"] < 0.05 for k in ["vc", "benzene"]]
    out["primary"] = prim
    out["decision"] = "SUPPORTED" if all(ok) else ("PARTIALLY SUPPORTED" if any(ok) else "NOT SUPPORTED")
    sec = {}
    sec["per_monitor"] = {s: {k: site_perm(new[new.site == s], k) for k in ["vc", "benzene"]} for s in RES}
    sec["site_adjusted_auc"] = {}
    for k in ["vc", "benzene", "total_voc"]:
        d = new[new.site.isin(RES)].dropna(subset=[k]).copy()
        d["hi"] = (d[k] >= d.groupby("site")[k].transform(lambda x: x.quantile(0.75))).astype(int)
        if k == "vc": d["hi"] = ((d[k] >= d.groupby("site")[k].transform(lambda x: x.quantile(0.75))) & (d[k] > 0)).astype(int)
        d["fc"] = d.forecast - d.groupby("site").forecast.transform("mean")
        sec["site_adjusted_auc"][k] = {"auc": round(roc_auc_score(d.hi, d.fc), 3), "ci": boot_auc(d),
                                       "events": int(d.hi.sum()), "n": len(d)}
    sec["other_pollutants"] = {k: joint_date_perm(new, k, B=5000) for k in ["total_voc", "butadiene", "edc"]}
    sec["all_dates_incl_poster"] = {k: joint_date_perm(data, k, B=5000) for k in ["vc", "benzene"]}
    sec["by_year"] = {yr: {k: joint_date_perm(new[new.year == yr], k, B=3000) for k in ["vc", "benzene"]}
                      for yr in sorted(new.year.unique())}
    d = new[new.site.isin(RES)].dropna(subset=["vc"]).copy()
    d["hi_fc"] = d.forecast > d.groupby("site").forecast.transform("median"); d["det"] = d.vc > 0
    tabs = [np.array([[(g.hi_fc & g.det).sum(), (g.hi_fc & ~g.det).sum()],
                      [(~g.hi_fc & g.det).sum(), (~g.hi_fc & ~g.det).sum()]]) + 0.5 for _, g in d.groupby("site")]
    sec["vc_detection"] = {s: {"hi": f"{int((g.hi_fc & g.det).sum())}/{int(g.hi_fc.sum())}",
                               "lo": f"{int((~g.hi_fc & g.det).sum())}/{int((~g.hi_fc).sum())}"} for s, g in d.groupby("site")}
    sec["vc_detection"]["cmh_p"] = round(float(StratifiedTable(tabs).test_null_odds().pvalue), 4)
    sec["vc_detection"]["cmh_or"] = round(float(StratifiedTable(tabs).oddsratio_pooled), 2)
    sec["lwd"] = {k: site_perm(new[new.site == "lwd"], k) for k in ["vc", "benzene"]}
    out["secondary"] = sec
    return out


# ------------------------------------------------------------------ paired comparisons (M2, M3)

def _rho(a, y):
    return spearmanr(a, y)[0]


def paired(new, y, col_a, col_b, B=B_PAIRED):
    """Refit (col_a) minus comparator (col_b): mean of the two residential monitors' within-monitor Spearman rho,
    identical monitor-days with valid samples at both monitors. 95% CI from B bootstrap resamples of dates (joint across
    monitors); two-sided p from B random swaps of the two forecasts within dates (same swap at both monitors), done on
    within-monitor percentile ranks so the swap compares like with like ((count + 1) / (B + 1))."""
    d = new[new.site.isin(RES)].dropna(subset=[y, col_a, col_b])
    both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)].sort_values(["site", "date"])
    dates = np.sort(d.date.unique())
    A, Bc, Y = {}, {}, {}
    for s in RES:
        g = d[d.site == s].set_index("date").loc[dates]
        A[s], Bc[s], Y[s] = rankdata(g[col_a]) / len(g), rankdata(g[col_b]) / len(g), g[y].to_numpy(float)

    def delta(idx=None, swap=None):
        da, db = [], []
        for s in RES:
            a, b, yy = A[s], Bc[s], Y[s]
            if swap is not None:
                a, b = np.where(swap, Bc[s], A[s]), np.where(swap, A[s], Bc[s])
            if idx is not None:
                a, b, yy = a[idx], b[idx], yy[idx]
            if np.unique(yy).size < 2:
                return np.nan
            da.append(_rho(a, yy)); db.append(_rho(b, yy))
        return float(np.mean(da) - np.mean(db)), float(np.mean(da)), float(np.mean(db)), [float(x) for x in da], [float(x) for x in db]

    obs, ra, rb, pa, pb = delta()
    n = len(dates)
    boots = [delta(idx=rng.integers(0, n, n)) for _ in range(B)]
    boots = np.array([b[0] if isinstance(b, tuple) else np.nan for b in boots])
    swaps = np.array([delta(swap=rng.random(n) < 0.5)[0] for _ in range(B)])
    p = (np.sum(np.abs(swaps) >= abs(obs)) + 1) / (B + 1)
    return {"refit_rho": round(ra, 3), "comparator_rho": round(rb, 3), "diff": round(obs, 3),
            "ci95": [round(float(np.nanpercentile(boots, 2.5)), 3), round(float(np.nanpercentile(boots, 97.5)), 3)],
            "p_swap": round(float(p), 4), "shared_dates": int(n),
            "per_monitor": {s: {"refit": round(pa[i], 3), "comparator": round(pb[i], 3)} for i, s in enumerate(RES)}}


def describe(new, col, B=3000):
    """Within-monitor rho (and joint date-permutation p) for a forecast column, both pollutants."""
    out = {}
    for y in ["vc", "benzene"]:
        d = new.dropna(subset=[col]).assign(forecast=lambda x: x[col])
        out[y] = joint_date_perm(d, y, B)
        out[y]["per_monitor"] = {s: round(float(spearmanr(g.forecast, g[y])[0]), 3)
                                 for s, g in d[d.site.isin(RES)].dropna(subset=[y]).groupby("site")}
    return out


# ------------------------------------------------------------------ main

def main():
    fits, t0, calvert_const, fit_manifest = load_refit()
    b = fits["main"]
    md = pd.read_csv(os.path.join(HERE, "monitor_days.csv")).rename(columns={"forecast": "f_asrun"})
    std = pd.read_csv(os.path.join(HERE, "forecast_standard_source.csv"))[["date", "site", "forecast_std"]]
    md = md.merge(std.rename(columns={"forecast_std": "f_corrected"}), on=["date", "site"], how="left")

    # refit on ERA5 cell inputs (primary)
    cd = pd.read_csv(os.path.join(INP, "era5_cell_daily.csv"))
    rows = []
    for s, (lat, lon) in SITES.items():
        g = with_source_terms(cd[(cd.date >= "2023-01-01") & (cd.date <= "2025-12-31")].copy(), lat, lon)
        g["lp_refit"] = refit_lp(g, b, t0)
        g["f_refit"] = 100 * expit(calvert_const + g.lp_refit)
        g["lp_pgh_only"] = refit_lp(g, fits["city_pgh_only"], t0)
        g["lp_lou_only"] = refit_lp(g, fits["city_lou_only"], t0)
        g["f_deployed_on_era5"] = deployed_pct(g)
        rows.append(g.assign(site=s)[["date", "site", "f_refit", "lp_refit", "lp_pgh_only", "lp_lou_only",
                                      "f_deployed_on_era5", "multi_source_exposure"]])
    fc = pd.concat(rows)
    data = md.merge(fc, on=["date", "site"], how="left", validate="one_to_one")
    if data.f_refit.isna().any():
        raise ValueError("a monitor-day has no ERA5 refit forecast")

    # refit on the deployed model's inputs (sensitivity); pressure anomaly against each series' own 2023-2025 mean
    dep = pd.read_csv(os.path.join(INP, "deployed_inputs_daily.csv"))
    rows, check = [], []
    for s, (lat, lon) in SITES.items():
        g = with_source_terms(dep[dep.site == s].copy(), lat, lon)
        g["pressure_anomaly"] = g.atmospheric_pressure - g.atmospheric_pressure.mean()
        g["lp_refit_dep_inputs"] = refit_lp(g, b, t0)
        g["f_deployed_recomputed"] = deployed_pct(g)
        rows.append(g[["date", "site", "lp_refit_dep_inputs", "f_deployed_recomputed"]])
    data = data.merge(pd.concat(rows), on=["date", "site"], how="left", validate="one_to_one")
    recompute_gap = float((data.f_deployed_recomputed - data.f_corrected).abs().max())

    # refit on archived ECMWF IFS lead-0 forecasts (secondary 8)
    ifs = pd.read_csv(os.path.join(INP, "ifs_lead0_daily.csv"))
    rows = []
    for s in RES:
        lat, lon = SITES[s]
        g = with_source_terms(ifs[ifs.site == s].copy(), lat, lon)
        g["pressure_anomaly"] = g.atmospheric_pressure - g.atmospheric_pressure.mean()
        g["lp_refit_ifs0"] = refit_lp(g, b, t0)
        rows.append(g[["date", "site", "lp_refit_ifs0"]])
    data = data.merge(pd.concat(rows), on=["date", "site"], how="left", validate="one_to_one")

    # report-free comparators (M3) on ERA5 inputs
    hc = pd.read_parquet(os.path.join(INP, "era5_comparator_hourly.parquet"))
    hc.index = pd.to_datetime(hc.index)
    complete = complete_comparator_days(hc)
    idx = []
    for s, (lat, lon) in SITES.items():
        day = comparator_indices(hc, lat, lon)
        day = day[(day.index >= "2023-01-01") & (day.index <= "2025-12-31")]
        idx.append(day.assign(site=s).rename_axis("date").reset_index())
    idx = pd.concat(idx)
    data = data.merge(idx, on=["date", "site"], how="left", validate="one_to_one")
    names = [c for c in idx.columns if c not in ("date", "site")]
    trap = [c for c in names if "downwind" not in c]
    new = data[~data.poster_window].copy()
    res_new = new.site.isin(RES)
    missing = sorted(set(new.loc[res_new, "date"]) - complete)
    if missing:
        raise ValueError(f"comparator inputs incomplete on scored dates {missing[:5]}")
    new.loc[res_new, "c_trap"] = new[res_new].groupby("site")[trap].rank(pct=True).mean(axis=1)
    new["c_downwind"] = new["hours downwind of complex (wind-direction analysis)"]

    results = {"seed": SEED, "refit_run": fit_manifest["run_utc"], "refit_script": fit_manifest["script_sha256"],
               "calvert_constant_logit": calvert_const, "temperature_centre_F": t0,
               "deployed_inputs_recomputed_max_abs_diff_pct_points": recompute_gap}

    # ---- the plan for the refit (primaries, decision, secondaries 1-7)
    print("primaries and secondaries", flush=True)
    data["forecast"] = data.f_refit
    new["forecast"] = new.f_refit
    results["refit_analysis_plan"] = analysis_plan(data, new)

    # ---- paired statistics: comparator (M1, M2) and report-free comparators (M3)
    print("paired comparisons", flush=True)
    results["paired"] = {
        "vc": {"vs_corrected_deployed": paired(new, "vc", "f_refit", "f_corrected"),
               "vs_as_run_deployed": paired(new, "vc", "f_refit", "f_asrun"),
               "vs_hours_downwind": paired(new, "vc", "f_refit", "c_downwind")},
        "benzene": {"vs_corrected_deployed": paired(new, "benzene", "f_refit", "f_corrected"),
                    "vs_as_run_deployed": paired(new, "benzene", "f_refit", "f_asrun"),
                    "vs_trapping_composite": paired(new, "benzene", "f_refit", "c_trap")}}

    # ---- model-by-inputs (s8), split at 2024-09 for the deployed-input runs
    print("model by inputs", flush=True)
    two = {"refit_on_era5": "f_refit", "refit_on_deployed_inputs": "lp_refit_dep_inputs",
           "deployed_on_era5": "f_deployed_on_era5", "deployed_on_deployed_inputs": "f_corrected"}
    results["model_by_inputs"] = {k: describe(new, c) for k, c in two.items()}
    results["deployed_inputs_split_2024_09"] = {
        period: {k: describe(sub, two[k], B=2000) for k in ("refit_on_deployed_inputs", "deployed_on_deployed_inputs")}
        for period, sub in (("before_2024_09", new[new.date < "2024-09-01"]), ("from_2024_09", new[new.date >= "2024-09-01"]))}

    # ---- secondaries: per-city fits (M8), IFS lead 0 (M15)
    print("per-city fits and IFS lead 0", flush=True)
    results["per_city_fits"] = {"pittsburgh_only": describe(new, "lp_pgh_only"),
                                "louisville_only": describe(new, "lp_lou_only")}
    ifs_days = new.dropna(subset=["lp_refit_ifs0"])
    results["ifs_lead0"] = {"refit_on_ifs_lead0": describe(ifs_days, "lp_refit_ifs0"),
                            "refit_on_era5_same_days": describe(ifs_days, "f_refit"),
                            "first_date": ifs_days.date.min(), "last_date": ifs_days.date.max(),
                            "note": "Leads >= 1 not scored: the archive has no boundary-layer height for ecmwf_ifs at "
                                    "any lead past day 0, and filling it from another lead would mix sources."}

    # ---- local Calvert reports, descriptive (M20)
    results["local_reports_2026"] = local_reports_line(b, t0)

    data_out = data.merge(new[["date", "site", "c_trap"]], on=["date", "site"], how="left")
    data_out.to_csv(os.path.join(HERE, "refit_calvert_monitor_days.csv"), index=False)
    json.dump(results, open(os.path.join(HERE, "refit_calvert_results.json"), "w"), indent=1, default=float)
    write_report(results)
    print("done", flush=True)


def local_reports_line(b, t0):
    """M20: where does the refit's index rank each local odor report's day among that spot's 2026 days? Descriptive."""
    try:
        rep = pd.read_csv(os.path.join(INP, "local_reports_2026.csv"))
    except pd.errors.EmptyDataError:
        rep = pd.DataFrame()
    if rep.empty:
        return {"note": "no local reports returned by the Smell MyCity API for the window; nothing to describe"}
    cd = pd.read_csv(os.path.join(INP, "era5_cell_daily_2026.csv"))
    mean_p = json.load(open(os.path.join(INP, "era5_cell_provenance.json")))["pressure_mean_hpa_2018_2026"]
    cd["pressure_anomaly"] = cd.atmospheric_pressure - mean_p
    odor = rep[rep.smell_value >= 2]
    pct = []
    for _, r in odor.iterrows():
        g = with_source_terms(cd.copy(), r.latitude, r.longitude)
        lp = pd.Series(refit_lp(g, b, t0), index=g.date)
        if r.date in lp.index:
            pct.append(float((lp < lp[r.date]).mean() + 0.5 * (lp == lp[r.date]).mean()))
    return {"window": [cd.date.min(), cd.date.max()], "reports_within_20_mi": int(len(rep)),
            "odor_reports_value_2_to_5": int(len(odor)), "odor_report_days": int(odor.date.nunique()),
            "odor_reports_scored": len(pct), "zip_codes": odor.zipcode.astype(str).value_counts().to_dict(),
            "median_percentile_of_report_day": round(float(np.median(pct)), 3) if pct else None,
            "iqr": [round(float(np.percentile(pct, 25)), 3), round(float(np.percentile(pct, 75)), 3)] if pct else None,
            "note": "Descriptive only: few reports, mostly one household; 0.5 = report days rank like any other day."}


def write_report(r):
    rp = r["refit_analysis_plan"]
    L = ["# ORI refit scored at the Calvert monitors", "",
         f"Refit run {r['refit_run']} (`fit_refit.py` {r['refit_script'][:12]}); scored under `refit/REFIT_PLAN.md` s8 "
         f"and `ANALYSIS_PLAN.md`; seed {r['seed']}. The deployed model's pre-registered results are unchanged. The refit's "
         "verdict is a pre-specified secondary re-analysis of monitor-days already scored several ways, not a confirmation.",
         "", "## Primary endpoints (refit on ERA5 inputs)", "",
         "| endpoint | mean within-monitor rho | p (joint date permutation) | Holm p | shared dates | monitor-days |",
         "|---|---|---|---|---|---|"]
    for k, lab in (("vc", "P1 vinyl chloride"), ("benzene", "P2 benzene")):
        x = rp["primary"][k]
        L.append(f"| {lab} | {x['rho']:.3f} | {x['p']:.4f} | {x['p_holm']:.4f} | {x['shared_dates']} | {x['monitor_days']} |")
    L += ["", f"Decision rule: **{rp['decision']}** (secondary re-analysis).", ""]
    L += ["## Paired comparisons (refit minus comparator, identical monitor-days)", "",
          "| pollutant | comparator | refit rho | comparator rho | difference | 95% CI | p (swap) | shared dates |",
          "|---|---|---|---|---|---|---|---|"]
    lab = {"vs_corrected_deployed": "deployed, corrected (primary comparator)",
           "vs_as_run_deployed": "deployed, as run (secondary)", "vs_hours_downwind": "hours downwind of the complex",
           "vs_trapping_composite": "trapping composite"}
    for y in ("vc", "benzene"):
        for k, x in r["paired"][y].items():
            L.append(f"| {y} | {lab[k]} | {x['refit_rho']:.3f} | {x['comparator_rho']:.3f} | {x['diff']:+.3f} | "
                     f"{x['ci95'][0]:+.3f} to {x['ci95'][1]:+.3f} | {x['p_swap']:.4f} | {x['shared_dates']} |")
    L += ["", "## Model by inputs (mean within-monitor rho; joint date-permutation p)", "",
          "| model and inputs | vinyl chloride | benzene |", "|---|---|---|"]
    for k, x in r["model_by_inputs"].items():
        L.append(f"| {k.replace('_', ' ')} | {x['vc']['rho']:.3f} (p {x['vc']['p']:.3f}) | "
                 f"{x['benzene']['rho']:.3f} (p {x['benzene']['p']:.3f}) |")
    L += ["", "Deployed-input runs split at 2024-09 (when that source's BLH switched from ERA5 to IFS):", "",
          "| period | model | vinyl chloride | benzene |", "|---|---|---|---|"]
    for per, xs in r["deployed_inputs_split_2024_09"].items():
        for k, x in xs.items():
            L.append(f"| {per} | {k.replace('_', ' ')} | {x['vc']['rho']:.3f} ({x['vc']['shared_dates']} dates) | "
                     f"{x['benzene']['rho']:.3f} ({x['benzene']['shared_dates']} dates) |")
    L += ["", f"Check: the deployed model recomputed from its own inputs matches the existing corrected forecast within "
              f"{r['deployed_inputs_recomputed_max_abs_diff_pct_points']:.3f} percentage points.", ""]
    L += ["## Secondaries", "", "| secondary | vinyl chloride | benzene |", "|---|---|---|"]
    sec = rp["secondary"]
    for s in RES:
        L.append(f"| monitor {s} (rho, p) | {sec['per_monitor'][s]['vc']['rho']:.3f}, {sec['per_monitor'][s]['vc']['p']:.4f} | "
                 f"{sec['per_monitor'][s]['benzene']['rho']:.3f}, {sec['per_monitor'][s]['benzene']['p']:.4f} |")
    L.append(f"| site-adjusted AUC, high days (95% CI) | {sec['site_adjusted_auc']['vc']['auc']:.3f} "
             f"({sec['site_adjusted_auc']['vc']['ci'][0]:.3f}-{sec['site_adjusted_auc']['vc']['ci'][1]:.3f}) | "
             f"{sec['site_adjusted_auc']['benzene']['auc']:.3f} ({sec['site_adjusted_auc']['benzene']['ci'][0]:.3f}-"
             f"{sec['site_adjusted_auc']['benzene']['ci'][1]:.3f}) |")
    L.append(f"| all dates incl. poster (rho, p) | {sec['all_dates_incl_poster']['vc']['rho']:.3f}, "
             f"{sec['all_dates_incl_poster']['vc']['p']:.4f} | {sec['all_dates_incl_poster']['benzene']['rho']:.3f}, "
             f"{sec['all_dates_incl_poster']['benzene']['p']:.4f} |")
    for yr, x in sec["by_year"].items():
        L.append(f"| {yr} (rho, p) | {x['vc']['rho']:.3f}, {x['vc']['p']:.4f} | {x['benzene']['rho']:.3f}, {x['benzene']['p']:.4f} |")
    L.append(f"| LWD (rho, p) | {sec['lwd']['vc']['rho']:.3f}, {sec['lwd']['vc']['p']:.4f} | {sec['lwd']['benzene']['rho']:.3f}, "
             f"{sec['lwd']['benzene']['p']:.4f} |")
    for k, x in r["per_city_fits"].items():
        L.append(f"| {k.replace('_', ' ')} fit (rho, p) | {x['vc']['rho']:.3f}, {x['vc']['p']:.4f} | "
                 f"{x['benzene']['rho']:.3f}, {x['benzene']['p']:.4f} |")
    i0 = r["ifs_lead0"]
    L.append(f"| refit on archived IFS lead 0, {i0['first_date']} to {i0['last_date']} (rho) | "
             f"{i0['refit_on_ifs_lead0']['vc']['rho']:.3f} (ERA5 same days {i0['refit_on_era5_same_days']['vc']['rho']:.3f}) | "
             f"{i0['refit_on_ifs_lead0']['benzene']['rho']:.3f} (ERA5 same days {i0['refit_on_era5_same_days']['benzene']['rho']:.3f}) |")
    L += ["", "Other pollutants (rho, p): " + ", ".join(f"{k} {x['rho']:.3f} (p {x['p']:.4f})"
                                                     for k, x in sec["other_pollutants"].items()) +
          f". Vinyl chloride detection by forecast median split: CMH OR {sec['vc_detection']['cmh_or']}, "
          f"p {sec['vc_detection']['cmh_p']}.", ""]
    lr = r["local_reports_2026"]
    L += ["## Local Calvert odor reports, 2026 (descriptive)", ""]
    if "window" in lr:
        L += [f"{lr['odor_reports_value_2_to_5']} odor reports (value 2-5) on {lr['odor_report_days']} days within 20 "
              f"miles of the complex, {lr['window'][0]} to {lr['window'][1]}. Median percentile of a report day's refit "
              f"index among that spot's days: {lr['median_percentile_of_report_day']} (IQR {lr['iqr']}). {lr['note']}", ""]
    else:
        L += [lr["note"], ""]
    open(os.path.join(HERE, "REFIT_CALVERT_RESULTS.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
