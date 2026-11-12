"""NOV2026_PLAN.md parts A and B (frozen 2026-11-02, SHA-256 0cca00e6...; DEVIATIONS 33-34).

A. ORI combined with hours downwind: equal-weight mean of within-monitor percentile ranks, no fitting.
B. A weather model fitted at Calvert: ridge (alpha 1.0) on the refit's 11 ERA5 terms, outcome = within-monitor
   percentile rank of the pollutant, leave-one-calendar-year-out; final models saved for part C.

Statistics are imported from score_refit.py (unchanged). Seed 20261102.
Inputs: refit_calvert_monitor_days.csv (item 26 output), refit_inputs/era5_cell_daily.csv.
Outputs: nov2026_ab_results.json, NOV2026_AB_RESULTS.md, nov2026_ab_monitor_days.csv, calvert_fitted_models.json.
Run: ~/weather-varaible-analysis/.venv/bin/python score_nov2026_ab.py
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

import score_refit as S

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261102
S.rng = np.random.default_rng(SEED)          # the imported statistics draw from this generator
RES = S.RES
HD = "hours downwind of complex (wind-direction analysis)"
TERMS = list(S.TERMS)
ALPHA = 1.0
YEARS = ["2023", "2024", "2025"]
POLL = ["vc", "benzene", "total_voc"]

plan_sha = hashlib.sha256(open(os.path.join(HERE, "NOV2026_PLAN.md"), "rb").read()).hexdigest()
assert plan_sha == open(os.path.join(HERE, "NOV2026_PLAN_HASH.txt")).read().split()[0], "plan changed since frozen"
assert hashlib.sha256(open(os.path.join(HERE, "score_refit.py"), "rb").read()).hexdigest().startswith("6d7a5458"), \
    "score_refit.py changed"


def primaries(d, col, B=10000):
    out = {y: S.joint_date_perm(d.assign(forecast=d[col]), y, B) for y in ["vc", "benzene"]}
    _, padj, _, _ = multipletests([out["vc"]["p"], out["benzene"]["p"]], method="holm")
    for y, pa in zip(["vc", "benzene"], padj):
        out[y]["p_holm"] = round(float(pa), 4)
    ok = [out[y]["rho"] > 0 and out[y]["p_holm"] < 0.05 for y in ["vc", "benzene"]]
    out["decision"] = "SUPPORTED" if all(ok) else ("PARTIALLY SUPPORTED" if any(ok) else "NOT SUPPORTED")
    return out


def describe3(d, col, B=3000):
    """describe() for vc and benzene plus total VOC."""
    out = S.describe(d, col, B)
    t = d.dropna(subset=[col]).assign(forecast=lambda x: x[col])
    out["total_voc"] = S.joint_date_perm(t, "total_voc", B)
    out["total_voc"]["per_monitor"] = {s: round(float(spearmanr(g.forecast, g.total_voc)[0]), 3)
                                       for s, g in t[t.site.isin(RES)].dropna(subset=["total_voc"]).groupby("site")}
    return out


def rank_combo(d, a, b, wa=1.0, wb=1.0):
    r = d.groupby("site")[[a, b]].rank(pct=True)
    return (wa * r[a] + wb * r[b]) / (wa + wb)


def features(t0):
    cd = pd.read_csv(os.path.join(S.INP, "era5_cell_daily.csv"))
    cd = cd[(cd.date >= "2023-01-01") & (cd.date <= "2025-12-31")]
    rows = []
    for s in RES:
        lat, lon = S.SITES[s]
        g = S.with_source_terms(cd.copy(), lat, lon)
        g["temp_c"] = g.temperature - t0
        g["temp_c2"] = g.temp_c ** 2
        rows.append(g.assign(site=s)[["date", "site"] + [t for t in TERMS]])
    return pd.concat(rows)


def fit(train, y, cols):
    tr = train.dropna(subset=[y] + cols).copy()
    tr["target"] = tr.groupby("site")[y].rank(pct=True)          # within-monitor percentile rank, ties averaged
    sc = StandardScaler().fit(tr[cols])
    m = Ridge(alpha=ALPHA).fit(sc.transform(tr[cols]), tr.target)
    return sc, m, len(tr)


def predict(sc, m, d, cols):
    return m.predict(sc.transform(d[cols]))


def main():
    fits, t0, _, _ = S.load_refit()
    data = pd.read_csv(os.path.join(HERE, "refit_calvert_monitor_days.csv"))
    res = data[data.site.isin(RES)].copy()
    res["cal_year"] = res.date.str[:4]
    X = features(t0)
    res = res.drop(columns=[c for c in TERMS if c in res.columns]).merge(X, on=["date", "site"], how="left", validate="one_to_one")
    if res[TERMS].isna().any().any():
        raise ValueError("a residential monitor-day lacks an ERA5 input")
    res["hd"] = res[HD]

    # scored set: non-poster dates on which both monitors have valid vc and benzene (115 dates, 230 monitor-days)
    new = res[~res.poster_window].copy()
    ok = new.dropna(subset=["vc", "benzene"]).groupby("date").site.nunique()
    scored_dates = set(ok[ok == 2].index)
    scored = new[new.date.isin(scored_dates)].copy()
    assert scored.date.nunique() == 115 and len(scored) == 230, (scored.date.nunique(), len(scored))

    out = {"plan_sha256": plan_sha, "seed": SEED, "scored_dates": 115, "scored_monitor_days": 230}

    # ------------------------------------------------------------------ A
    print("A", flush=True)
    scored["combo"] = rank_combo(scored, "f_corrected", "hd")
    scored["combo_refit"] = rank_combo(scored, "f_refit", "hd")
    scored["combo_2to1"] = rank_combo(scored, "f_corrected", "hd", 2, 1)
    scored["combo_1to2"] = rank_combo(scored, "f_corrected", "hd", 1, 2)
    A = {"primary": primaries(scored, "combo")}
    A["paired"] = {y: {"combo_vs_ori": S.paired(scored, y, "combo", "f_corrected"),
                       "combo_vs_hours_downwind": S.paired(scored, y, "combo", "hd")} for y in ["vc", "benzene"]}
    A["components_same_days"] = {"ori_corrected": describe3(scored, "f_corrected"), "hours_downwind": describe3(scored, "hd")}
    A["secondary"] = {"combo_described": describe3(scored, "combo"), "combo_with_refit": describe3(scored, "combo_refit"),
                      "weights_2_ori_to_1_downwind": describe3(scored, "combo_2to1"),
                      "weights_1_ori_to_2_downwind": describe3(scored, "combo_1to2")}
    out["A_combo"] = A

    # ------------------------------------------------------------------ B
    print("B", flush=True)
    cols12 = TERMS + ["hd"]
    for y in POLL:
        for tag, cols in (("", TERMS), ("_12", cols12)):
            res[f"cf_{y}{tag}"] = np.nan
            for yr in YEARS:                                   # leave one calendar year out
                sc, m, _ = fit(res[res.cal_year != yr], y, cols)
                te = res.cal_year == yr
                res.loc[te, f"cf_{y}{tag}"] = predict(sc, m, res[te], cols)
            res[f"fwd_{y}"] = np.nan                           # forward only
            for yr, prior in (("2024", ["2023"]), ("2025", ["2023", "2024"])):
                sc, m, _ = fit(res[res.cal_year.isin(prior)], y, TERMS)
                te = res.cal_year == yr
                res.loc[te, f"fwd_{y}"] = predict(sc, m, res[te], TERMS)
    keep = ["date", "site"] + [c for c in res.columns if c.startswith(("cf_", "fwd_"))]
    scored = scored.merge(res[keep], on=["date", "site"], how="left", validate="one_to_one")

    Bres = {"primary": {y: S.joint_date_perm(scored.assign(forecast=scored[f"cf_{y}"]), y, 10000) for y in ["vc", "benzene"]}}
    _, padj, _, _ = multipletests([Bres["primary"]["vc"]["p"], Bres["primary"]["benzene"]["p"]], method="holm")
    for y, pa in zip(["vc", "benzene"], padj):
        Bres["primary"][y]["p_holm"] = round(float(pa), 4)       # Holm over B1 (vc model) and B2 (benzene model)
    okb = [Bres["primary"][y]["rho"] > 0 and Bres["primary"][y]["p_holm"] < 0.05 for y in ["vc", "benzene"]]
    Bres["primary"]["decision"] = "SUPPORTED" if all(okb) else ("PARTIALLY SUPPORTED" if any(okb) else "NOT SUPPORTED")
    Bres["paired"] = {y: {"vs_ori_corrected": S.paired(scored, y, f"cf_{y}", "f_corrected"),
                          "vs_refit_same_era5_inputs": S.paired(scored, y, f"cf_{y}", "f_refit"),
                          "vs_hours_downwind": S.paired(scored, y, f"cf_{y}", "hd")} for y in ["vc", "benzene"]}
    Bres["secondary"] = {
        "total_voc_model": S.joint_date_perm(scored.assign(forecast=scored.cf_total_voc), "total_voc", 3000),
        "with_hours_downwind_12_inputs": {y: S.describe(scored, f"cf_{y}_12", 3000)[y] for y in ["vc", "benzene"]},
        "forward_only_2024_2025": {y: {"calvert_fitted": S.describe(scored.dropna(subset=[f"fwd_{y}"]), f"fwd_{y}", 3000)[y],
                                       "ori_same_days": S.describe(scored.dropna(subset=[f"fwd_{y}"]), "f_corrected", 3000)[y]}
                                   for y in ["vc", "benzene"]},
        "within_test_year": {yr: {y: S.describe(scored[scored.date.str[:4] == yr], f"cf_{y}", 2000)[y]
                                  for y in ["vc", "benzene"]} for yr in YEARS}}
    out["B_calvert_fitted"] = Bres

    # final models for part C (all 2023-2025 residential monitor-days)
    final = {}
    for y in POLL:
        for tag, cols in (("", TERMS), ("_12", cols12)):
            sc, m, n = fit(res, y, cols)
            final[f"{y}{tag}"] = {"features": cols, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
                                  "coef": m.coef_.tolist(), "intercept": float(m.intercept_), "alpha": ALPHA,
                                  "n_train": n, "temperature_centre_F": t0,
                                  "outcome": "within-monitor percentile rank", "hours_downwind_column": HD}
    json.dump(final, open(os.path.join(HERE, "calvert_fitted_models.json"), "w"), indent=1)
    out["B_final_model_standardized_coefficients"] = {k: dict(zip(v["features"], np.round(v["coef"], 4).tolist()))
                                                      for k, v in final.items() if not k.endswith("_12")}

    scored.to_csv(os.path.join(HERE, "nov2026_ab_monitor_days.csv"), index=False)
    json.dump(out, open(os.path.join(HERE, "nov2026_ab_results.json"), "w"), indent=1, default=float)
    write_report(out)
    print("done", flush=True)


def fmt(r):
    return f"{r['rho']:.3f} (p {r['p']:.4f})"


def write_report(o):
    A, Bres = o["A_combo"], o["B_calvert_fitted"]
    L = ["# Parts A and B of NOV2026_PLAN.md (ORI + hours downwind; weather model fitted at Calvert)", "",
         f"Plan SHA-256 {o['plan_sha256'][:12]}; seed {o['seed']}; 115 dates, 230 residential monitor-days "
         "(the item 26/30 sample). Mean within-monitor Spearman rho; p from joint date permutation.", "",
         "## A. ORI combined with hours downwind (equal-weight rank average)", "",
         "| endpoint | combo rho | p | Holm p |", "|---|---|---|---|"]
    for y in ["vc", "benzene"]:
        r = A["primary"][y]
        L.append(f"| {y} | {r['rho']:.3f} | {r['p']:.4f} | {r['p_holm']:.4f} |")
    L += ["", f"Decision: **{A['primary']['decision']}**", "",
          "| pollutant | comparison | combo | comparator | difference | 95% CI | p (swap) |", "|---|---|---|---|---|---|---|"]
    for y in ["vc", "benzene"]:
        for k, lab in (("combo_vs_ori", "ORI alone"), ("combo_vs_hours_downwind", "hours downwind alone")):
            p = A["paired"][y][k]
            L.append(f"| {y} | {lab} | {p['refit_rho']:.3f} | {p['comparator_rho']:.3f} | {p['diff']:+.3f} | "
                     f"{p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f} | {p['p_swap']:.4f} |")
    L += ["", "Secondary (descriptive):", "", "| score | vc | benzene | total VOC |", "|---|---|---|---|"]
    rows = [("ORI corrected alone", A["components_same_days"]["ori_corrected"]),
            ("hours downwind alone", A["components_same_days"]["hours_downwind"]),
            ("combo (primary)", A["secondary"]["combo_described"]),
            ("combo with refit", A["secondary"]["combo_with_refit"]),
            ("combo 2:1 ORI:downwind", A["secondary"]["weights_2_ori_to_1_downwind"]),
            ("combo 1:2 ORI:downwind", A["secondary"]["weights_1_ori_to_2_downwind"])]
    for lab, r in rows:
        L.append(f"| {lab} | {fmt(r['vc'])} | {fmt(r['benzene'])} | {fmt(r['total_voc'])} |")
    L += ["", "## B. Weather model fitted at Calvert (ridge on the refit's 11 ERA5 inputs, leave-one-year-out)", "",
          "| endpoint | rho | p | Holm p |", "|---|---|---|---|"]
    for y in ["vc", "benzene"]:
        r = Bres["primary"][y]
        L.append(f"| {y} | {r['rho']:.3f} | {r['p']:.4f} | {r['p_holm']:.4f} |")
    L += ["", f"Decision: **{Bres['primary']['decision']}**", "",
          "| pollutant | comparator | Calvert-fitted | comparator | difference | 95% CI | p (swap) |", "|---|---|---|---|---|---|---|"]
    for y in ["vc", "benzene"]:
        for k, lab in (("vs_ori_corrected", "ORI (deployed, corrected)"), ("vs_refit_same_era5_inputs", "refit, same ERA5 inputs"),
                       ("vs_hours_downwind", "hours downwind")):
            p = Bres["paired"][y][k]
            L.append(f"| {y} | {lab} | {p['refit_rho']:.3f} | {p['comparator_rho']:.3f} | {p['diff']:+.3f} | "
                     f"{p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f} | {p['p_swap']:.4f} |")
    s = Bres["secondary"]
    L += ["", "Secondaries:", "",
          f"- Total VOC model: {fmt(s['total_voc_model'])}",
          f"- 12 inputs (adds hours downwind): vc {fmt(s['with_hours_downwind_12_inputs']['vc'])}, "
          f"benzene {fmt(s['with_hours_downwind_12_inputs']['benzene'])}"]
    for y in ["vc", "benzene"]:
        f = s["forward_only_2024_2025"][y]
        L.append(f"- Forward only (2024-2025 dates), {y}: Calvert-fitted {fmt(f['calvert_fitted'])} vs ORI {fmt(f['ori_same_days'])}")
    for yr, r in s["within_test_year"].items():
        L.append(f"- Test year {yr}: vc {fmt(r['vc'])}, benzene {fmt(r['benzene'])}")
    L += ["", "Final-model standardized coefficients (trained on all 2023-2025 residential monitor-days):", ""]
    for k, c in o["B_final_model_standardized_coefficients"].items():
        L.append(f"- {k}: " + ", ".join(f"{t} {v:+.3f}" for t, v in c.items()))
    open(os.path.join(HERE, "NOV2026_AB_RESULTS.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
