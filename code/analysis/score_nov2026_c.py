"""NOV2026_PLAN.md part C (frozen 2026-11-02, SHA-256 0cca00e6...; DEVIATIONS 33, 35): older Calvert-area monitors.

Inputs (built blind to measurements by older_monitors/ scripts, see older_monitors/BUILD_LOG.md):
  older_monitors/older_monitor_days.csv   measurements (24-hour samples before 2023)
  older_monitors/older_scores.csv         every score for every (date, monitor) on the union of sample dates
  older_monitors/monitor_inventory.csv    monitors; IN_PLANT below is the pre-scoring classification (DEVIATIONS 35)
  calvert_fitted_models.json              part B final models (trained on 2023-2025, applied unchanged)

Statistics follow score_refit.py (mean within-monitor Spearman; joint date permutation; date-bootstrap CI and
within-date swap for paired differences), generalized to any number of monitors with different sample dates:
one permutation of the union of sample dates is applied to every monitor's forecast series.
Seed 20261102. Outputs: older_monitors/c_results.json, NOV2026_C_RESULTS.md, older_monitors/c_monitor_days.csv.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
OM = os.path.join(HERE, "older_monitors")
SEED = 20261102
rng = np.random.default_rng(SEED)
B_PRIMARY, B_SEC = 10000, 5000

plan_sha = hashlib.sha256(open(os.path.join(HERE, "NOV2026_PLAN.md"), "rb").read()).hexdigest()
assert plan_sha == open(os.path.join(HERE, "NOV2026_PLAN_HASH.txt")).read().split()[0], "plan changed since frozen"

# Pre-scoring classification (DEVIATIONS 35, made from monitor_inventory.csv metadata before any score was computed):
# LWD (inside the complex, as in ANALYSIS_PLAN) and TVA Substation (same Plant Cutoff Road as LWD, EPA land use
# industrial, source-oriented middle-scale objective, 0.39 km from the Lubrizol and Westlake Vinyls points).
IN_PLANT = {"21-157-0021", "21-157-0014"}
TVA = "21-157-0014"
MIN_SPECIES_TOTAL_VOC = 50     # before mid-2012 the VOC files carry only 13-14 species

SCORES = {"deployed_on_era5": "ORI (deployed model, ERA5 inputs)", "refit": "ORI refit (ERA5)",
          "hours_downwind": "hours downwind of the complex", "combo": "ORI + hours downwind (part A)",
          "calvert_fitted": "weather model fitted at Calvert (part B)"}


# ------------------------------------------------------------------ statistics for N monitors

def _prep(d, col, y):
    """Per monitor: union-date index of each sample, the forecast on every union date, and the outcome."""
    d = d.dropna(subset=[y])
    U = np.sort(d.date.unique())
    pos = {dt: i for i, dt in enumerate(U)}
    per = []
    for s, g in d.groupby("site"):
        if g[y].nunique() < 2:
            continue
        f_all = SC.loc[SC.site == s].set_index("date")[col].reindex(U).to_numpy(float)
        if np.isnan(f_all).any():
            raise ValueError(f"{s}: score {col} missing on a union date")
        per.append((s, g.date.map(pos).to_numpy(), f_all, g[y].to_numpy(float)))
    return U, per


def mean_rho(per, perm=None):
    vals = []
    for _, idx, f_all, yy in per:
        f = f_all[perm[idx]] if perm is not None else f_all[idx]
        vals.append(spearmanr(f, yy)[0])
    return float(np.mean(vals)), [float(v) for v in vals]


def joint_perm(d, col, y, B=B_PRIMARY):
    U, per = _prep(d, col, y)
    if not per:
        return {"rho": None, "note": "no monitor with variation"}
    obs, each = mean_rho(per)
    null = np.array([mean_rho(per, rng.permutation(len(U)))[0] for _ in range(B)])
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1)
    return {"rho": round(obs, 3), "p": round(float(p), 4), "monitors": len(per),
            "per_monitor": {s: round(v, 3) for (s, *_), v in zip(per, each)},
            "monitor_days": int(sum(len(x[1]) for x in per)), "dates": int(len(U))}


def paired(d, y, a, b, B=B_SEC):
    """a minus b, mean within-monitor rho on identical monitor-days. CI: bootstrap of union dates (joint across
    monitors). p: random swaps of the two scores within dates (same swap at every monitor), on within-monitor
    percentile ranks."""
    d = d.dropna(subset=[y, a, b])
    U = np.sort(d.date.unique()); pos = {dt: i for i, dt in enumerate(U)}
    per = []
    for s, g in d.groupby("site"):
        if g[y].nunique() < 2:
            continue
        per.append((g.date.map(pos).to_numpy(), rankdata(g[a]) / len(g), rankdata(g[b]) / len(g), g[y].to_numpy(float)))

    def delta(w=None, swap=None):
        da, db = [], []
        for idx, ra, rb, yy in per:
            A_, B_ = ra, rb
            if swap is not None:
                sw = swap[idx]; A_, B_ = np.where(sw, rb, ra), np.where(sw, ra, rb)
            if w is not None:
                rep = w[idx]
                A_, B_, Y_ = np.repeat(A_, rep), np.repeat(B_, rep), np.repeat(yy, rep)
            else:
                Y_ = yy
            if len(Y_) < 3 or np.unique(Y_).size < 2:
                return np.nan, np.nan, np.nan
            da.append(spearmanr(A_, Y_)[0]); db.append(spearmanr(B_, Y_)[0])
        return float(np.mean(da) - np.mean(db)), float(np.mean(da)), float(np.mean(db))

    obs, ra, rb = delta()
    n = len(U)
    boots = np.array([delta(w=np.bincount(rng.integers(0, n, n), minlength=n))[0] for _ in range(B)])
    swaps = np.array([delta(swap=rng.random(n) < 0.5)[0] for _ in range(B)])
    p = (np.sum(np.abs(swaps) >= abs(obs)) + 1) / (B + 1)
    return {"a_rho": round(ra, 3), "b_rho": round(rb, 3), "diff": round(obs, 3),
            "ci95": [round(float(np.nanpercentile(boots, 2.5)), 3), round(float(np.nanpercentile(boots, 97.5)), 3)],
            "p_swap": round(float(p), 4), "dates": int(n)}


def site_perm(g, col, y, B=B_SEC):
    g = g.dropna(subset=[y, col])
    if g[y].nunique() < 2:
        return {"rho": None, "n": int(len(g))}
    obs = spearmanr(g[col], g[y])[0]
    xr = rankdata(g[col].values); yr = rankdata(g[y].values)
    null = np.array([spearmanr(rng.permutation(xr), yr)[0] for _ in range(B)])
    return {"rho": round(float(obs), 3), "p": round(float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1)), 4),
            "n": int(len(g))}


# ------------------------------------------------------------------ scores

def apply_calvert_model(df, spec):
    X = df[spec["features"]].to_numpy(float)
    z = (X - np.array(spec["scaler_mean"])) / np.array(spec["scaler_scale"])
    return z @ np.array(spec["coef"]) + spec["intercept"]


def main():
    global SC
    inv = pd.read_csv(os.path.join(OM, "monitor_inventory.csv"), dtype={"site_id": str})
    meas = pd.read_csv(os.path.join(OM, "older_monitor_days.csv"), dtype={"site_id": str})
    SC = pd.read_csv(os.path.join(OM, "older_scores.csv"), dtype={"site_id": str})
    for t in (meas, SC):
        t["site"] = t.site_id
    meas.loc[meas.n_species < MIN_SPECIES_TOTAL_VOC, "total_voc"] = np.nan
    SC = SC.drop(columns=["name", "sampled"])
    models = json.load(open(os.path.join(HERE, "calvert_fitted_models.json")))
    hd = [c for c in SC.columns if c.lower().replace(" ", "_").startswith("hours_downwind")][0]
    SC["hours_downwind"] = SC[hd]
    SC["deployed_on_era5"] = SC.f_deployed_on_era5
    SC["refit"] = SC.f_refit
    if "hours downwind of complex (wind-direction analysis)" not in SC.columns:
        SC["hours downwind of complex (wind-direction analysis)"] = SC.hours_downwind
    SC["hd"] = SC.hours_downwind
    for y in ["vc", "benzene", "total_voc"]:
        SC[f"calvert_fitted_{y}"] = apply_calvert_model(SC, models[y])
    # combo: within-monitor percentile ranks over each monitor's sampled days (part A rule), computed per outcome below

    community = [s for s in inv.site_id if s not in IN_PLANT and s in set(meas.site_id)]
    data = meas.merge(SC.drop(columns=["site_id"]), on=["date", "site"], how="left", validate="one_to_one")
    if data[["deployed_on_era5", "refit", "hours_downwind"]].isna().any().any():
        raise ValueError("a sampled monitor-day has no score")
    yr = data.date.str[:4].astype(int)
    data["block"] = (yr // 5 * 5).astype(str)
    lab = data.groupby("block").date.agg(lambda x: f"{x.min()[:4]}-{x.max()[:4]}")
    data["block"] = data.block.map(lab)

    # the combo must exist on every union date for the joint permutation: rank against each monitor's sampled days
    def combo_scores(d):
        out = []
        for s, g in SC.groupby("site"):
            ref = d[d.site == s]
            if ref.empty:
                continue
            def avg_pct(refv, v):
                # average-rank percentile (pandas rank(pct=True) on sampled days; part A's tie rule; DEVIATIONS 38b)
                srt = np.sort(refv)
                lo = np.searchsorted(srt, v, side="left"); hi = np.searchsorted(srt, v, side="right")
                return (lo + (hi - lo + 1) / 2) / len(srt)
            a = avg_pct(ref.deployed_on_era5.values, g.deployed_on_era5.values)
            b = avg_pct(ref.hours_downwind.values, g.hours_downwind.values)
            out.append(pd.DataFrame({"date": g.date.values, "site": s, "combo": (a + b) / 2}))
        return pd.concat(out)

    comm = data[data.site.isin(community)].copy()
    cs = combo_scores(comm)
    SC = SC.drop(columns=["combo"], errors="ignore").merge(cs, on=["date", "site"], how="left")
    comm = comm.merge(cs, on=["date", "site"], how="left")
    out = {"plan_sha256": plan_sha, "seed": SEED, "in_plant_set_aside": sorted(IN_PLANT), "community_monitors": community,
           "counts": comm.groupby("site").size().to_dict(), "date_range": [comm.date.min(), comm.date.max()]}

    def score_col(name, y):
        return f"calvert_fitted_{y}" if name == "calvert_fitted" else name

    # ---- primaries C1, C2 (deployed ORI on ERA5)
    print("primaries", flush=True)
    prim = {y: joint_perm(comm, "deployed_on_era5", y) for y in ["vc", "benzene"]}
    _, padj, _, _ = multipletests([prim["vc"]["p"], prim["benzene"]["p"]], method="holm")
    for y, pa in zip(["vc", "benzene"], padj):
        prim[y]["p_holm"] = round(float(pa), 4)
    ok = [prim[y]["rho"] > 0 and prim[y]["p_holm"] < 0.05 for y in ["vc", "benzene"]]
    out["primary"] = prim
    out["decision"] = "SUPPORTED" if all(ok) else ("PARTIALLY SUPPORTED" if any(ok) else "NOT SUPPORTED")
    print(json.dumps(prim), out["decision"], flush=True)

    # ---- every score, every pollutant
    print("all scores", flush=True)
    out["all_scores"] = {k: {y: joint_perm(comm, score_col(k, y), y, B_SEC) for y in ["vc", "benzene", "total_voc"]
                             if comm[y].notna().any()} for k in SCORES}
    # ---- each monitor alone
    out["per_monitor"] = {s: {k: {y: site_perm(g, score_col(k, y), y) for y in ["vc", "benzene"]} for k in SCORES}
                          for s, g in comm.groupby("site")}
    # ---- 5-year blocks (primary score)
    out["by_block"] = {blk: {y: joint_perm(g, "deployed_on_era5", y, 3000) for y in ["vc", "benzene"]}
                       for blk, g in comm.groupby("block")}
    # ---- paired comparisons among the five scores
    print("paired", flush=True)
    keys = list(SCORES)
    out["paired"] = {}
    for y in ["vc", "benzene"]:
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = score_col(keys[i], y), score_col(keys[j], y)
                out["paired"][f"{y}: {keys[i]} minus {keys[j]}"] = paired(comm, y, a, b)
    # ---- vinyl chloride detection by forecast half (CMH over monitors)
    d = comm.dropna(subset=["vc"]).copy()
    d["hi_fc"] = d.deployed_on_era5 > d.groupby("site").deployed_on_era5.transform("median"); d["det"] = d.vc > 0
    tabs = [np.array([[(g.hi_fc & g.det).sum(), (g.hi_fc & ~g.det).sum()],
                      [(~g.hi_fc & g.det).sum(), (~g.hi_fc & ~g.det).sum()]]) + 0.5 for _, g in d.groupby("site")]
    out["vc_detection"] = {s: {"hi": f"{int((g.hi_fc & g.det).sum())}/{int(g.hi_fc.sum())}",
                               "lo": f"{int((~g.hi_fc & g.det).sum())}/{int((~g.hi_fc).sum())}"} for s, g in d.groupby("site")}
    if len(tabs) > 0:
        st = StratifiedTable(tabs)
        out["vc_detection"]["cmh_or"] = round(float(st.oddsratio_pooled), 2)
        pv = float(st.test_null_odds().pvalue)
        out["vc_detection"]["cmh_p"] = "< 0.0001" if pv < 1e-4 else round(pv, 4)
    # ---- in-plant monitors, separately
    ip = data[data.site.isin(IN_PLANT)]
    out["in_plant"] = {s: {k: {y: site_perm(g.assign(combo=np.nan) if k == "combo" else g, score_col(k, y), y)
                               for y in ["vc", "benzene"]} for k in SCORES if k != "combo"}
                       for s, g in ip.groupby("site")}
    # ---- sensitivity (not in the plan, labelled): primary score with TVA Substation counted as a community monitor
    with_tva = data[data.site.isin(community + [TVA])]
    out["sensitivity_tva_included"] = {y: joint_perm(with_tva, "deployed_on_era5", y, B_SEC) for y in ["vc", "benzene"]}

    comm.to_csv(os.path.join(OM, "c_monitor_days.csv"), index=False)
    json.dump(out, open(os.path.join(OM, "c_results.json"), "w"), indent=1, default=float)
    write_report(out, inv)
    print("done", flush=True)


def write_report(o, inv):
    names = dict(zip(inv.site_id, inv.get("name", inv.site_id)))
    L = ["# Part C of NOV2026_PLAN.md: older Calvert-area monitors (external test in time)", "",
         f"Plan SHA-256 {o['plan_sha256'][:12]}; seed {o['seed']}. Community monitors: "
         + ", ".join(f"{s} {names.get(s, '')} ({o['counts'][s]} monitor-days)" for s in o["community_monitors"])
         + f"; dates {o['date_range'][0]} to {o['date_range'][1]}. Set aside as in-plant: {o['in_plant_set_aside'] or 'none'}.", "",
         "## Primary (deployed ORI on ERA5 inputs)", "", "| endpoint | mean within-monitor rho | p | Holm p | monitor-days |",
         "|---|---|---|---|---|"]
    for y in ["vc", "benzene"]:
        r = o["primary"][y]
        L.append(f"| {y} | {r['rho']:.3f} | {r['p']:.4f} | {r['p_holm']:.4f} | {r['monitor_days']} |")
    L += ["", f"Decision: **{o['decision']}**", "", "## All five scores", "", "| score | vc | benzene | total VOC |", "|---|---|---|---|"]
    for k, lab in SCORES.items():
        r = o["all_scores"][k]
        cell = lambda y: (f"{r[y]['rho']:.3f} (p {r[y]['p']:.4f})" if y in r and r[y].get("rho") is not None else "n/a")
        L.append(f"| {lab} | {cell('vc')} | {cell('benzene')} | {cell('total_voc')} |")
    L += ["", "## Each monitor (rho, p)", ""]
    for s, r in o["per_monitor"].items():
        L.append(f"- {s} {names.get(s, '')}: " + "; ".join(
            f"{k} vc {r[k]['vc']['rho']} / benzene {r[k]['benzene']['rho']}" for k in SCORES))
    L += ["", "## 5-year blocks (deployed ORI on ERA5)", ""]
    for blk, r in o["by_block"].items():
        L.append(f"- {blk}: vc {r['vc'].get('rho')} (p {r['vc'].get('p')}), benzene {r['benzene'].get('rho')} (p {r['benzene'].get('p')})")
    L += ["", "## Paired differences (a minus b)", "", "| comparison | a | b | difference | 95% CI | p (swap) |", "|---|---|---|---|---|---|"]
    for k, p in o["paired"].items():
        L.append(f"| {k} | {p['a_rho']:.3f} | {p['b_rho']:.3f} | {p['diff']:+.3f} | {p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f} | {p['p_swap']:.4f} |")
    L += ["", f"Vinyl chloride detection, above vs below each monitor's median forecast: {json.dumps(o['vc_detection'])}", "",
          f"In-plant monitors (deployed ORI on ERA5): {json.dumps(o['in_plant'])}"]
    open(os.path.join(HERE, "NOV2026_C_RESULTS.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
