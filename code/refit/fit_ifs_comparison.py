"""Pre-registered training-city comparison: IFS-trained versus ERA5-trained refit (refit/IFS_COMPARISON_PLAN.md,
DEVIATIONS item 28). Same specification, folds and seed as the main refit (training/fit_refit.py); only the weather
source differs. Reads training_panel_v2/zip_day_panel.parquet and ifs_variant/zip_day_panel_ifs.parquet and writes
training_panel_v2/fit_ifs/. Scores nothing at Calvert.

Run from the repo root:  .venv/bin/python -m training.fit_ifs_comparison
"""
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from training import config
from training import fit_refit as F

IFS_PANEL = os.path.join(config.OUT_DIR, "ifs_variant", "zip_day_panel_ifs.parquet")
IFS_LOOKUP = os.path.join(config.OUT_DIR, "ifs_variant", "ifs_cell_lookup.csv")
MAIN_FIT = os.path.join(config.OUT_DIR, "fit")
OUT = os.path.join(config.OUT_DIR, "fit_ifs")
PLAN = os.path.expanduser("~/school/R!Louisville/Re-analysis/refit/IFS_COMPARISON_PLAN.md")
PLAN_HASH = os.path.expanduser("~/school/R!Louisville/Re-analysis/refit/IFS_COMPARISON_PLAN_HASH.txt")
CITY = {"pgh": "Pittsburgh", "lou": "Louisville"}


def check_plan_frozen():
    want = open(PLAN_HASH).read().split()[0]
    got = F._sha256(PLAN)
    if want != got:
        raise F.FitError("IFS_COMPARISON_PLAN.md changed since it was frozen; log a deviation first")
    return want


def align(P, Q, values):
    """Map arrays over Q's rows onto P's rows by (zipcode, date); rows Q lacks get NaN. Returns (arrays, mask)."""
    kp = (P.p.zipcode + "|" + P.p.date).to_numpy()
    kq = pd.Series(np.arange(len(Q.y)), index=(Q.p.zipcode + "|" + Q.p.date).to_numpy())
    qi = kq.reindex(kp).to_numpy()
    mask = ~np.isnan(qi)
    qi_m = qi[mask].astype(int)
    if not (P.y[mask] == Q.y[qi_m]).all():
        raise F.FitError("the IFS and ERA5 panels disagree on an event; the outcome must be identical")
    out = []
    for v in values:
        a = np.full(len(P.y), np.nan)
        a[mask] = v[qi_m]
        out.append(a)
    return out, mask


def reading(ci):
    return "better" if ci[0] > 0 else ("worse" if ci[1] < 0 else "no detectable difference")


def forward_rule_b(P):
    names = np.array(P.D.names)
    rows = []
    for Y in F.FORWARD_YEARS:
        tr, te = (P.p.year < Y).to_numpy(), (P.p.year == Y).to_numpy()
        f, k = F.fit_rows(P, tr)
        beta = dict(zip(names[k], f["beta"]))
        icpt = (beta[f"cy:pgh:{Y - 1}"] + beta[f"cy:lou:{Y - 1}"]) / 2
        for c in ("pgh", "lou"):
            tc = te & (P.p.city == c).to_numpy()
            _, lp, _ = F.predict_rows(P, f, k, tr, tc, intercept=icpt)
            rows.append({"test_year": Y, "city": c, "zip_strat_c": F.StratifiedC(P.y[tc], lp, P.strata[tc]).value()})
    return rows


def run(log=print):
    t_start = time.time()
    plan_sha = check_plan_frozen()
    main_manifest = json.load(open(os.path.join(MAIN_FIT, "run_manifest.json")))
    t0 = main_manifest["temperature_centre_F"]
    os.makedirs(OUT, exist_ok=True)
    d_era5, d_ifs = F.load_panel(), F.load_panel(IFS_PANEL)
    P, Q = F.prepare(d_era5, F.MAIN, t0), F.prepare(d_ifs, F.MAIN, t0)

    log("cross-validating the ERA5 main fit and the IFS fit (same folds and seed)")
    _, lp_e, lw_e = F.crossval(P, F.month_folds(P.p.year_month.to_numpy()))
    _, lp_i, lw_i = F.crossval(Q, F.month_folds(Q.p.year_month.to_numpy()))
    (p_i, w_i), common = align(P, Q, [expit(lp_i), lw_i])
    p_e = expit(lp_e)
    left_out = int(len(P.y) - common.sum())

    log("paired comparison")
    draws = F.month_draws(d_era5.year_month[~d_era5.sparse_reporting_period])
    comp = F.paired_bootstrap(P, p_i, p_e, draws, rows=common, rank_a=w_i, rank_b=lw_e)
    for c in comp:
        comp[c]["reading"] = reading(comp[c]["ci_zip_strat_c"])
    readings = [comp[c]["reading"] for c in comp]
    helps = ("better" in readings) and ("worse" not in readings)

    metrics = {"era5": F.metrics_by_city(P, p_e, rows=common, rank=lw_e),
               "ifs": F.metrics_by_city(P, p_i, rows=common, rank=w_i)}

    log("full IFS fit and forward by year")
    fq, kq, Vq, info = F.full_fit(Q)
    coefs = F.coefficient_table(Q, fq, kq, Vq).assign(fit="ifs_main")
    main = pd.read_csv(os.path.join(MAIN_FIT, "coefficients.csv"))
    main = main[main.fit == "main"].set_index("term")
    terms = [t for t in F.WEATHER + F.SOURCE]
    side = pd.DataFrame({"term": terms,
                         "era5_coef": [main.coef[t] for t in terms],
                         "ifs_coef": [coefs.set_index("term").coef[t] for t in terms],
                         "era5_per_sd_main_scale": [main.coef[t] * main.sd[t] for t in terms],
                         "ifs_per_sd_main_scale": [coefs.set_index("term").coef[t] * main.sd[t] for t in terms],
                         "ifs_ci_low": [coefs.set_index("term").ci_low[t] for t in terms],
                         "ifs_ci_high": [coefs.set_index("term").ci_high[t] for t in terms]})
    lk = pd.read_csv(IFS_LOOKUP, dtype={"zipcode": str})
    series = {"ifs_cells": lk.groupby("city").ifs_cell.nunique().to_dict(),
              "era5_cells": d_era5.groupby("city").cell_id.nunique().to_dict()}
    results = {"plan_sha256": plan_sha, "rows_main": int(len(P.y)), "rows_compared": int(common.sum()),
               "zip_days_left_out": left_out, "left_out_share": left_out / len(P.y),
               "comparison_ifs_minus_era5": comp, "ifs_helps_overall": helps, "metrics_on_compared_rows": metrics,
               "weather_series": series, "forward_rule_b_ifs": forward_rule_b(Q),
               "twoway_negative_eigenvalues": info.get("twoway_negative_eigenvalues")}
    coefs.to_csv(os.path.join(OUT, "coefficients.csv"), index=False)
    side.to_csv(os.path.join(OUT, "coefficients_side_by_side.csv"), index=False)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=F._json_default)
    manifest = {"run_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "plan_sha256": plan_sha,
                "script_sha256": F._sha256(os.path.abspath(__file__)), "fit_refit_sha256": F._sha256(F.__file__),
                "ifs_panel_sha256": F._sha256(IFS_PANEL), "era5_panel_sha256": F._sha256(F.PANEL), "seed": F.SEED,
                "minutes": round((time.time() - t_start) / 60, 1)}
    json.dump(manifest, open(os.path.join(OUT, "run_manifest.json"), "w"), indent=1)
    write_report(results, side, manifest)
    log(f"done in {manifest['minutes']} min")
    return results


def write_report(r, side, m):
    L = ["# IFS-trained versus ERA5-trained refit: training-city comparison", "",
         f"Pre-registered in `refit/IFS_COMPARISON_PLAN.md` (SHA-256 `{m['plan_sha256'][:12]}`; DEVIATIONS item 28). "
         f"Run {m['run_utc']}. Same specification, folds and seed as the main refit; only the weather source differs. "
         "Nothing here is scored at Calvert.", "",
         f"Weather series: IFS cells {r['weather_series']['ifs_cells']} against ERA5 cells "
         f"{r['weather_series']['era5_cells']}. Rows compared: {r['rows_compared']:,} of {r['rows_main']:,} "
         f"({r['zip_days_left_out']:,} zip-days left out for incomplete IFS hours, {r['left_out_share']:.2%}).", "",
         "## Primary: IFS minus ERA5, out of fold, identical rows", "",
         "| city | IFS c | ERA5 c | difference in within-zip c (95% CI) | difference in log-loss (95% CI) | reading |",
         "|---|---|---|---|---|---|"]
    for c, x in r["comparison_ifs_minus_era5"].items():
        L.append(f"| {CITY[c]} | {r['metrics_on_compared_rows']['ifs'][c]['zip_strat_c']:.3f} | "
                 f"{r['metrics_on_compared_rows']['era5'][c]['zip_strat_c']:.3f} | {x['diff_zip_strat_c']:+.4f} "
                 f"({x['ci_zip_strat_c'][0]:+.4f} to {x['ci_zip_strat_c'][1]:+.4f}) | {x['diff_logloss']:+.4f} "
                 f"({x['ci_logloss'][0]:+.4f} to {x['ci_logloss'][1]:+.4f}) | **{x['reading']}** |")
    L += ["", f"Pre-registered overall reading: IFS training **{'helps' if r['ifs_helps_overall'] else 'does not help'}** "
              "(better in at least one city and not worse in the other).", "",
          "## Secondary: coefficients per SD (main fit's SD), forward by year", "",
          "| term | ERA5 | IFS | IFS 95% CI (raw units) |", "|---|---|---|---|"]
    for row in side.itertuples():
        L.append(f"| {row.term} | {row.era5_per_sd_main_scale:+.3f} | {row.ifs_per_sd_main_scale:+.3f} | "
                 f"{row.ifs_ci_low:.3g} to {row.ifs_ci_high:.3g} |")
    L += ["", "Forward by year, rule B (IFS model): " + ", ".join(
        f"{x['test_year']} {CITY[x['city']]} {x['zip_strat_c']:.3f}" for x in r["forward_rule_b_ifs"]) + ".", ""]
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
