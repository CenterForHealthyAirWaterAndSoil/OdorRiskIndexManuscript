"""RESULTS.md for the ORI refit (training/fit_refit.py): the numbers a reader needs, in plain tables."""
import os

import numpy as np
import pandas as pd

CITY = {"pgh": "Pittsburgh", "lou": "Louisville"}
LABEL = {"temp_c": "temperature (centred, per deg F)", "temp_c2": "temperature squared (centred)",
         "diurnal_temperature_range": "diurnal temperature range (per deg F)",
         "relative_humidity": "relative humidity (per %)", "wind_speed": "wind speed (per mph)",
         "boundary_layer_height": "BLH (per ft)", "solar_radiation": "solar radiation (per W/m2)",
         "precipitation": "rain (per inch)", "pressure_anomaly": "pressure anomaly (per hPa)",
         "multi_source_exposure": "proximity index (0-1)", "wind_align_weighted": "wind alignment (0-1)"}
TERMS = list(LABEL)
SIGN = {1: "+", -1: "-", 0: "none"}


def g(x, d=3):
    """d significant digits."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return ""
    return f"{x:.{d}g}"


def f3(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci(v):
    return f"{v[0]:+.3f} to {v[1]:+.3f}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_report(out, coefs, metrics, results, manifest):
    L = ["# ORI refit: results in the training cities", "",
         f"Run {manifest['run_utc']} ({manifest['minutes']} min). Plan `REFIT_PLAN.md` SHA-256 "
         f"`{manifest['plan_sha256'][:12]}`; script `{manifest['script_sha256'][:12]}`; panel "
         f"`{manifest['panel_sha256'][:12]}`; seed {manifest['seed']}. Nothing here is scored at Calvert, and no "
         "deployed coefficient or live-site setting was changed.", ""]

    main = coefs[coefs.fit == "main"].set_index("term")
    signs = {r["term"]: r for r in results["readings"]["expected_signs"]}
    L += ["## 1. Main fit (all 107 Pittsburgh zips, all Louisville zips after 2019-03-01)", "",
          f"{results['main']['n']:,} zip-days, {results['main']['events']:,} events (any report other than "
          f"\"just fine\"). Event thresholds (city mean burden): " +
          ", ".join(f"{CITY[k]} {v:.3f}" for k, v in results["main"]["thresholds"].items()) +
          f". Temperature centred at {manifest['temperature_centre_F']:.2f} F.", "",
          "Effect per SD is the change in log-odds for a one-standard-deviation change in the term (comparable "
          "across terms). Two-way SEs (zip and month) are the primary; date-clustered SEs are beside them.", ""]
    rows = []
    for t in TERMS:
        r = main.loc[t]
        s = signs[t]
        rows.append([LABEL[t], g(r.coef, 4), f3(r.per_sd), g(r.se_twoway, 3), g(r.se_date, 3),
                     f"{g(r.ci_low, 3)} to {g(r.ci_high, 3)}", SIGN[s["expected"]],
                     "" if s["matches"] is None else ("yes" if s["matches"] else "**no**")])
    L += table(rows, ["term", "coefficient", "effect per SD", "SE two-way", "SE date", "95% CI (two-way)",
                      "expected sign", "matches"])
    w = results["main"]["weather_block_wald"]
    L += ["", f"Weather block (nine terms) joint Wald test, two-way: chi2 = {w['chi2']:.1f} on {w['df']} df, "
              f"p = {w['p']:.2g}. Variance inflation factors: " +
          ", ".join(f"{k} {v}" for k, v in results["main"]["vif"].items()) + "."]
    if results["main"].get("twoway_negative_eigenvalues"):
        L.append(f"Two-way covariance had {results['main']['twoway_negative_eigenvalues']} negative eigenvalue(s), "
                 "clipped to zero.")
    se = coefs[coefs.fit == "main"].set_index("term")
    L += ["", "SE diagnostic (same coefficients, different clustering):", ""]
    L += table([[LABEL[t]] + [g(se.loc[t, f"se_{c}"], 3) for c in ("twoway", "date", "week", "month", "quarter",
                                                                      "zip")] for t in TERMS],
               ["term", "two-way", "date", "week", "month", "quarter", "zip"])
    dowt = [t for t in ("dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "holiday") if t in se.index]
    L += ["", "Day of week (vs Monday) and holiday: " +
          ", ".join(f"{t} {se.loc[t, 'coef']:+.3f}" for t in dowt) + ".", ""]
    cy = se[se.index.str.startswith("cy:")]
    L += ["City x year intercepts: " + ", ".join(f"{i[3:]} {v:+.2f}" for i, v in cy.coef.items()) + ".", ""]

    m = metrics
    L += ["## 2. Out-of-fold skill (calendar-month folds, day-of-week at its average effect)", "",
          "Zip-stratified concordance asks: within one zip and year, do event days get higher predictions than "
          "non-event days? (0.5 = no better than chance.) It is the training-city version of what the Calvert test "
          "scores. Log-loss: lower is better. Pooled AUC also rewards knowing which zips and years report a lot.", "",
          "**Correction (DEVIATIONS item 24, made after the first run's results were seen).** Ranking days by the "
          "full out-of-fold prediction lets each fold's re-estimated city-year intercept and proximity term enter "
          "the within-zip ranking; a held-out month with many events gets a lower intercept, which pushed models "
          "with no within-zip information to 0.43 (Pittsburgh) and 0.33 (Louisville) instead of 0.5. Concordance "
          "now ranks days by the part of the prediction that varies within a zip and year (weather, wind alignment, "
          "the seasonal curve, or the deployed model's fixed weather terms), which is what varies within a Calvert "
          "monitor. The literal pre-registered computation is in the 'c (literal)' column.", ""]
    names = [("main", "refit (main)"), ("B0", "B0: city-year + weekday"), ("B1", "B1: + proximity"),
             ("B2", "B2: + seasonal curve"), ("deployed_slopes_true_feet", "deployed weights, BLH true feet"),
             ("deployed_slopes_fitted_scale", "deployed weights, BLH at fitted scale")]
    rows = []
    for key, label in names:
        for c in ("pgh", "lou"):
            r = m[(m.fit == key) & (m.folds == "month") & (m.dow == "average") & (m.city == c)]
            if len(r):
                r = r.iloc[0]
                rows.append([label, CITY[c], f3(r.zip_strat_c), f3(r.zip_strat_c_literal), f3(r.logloss),
                             f3(r.auc), f3(r.brier), f3(r.cal_in_the_large), f3(r.cal_slope)])
    L += table(rows, ["model", "city", "zip-stratified c", "c (literal)", "log-loss", "pooled AUC", "Brier",
                      "calibration-in-the-large", "calibration slope"])
    L += ["", "Paired differences, 95% CIs from 1,000 resamples of calendar months:", ""]
    rows = []
    for key, label in (("main_minus_B2", "refit minus B2 (what weather adds beyond season)"),
                       ("main_minus_deployed_slopes", "refit minus deployed weights (BLH true feet)")):
        for c in ("pgh", "lou"):
            r = results[key][c]
            rows.append([label, CITY[c], f"{r['diff_zip_strat_c']:+.4f}", ci(r["ci_zip_strat_c"]),
                         f"{r['diff_logloss']:+.4f}", ci(r["ci_logloss"])])
    L += table(rows, ["comparison", "city", "diff in c", "95% CI", "diff in log-loss", "95% CI"])
    da = results["deployed_a"]
    L += ["", "Deployed model applied as is (in-sample with respect to dates, ranking only): zip-stratified c " +
          ", ".join(f"{CITY[c]} {da[c]['zip_strat_c']:.3f} (pooled AUC {da[c]['auc']:.3f})" for c in da) + ".", ""]
    lk = results["leakage"]
    L += ["Leakage check (zip-stratified c by fold scheme): " + "; ".join(
        f"{CITY[c]} month {lk[c]['month_c']:.3f}, week {lk[c]['week_c']:.3f}, date {lk[c]['date_c']:.3f}"
        f" (date minus month {lk[c]['date_minus_month']:+.4f}"
        f"{', reported as leakage' if lk[c]['date_minus_month'] > 0.01 else ''})" for c in ("pgh", "lou")) + ".",
          ""]
    act = m[(m.fit == "main") & (m.dow == "actual")]
    L += ["With the actual day-of-week effects (diagnostic): " + ", ".join(
        f"{CITY[r.city]} c {r.zip_strat_c:.3f}" for r in act.itertuples()) + ".", ""]

    L += ["## 3. Transfer checks", "", "Forward by year: fit on earlier years, predict the test year. Rule A uses "
          "the city's own previous-year intercept; rule B uses the two-city average (how Calvert is run).", ""]
    rows = [[r["test_year"], r["rule"][0], CITY[r["city"]], f3(r["zip_strat_c"]), f3(r["logloss"]),
             f3(r["cal_in_the_large"]), f3(r["cal_slope"])] for r in results["forward"]]
    L += table(rows, ["test year", "rule", "city", "zip-stratified c", "log-loss", "calibration-in-the-large",
                      "calibration slope"])
    L += ["", "Leave one city out (fit one city, predict the other with the fitting city's 2025 intercept):", ""]
    L += table([[CITY[r["fit_city"]], CITY[r["test_city"]], f3(r["zip_strat_c"]), f3(r["logloss"]),
                 f3(r["cal_in_the_large"]), f3(r["cal_slope"])] for r in results["leave_one_city_out"]],
               ["fitted on", "predicting", "zip-stratified c", "log-loss", "calibration-in-the-large",
                "calibration slope"])
    iw = results["interaction_wald"]
    L += ["", f"Do the cities' weather slopes differ? Joint Wald test of the ten city x term interactions: "
              f"chi2 = {iw['chi2']:.1f} on {iw['df']} df, p = {iw['p']:.2g}.", ""]
    pc = coefs[coefs.fit.isin(["city_pgh_only", "city_lou_only"])].pivot(index="term", columns="fit",
                                                                          values="coef")
    pcs = coefs[coefs.fit.isin(["city_pgh_only", "city_lou_only"])].pivot(index="term", columns="fit",
                                                                           values="se_twoway")
    L += table([[LABEL[t], f"{g(main.loc[t, 'coef'], 3)}", f"{g(pc.loc[t, 'city_pgh_only'], 3)} "
                 f"({g(pcs.loc[t, 'city_pgh_only'], 2)})", f"{g(pc.loc[t, 'city_lou_only'], 3)} "
                 f"({g(pcs.loc[t, 'city_lou_only'], 2)})"] for t in TERMS],
               ["term", "pooled", "Pittsburgh only (SE)", "Louisville only (SE)"])
    rc = results["replacement_checks"]
    L += ["", "Training-city checks for replacement (plan s6.6): " + "; ".join(
        f"{CITY[c]}: (a) beats the seasonal baseline {'yes' if rc[c]['a_beats_climatology'] else 'no'}, "
        f"(b) forward rule-B c above 0.5 every year {'yes' if rc[c]['b_forward_above_half'] else 'no'}"
        for c in ("pgh", "lou")) + f". Overall: **{'pass' if rc['both_pass'] else 'fail'}**.", ""]

    th = results["threshold"]
    L += ["## 4. Alert threshold", "",
          f"F1-optimal threshold on the out-of-fold predictions, pooled with equal city weight: "
          f"**{th['pooled_equal_city_weight']:.4f}** (F1 {th['pooled_f1']:.3f}). Per city: " +
          ", ".join(f"{CITY[c]} {v['threshold']:.4f} (F1 {v['f1']:.3f})" for c, v in th["per_city"].items()) +
          ". With the threshold chosen inside each training set (nested) and applied to the held-out months: " +
          "; ".join(f"{CITY[c]} precision {v['precision']:.3f}, recall {v['recall']:.3f}, F1 {v['f1']:.3f}, "
                    f"{v['share_flagged']:.1%} of zip-days flagged" for c, v in th["nested_operating"].items()) +
          ". This is on an all-zip-days scale and is not comparable with the deployed 0.2113 (report-days scale).",
          ""]

    L += ["## 5. Sensitivities", "", "Out-of-fold zip-stratified c and the main source and weather coefficients "
          "(per SD of the term in the main fit, so columns are comparable). Fits marked 'coef only' have no "
          "out-of-fold metrics by design.", ""]
    show = ["boundary_layer_height", "wind_speed", "diurnal_temperature_range", "precipitation",
            "pressure_anomaly", "wind_align_weighted", "multi_source_exposure"]
    sd = main.sd
    fits = ["main"] + list(results["sensitivities"])
    rows = []
    for fname in fits:
        cf = coefs[coefs.fit == fname].set_index("term")
        mm = m[(m.fit == fname) & (m.folds == "month") & (m.dow == "average")].set_index("city")
        cpg = f3(mm.loc["pgh", "zip_strat_c"]) if "pgh" in mm.index else "coef only"
        clo = f3(mm.loc["lou", "zip_strat_c"]) if "lou" in mm.index else ("coef only" if cpg == "coef only" else "")
        n = results["main"]["n"] if fname == "main" else results["sensitivities"][fname]["n"]
        cells = []
        for t in show:
            if t in cf.index and t in sd.index and not np.isnan(sd[t]):
                cells.append(f"{cf.loc[t, 'coef'] * sd[t]:+.3f}")
            else:
                cells.append("")
        rows.append([fname, f"{n:,}", cpg, clo] + cells)
    L += table(rows, ["fit", "zip-days", "Pgh c", "Lou c", "BLH", "wind speed", "DTR", "rain", "pressure anom.",
                      "alignment", "proximity"])
    s61 = results["sensitivities"].get("original_61_zips", {}).get("pgh_61_shared_zip_strat_c")
    if s61:
        L += ["", f"Over the 61 shared Pittsburgh zips: main fit c {s61['main_fit']:.3f}, original-61 fit c "
                  f"{s61['original_61_fit']:.3f}."]
    L += ["", f"Sewer sensitivity removed {results['sewer_reports_removed']:,} Louisville reports.", ""]

    rd = results["readings"]
    L += ["## 6. Readings fixed before the fit", "",
          "Zip fixed effects (a term that flips sign or leaves the main fit's 95% CI is 'reporting-sensitive'); "
          "negative control (a term that moves 'just fine' reports the same way, at least half as strongly, with a CI "
          "excluding 0 is 'reporting behaviour'); leave-one-zip-out (largest change, in main-fit SEs).", ""]
    rows = []
    for fe, nc, lo in zip(rd["zip_fixed_effects"], rd["negative_control"], rd["leave_one_zip_out"]):
        rows.append([LABEL[fe["term"]], g(fe["main"], 3), g(fe["fe"], 3),
                     "**yes**" if fe["reporting_sensitive"] else "no", g(nc["negative_control"], 3),
                     "**yes**" if nc["reporting_behaviour"] else "no",
                     f"{lo['max_change_in_se']:.2f} ({lo['fit'].replace('leave_out_', '')})"])
    L += table(rows, ["term", "main", "zip FE", "reporting-sensitive", "negative control", "reporting behaviour",
                      "largest leave-one-zip-out change, SEs (zip)"])

    br = coefs[coefs.fit == "bridging"].set_index("term")
    dep = __import__("odor_forecast_core").COEFFS_POOLED_PROXIMITY
    L += ["", "## 7. Bridging fit (deployed recipe on the new ERA5 panel) beside the deployed coefficients", "",
          "Same recipe as far as it is documented: report days, 61 Pittsburgh zips, original target, raw temperature "
          "and pressure, circular-mean wind alignment, constant plus a Louisville indicator. BLH is in true feet here; "
          "the deployed BLH coefficient was fitted on feet that were 3.28 times too large.", ""]
    bmap = {"temperature": "temperature", "temperature_squared": "temperature_squared",
            "diurnal_temperature_range": "diurnal_temperature_range", "relative_humidity": "relative_humidity",
            "wind_speed": "wind_speed", "boundary_layer_height": "boundary_layer_height",
            "solar_radiation": "solar_radiation", "precipitation": "precipitation",
            "atmospheric_pressure": "atmospheric_pressure", "multi_source_exposure": "multi_source_exposure",
            "wind_align_weighted_circmean": "wind_align_weighted", "const": "const"}
    L += table([[k, g(br.loc[k, "coef"], 4) if k in br.index else "", g(br.loc[k, "se_twoway"], 3)
                 if k in br.index else "", g(dep[v], 4)] for k, v in bmap.items()],
               ["term", "bridging fit", "SE two-way", "deployed"])
    L += ["", f"Louisville indicator in the bridging fit: {g(br.loc['lou', 'coef'], 3)}." if "lou" in br.index else ""]

    dfm = results["deployment_form"]
    L += ["", "## 8. Refit in deployment form (not deployed)", "",
          "The main fit rewritten in the live code's raw-temperature form, with the Calvert constant of plan s7 (mean "
          "of the 2025 intercepts plus the average weekday effect). Not applied anywhere.", ""]
    L += table([[k, g(v, 5)] for k, v in dfm["coefficients"].items()], ["term", "coefficient"])
    L += ["", "## Not done here", "", "- Calvert scoring: waits for your approval of this fit and a scoring script "
          "logged in DEVIATIONS first.", "- No change to the deployed coefficients or the live site.", ""]
    with open(os.path.join(out, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")
