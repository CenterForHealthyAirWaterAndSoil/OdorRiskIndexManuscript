"""Score the daily AQI against ORI at the Calvert monitors: AQI_COMPARISON_PLAN.md (frozen, SHA-256 d7b3400d...),
DEVIATIONS 30-31.

Inputs: refit_calvert_monitor_days.csv (ORI forecasts and VOCs, unchanged) and aqi_inputs/aqi_daily.csv (built by
aqi_inputs.py). Statistics are score_refit.py's functions, imported unchanged (seed 20261028).
Outputs: aqi_calvert_results.json, AQI_CALVERT_RESULTS.md, aqi_calvert_monitor_days.csv.
Run: ~/weather-varaible-analysis/.venv/bin/python score_aqi.py
"""
import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import score_refit as S

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = {"a1_aqi": "observed AQI, McCracken County (Paducah)", "m1_aqi": "modelled AQI at the monitor (CAMS)"}
SECONDARY = {"a1_pm25_aqi": "observed PM2.5 AQI, Paducah", "a1_ozone_aqi": "observed ozone AQI, Paducah",
             "a2_ozone_aqi": "observed ozone AQI, Smithland", "m1_pm25_aqi": "modelled PM2.5 AQI at the monitor",
             "m1_ozone_aqi": "modelled ozone AQI at the monitor"}
ORI = {"f_corrected": "ORI, deployed model, corrected inputs", "f_refit": "ORI, refit"}
COLD = [10, 11, 12, 1, 2, 3]


def load():
    d = pd.read_csv(os.path.join(HERE, "refit_calvert_monitor_days.csv"))
    d = d[d.site.isin(S.RES) & ~d.poster_window]
    both = d.groupby("date").site.nunique()
    d = d[d.date.isin(both[both == 2].index)]
    a = pd.read_csv(os.path.join(HERE, "aqi_inputs", "aqi_daily.csv"))
    d = d.merge(a.drop(columns=["cams_lat", "cams_lon"]), on=["date", "site"], how="left", validate="one_to_one")
    d["month"] = pd.to_datetime(d.date).dt.month
    return d


def rho_of(d, col, y, B):
    return S.joint_date_perm(d.dropna(subset=[col]).assign(forecast=lambda x: x[col]), y, B)


def paired_named(d, y, ori, comp):
    r = S.paired(d, y, ori, comp)
    r["ori_rho"] = r.pop("refit_rho")
    r["reading"] = "ORI better" if r["ci95"][0] > 0 else ("AQI better" if r["ci95"][1] < 0 else "no detectable difference")
    return r


def high_days(d, y):
    g = d.dropna(subset=[y]).copy()
    q = g.groupby("site")[y].transform(lambda x: x.quantile(0.75))
    g["hi"] = (g[y] >= q) & ((g[y] > 0) if y == "vc" else True)
    return g


def good_shares(d):
    out = {}
    for y in ["vc", "benzene"]:
        g = high_days(d, y)
        g["ori_above_median"] = g.f_corrected > g.groupby("site").f_corrected.transform("median")
        per = {}
        for s, h in g.groupby("site"):
            hi = h[h.hi]
            row = {"high_days": int(len(hi)), "ori_above_median_on_high_days": round(float(hi.ori_above_median.mean()), 3)}
            for c in PRIMARY:
                row[f"{c}_good_on_high_days"] = round(float((hi[c].dropna() <= 50).mean()), 3)
                row[f"{c}_good_all_days"] = round(float((h[c].dropna() <= 50).mean()), 3)
                row[f"{c}_n_high_with_value"] = int(hi[c].notna().sum())
            per[s] = row
        out[y] = per
    return out


def main():
    d = load()
    res = {"plan_sha256": open(os.path.join(HERE, "AQI_COMPARISON_PLAN_HASH.txt")).read().split()[0],
           "sample": {"dates": int(d.date.nunique()), "monitor_days": int(len(d)),
                      "missing": {c: int(d[c].isna().sum()) for c in list(PRIMARY) + list(SECONDARY)}},
           "context": json.load(open(os.path.join(HERE, "aqi_inputs", "context.json")))}

    q1 = {c: {y: rho_of(d, c, y, S.B_PRIMARY) for y in ["vc", "benzene"]} for c in PRIMARY}
    keys = [(c, y) for c in PRIMARY for y in ["vc", "benzene"]]
    _, padj, _, _ = multipletests([q1[c][y]["p"] for c, y in keys], method="holm")
    for (c, y), pa in zip(keys, padj):
        q1[c][y]["p_holm"] = round(float(pa), 4)
        q1[c][y]["tracks"] = bool(q1[c][y]["rho"] > 0 and pa < 0.05)
    res["q1"] = q1
    res["ori_rho_same_sample"] = {o: {y: rho_of(d, o, y, 3000) for y in ["vc", "benzene"]} for o in ORI}

    res["q2"] = {c: {y: paired_named(d, y, "f_corrected", c) for y in ["vc", "benzene"]} for c in PRIMARY}

    sec = {}
    sec["refit_vs_aqi"] = {c: {y: paired_named(d, y, "f_refit", c) for y in ["vc", "benzene"]} for c in PRIMARY}
    sec["components"] = {c: {y: rho_of(d, c, y, 3000) for y in ["vc", "benzene"]} for c in SECONDARY}
    sec["total_voc"] = {"rho": {c: rho_of(d, c, "total_voc", 3000) for c in list(PRIMARY) + ["f_corrected"]},
                        "paired": {c: paired_named(d, "total_voc", "f_corrected", c) for c in PRIMARY}}
    sec["season"] = {}
    for name, m in [("cold (Oct-Mar)", d.month.isin(COLD)), ("warm (Apr-Sep)", ~d.month.isin(COLD))]:
        sec["season"][name] = {c: {y: rho_of(d[m], c, y, 3000) for y in ["vc", "benzene"]}
                               for c in list(PRIMARY) + ["a1_pm25_aqi", "a1_ozone_aqi", "f_corrected"]}
    res["secondary"] = sec
    res["good_days"] = good_shares(d)

    d.to_csv(os.path.join(HERE, "aqi_calvert_monitor_days.csv"), index=False)
    json.dump(res, open(os.path.join(HERE, "aqi_calvert_results.json"), "w"), indent=1)
    write_report(res)
    print(open(os.path.join(HERE, "AQI_CALVERT_RESULTS.md")).read())


def write_report(r):
    L = ["# Daily AQI versus ORI at the Calvert monitors", "",
         f"Plan `AQI_COMPARISON_PLAN.md` (SHA-256 {r['plan_sha256'][:12]}...), DEVIATIONS 30-31. "
         f"{r['sample']['dates']} shared dates, {r['sample']['monitor_days']} monitor-days. Missing comparator values "
         f"(left out, never filled): {r['sample']['missing']}.", "",
         "## Q1: does the AQI rank vinyl chloride and benzene days?", "",
         "| comparator | pollutant | mean within-monitor rho | p (joint date permutation) | Holm p | tracks? | dates |",
         "|---|---|---|---|---|---|---|"]
    for c, lab in PRIMARY.items():
        for y in ["vc", "benzene"]:
            q = r["q1"][c][y]
            L.append(f"| {lab} | {y} | {q['rho']} | {q['p']} | {q['p_holm']} | {'yes' if q['tracks'] else 'no'} | {q['shared_dates']} |")
    L += ["", "ORI on the same monitor-days: " + "; ".join(
        f"{ORI[o]}: vc {r['ori_rho_same_sample'][o]['vc']['rho']}, benzene {r['ori_rho_same_sample'][o]['benzene']['rho']}"
        for o in ORI) + ".", "",
        "## Q2: ORI (deployed, corrected inputs) minus AQI", "",
        "| comparator | pollutant | ORI rho | AQI rho | difference | 95% CI | p (swap) | reading |", "|---|---|---|---|---|---|---|---|"]
    for c, lab in PRIMARY.items():
        for y in ["vc", "benzene"]:
            q = r["q2"][c][y]
            L.append(f"| {lab} | {y} | {q['ori_rho']} | {q['comparator_rho']} | {q['diff']:+.3f} | {q['ci95'][0]:+.3f} to {q['ci95'][1]:+.3f} | {q['p_swap']} | {q['reading']} |")
    s = r["secondary"]
    L += ["", "## Secondary", "", "Refit minus AQI:", "",
          "| comparator | pollutant | refit rho | AQI rho | difference | 95% CI | reading |", "|---|---|---|---|---|---|---|"]
    for c, lab in PRIMARY.items():
        for y in ["vc", "benzene"]:
            q = s["refit_vs_aqi"][c][y]
            L.append(f"| {lab} | {y} | {q['ori_rho']} | {q['comparator_rho']} | {q['diff']:+.3f} | {q['ci95'][0]:+.3f} to {q['ci95'][1]:+.3f} | {q['reading']} |")
    L += ["", "AQI components (mean within-monitor rho, joint date-permutation p, dates):", "",
          "| component | vinyl chloride | benzene |", "|---|---|---|"]
    for c, lab in SECONDARY.items():
        v, b = s["components"][c]["vc"], s["components"][c]["benzene"]
        L.append(f"| {lab} | {v['rho']} (p {v['p']}, {v['shared_dates']}) | {b['rho']} (p {b['p']}, {b['shared_dates']}) |")
    t = s["total_voc"]
    L += ["", "Total VOC: " + "; ".join(f"{(PRIMARY | ORI)[c]} rho {q['rho']} (p {q['p']})" for c, q in t["rho"].items())
          + ". ORI minus AQI: " + "; ".join(f"{PRIMARY[c]} {q['diff']:+.3f} ({q['ci95'][0]:+.3f} to {q['ci95'][1]:+.3f}), {q['reading']}"
                                          for c, q in t["paired"].items()) + ".", "",
          "By season (mean within-monitor rho, p, dates):", "",
          "| season | series | vinyl chloride | benzene |", "|---|---|---|---|"]
    names = PRIMARY | ORI | SECONDARY
    for sea, block in s["season"].items():
        for c, q in block.items():
            L.append(f"| {sea} | {names[c]} | {q['vc']['rho']} (p {q['vc']['p']}, {q['vc']['shared_dates']}) | {q['benzene']['rho']} (p {q['benzene']['p']}, {q['benzene']['shared_dates']}) |")
    L += ["", "## High-VOC days on which the AQI said Good (0-50)", "",
          "| pollutant | monitor | high days | observed AQI Good on high days | observed AQI Good, all days | modelled AQI Good on high days | modelled AQI Good, all days | ORI above its median on high days |",
          "|---|---|---|---|---|---|---|---|"]
    for y, per in r["good_days"].items():
        for site, q in per.items():
            L.append(f"| {y} | {site} | {q['high_days']} | {q['a1_aqi_good_on_high_days']:.0%} (n {q['a1_aqi_n_high_with_value']}) | {q['a1_aqi_good_all_days']:.0%} | {q['m1_aqi_good_on_high_days']:.0%} | {q['m1_aqi_good_all_days']:.0%} | {q['ori_above_median_on_high_days']:.0%} |")
    c = r["context"]
    L += ["", f"Context: {c['us_counties_with_any_daily_aqi_2023_2025']} US counties and "
          f"{c['ky_counties_with_any_daily_aqi_2023_2025']} of Kentucky's 120 had any daily AQI in the 2023-2025 AirData "
          f"files; Marshall County (Calvert City) had none. CAMS grid points: {c['cams_grid_points']}."]
    open(os.path.join(HERE, "AQI_CALVERT_RESULTS.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
