"""Summary tables, built only from the package's datasets and result files (rebuilt 2026-11-11: the ORI is the 2026 refit,
DEVIATIONS 40; Table S1 keeps the earlier version's results).
Writes CSVs to 05_results/tables/ and prints Markdown versions.
Run: ~/weather-varaible-analysis/.venv/bin/python 04_code/make_tables.py
"""
import json
import os

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D, R = os.path.join(PKG, "02_analysis_datasets"), os.path.join(PKG, "05_results")
OUT = os.path.join(R, "tables"); os.makedirs(OUT, exist_ok=True)
J = lambda *p: json.load(open(os.path.join(R, *p)))
UG = {"vinyl_chloride_ppbC": (2, 62.5), "benzene_ppbC": (6, 78.1)}
ug = lambda s, c: s.clip(lower=0) / UG[c][0] * UG[c][1] / 24.45

# ---------- Table 1: monitors and measurements ----------
cur = pd.read_csv(os.path.join(D, "calvert_monitor_days_2023_2025.csv"), dtype={"aqs_site_id": str})
old = pd.read_csv(os.path.join(D, "older_monitor_days_2011_2022.csv"), dtype={"aqs_site_id": str})
inv = pd.read_csv(os.path.join(D, "monitors.csv"), dtype={"site_id": str}).set_index("site_id")
rows = []
for period, t, role in [("2023-2025", cur, lambda s: "inside complex" if s == "21-157-0021" else "community"),
                        ("2011-2022", old, lambda s: "set aside (in-plant)" if s in ("21-157-0014", "21-157-0021") else "community")]:
    for s, g in t.groupby("aqs_site_id"):
        rows.append({"Monitor": g.monitor.iloc[0], "AQS site": s, "Period": period, "Role": role(s),
                     "km from complex centre": round(inv.loc[s, "dist_complex_km"], 2) if s in inv.index else "",
                     "Sampling days": len(g),
                     "Vinyl chloride detected (%)": round(100 * (g.vinyl_chloride_ppbC > 0).mean()),
                     "Vinyl chloride mean (ug/m3)": round(ug(g.vinyl_chloride_ppbC, "vinyl_chloride_ppbC").mean(), 2),
                     "Benzene mean (ug/m3)": round(ug(g.benzene_ppbC, "benzene_ppbC").mean(), 2)})
t1 = pd.DataFrame(rows); t1.to_csv(os.path.join(OUT, "Table1_monitors_and_measurements.csv"), index=False)


# ---------- Table 2: every pre-registered primary result ----------
def cell(r):
    ph = r.get("p_holm", r.get("p"))
    return f"{r['rho']:.3f} ({'Holm ' if 'p_holm' in r else ''}p {ph:.4f})"


pre, std = J("ori", "results.json"), J("ori", "standard_source_results.json")
hy, rf = J("hysplit", "plume_results_new156.json"), J("refit", "refit_calvert_results.json")
ab, c = J("nov2026", "nov2026_ab_results.json"), J("nov2026", "c_results.json")
ad, adr = J("nov2026", "all_dates_results.json"), J("nov2026", "all_dates_refit_results.json")
# Since 2026-11-11 the paper's ORI is the 2026 refit (DEVIATIONS 40); the deployed model is "the earlier version" (Table S1).
older_refit = {k: {"rho": c["all_scores"]["refit"][k]["rho"], "p": c["all_scores"]["refit"][k]["p"]} for k in ("vc", "benzene")}
rows = [
    ("ORI (2026 refit), all dates: the paper's ORI", "REFIT_PLAN; DEVIATIONS 39-40 (post hoc sample)", "282 monitor-days, 141 dates, 2023-2025",
     adr["scores"]["ori_refit"], "n/a"),
    ("ORI (2026 refit), dates specified in advance", "REFIT_PLAN", "230 monitor-days, 115 dates", rf["refit_analysis_plan"]["primary"],
     rf["refit_analysis_plan"]["decision"]),
    ("HYSPLIT, all 156 new dates", "PLUME_PLAN + DEVIATIONS 16, 22", "230 monitor-days", hy["primary"], hy["decision"]),
    ("ORI + hours downwind (with the earlier ORI)", "NOV2026_PLAN A", "same", ab["A_combo"]["primary"], ab["A_combo"]["primary"]["decision"]),
    ("Weather model fitted at Calvert", "NOV2026_PLAN B", "same", ab["B_calvert_fitted"]["primary"],
     ab["B_calvert_fitted"]["primary"]["decision"]),
    ("Earlier ORI on older monitors (ERA5 inputs)", "NOV2026_PLAN C", "1,396 monitor-days, 2011-2022, 5 monitors", c["primary"], c["decision"])]
if older_refit:
    rows.append(("ORI (2026 refit) on older monitors", "NOV2026_PLAN C (secondary)", "same", older_refit, "n/a"))
t2 = pd.DataFrame([{"Analysis": a, "Plan": p, "Sample": n, "Vinyl chloride rho": cell(r["vc"]), "Benzene rho": cell(r["benzene"]),
                    "Decision": dec} for a, p, n, r, dec in rows])
t2.to_csv(os.path.join(OUT, "Table2_primary_results.csv"), index=False)


# ---------- Table 3: head-to-head comparisons on identical monitor-days (the paper's, all 141 dates, refit ORI) ----------
def pair(lab, pol, r, a_name="a", b_name="b"):
    return {"Comparison (a minus b)": lab, "Pollutant": pol, "a rho": round(r.get("refit_rho", r.get("a_rho", r.get("ori_rho"))), 3),
            "b rho": round(r.get("comparator_rho", r.get("b_rho")), 3), "Difference": round(r["diff"], 3),
            "95% CI": f"{r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}",
            "Reading": "a better" if r["ci95"][0] > 0 else ("b better" if r["ci95"][1] < 0 else "no detectable difference")}


POL = {"vc": "vinyl chloride", "edc": "EDC", "benzene": "benzene", "butadiene": "1,3-butadiene", "total_voc": "total VOC"}
rows = []
for key, r in adr["paired"].items():
    lab, y = key.split(" | ")
    rows.append(pair(lab + (" (post hoc)" if "HYSPLIT" in lab else ""), POL[y], r))
for k in ("vc", "benzene"):
    for lab, key in [("Older monitors: Calvert-fitted minus ORI", f"{k}: refit minus calvert_fitted"),
                     ("Older monitors: hours downwind minus ORI", f"{k}: refit minus hours_downwind")]:
        r = c["paired"][key]                                      # stored as ORI minus other: flip the sign
        rows.append(pair(lab, POL[k], {"a_rho": r["b_rho"], "b_rho": r["a_rho"], "diff": -r["diff"],
                                       "ci95": [-r["ci95"][1], -r["ci95"][0]]}))
t3 = pd.DataFrame(rows); t3.to_csv(os.path.join(OUT, "Table3_head_to_head.csv"), index=False)


# ---------- Table S1: the earlier version of the ORI (deployed model), as reported ----------
def ecell(r):
    from decimal import Decimal, ROUND_HALF_UP          # stored rho has 3 decimals; round half up like the text (0.495 -> 0.50)
    v = Decimal(str(r["rho"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{v} (p {r['p']:.4f})" if r.get("p") is not None else f"{v}"


s1 = [("All 2023-2025 dates (the paper's sample)", "282 monitor-days, 141 dates",
       ecell(ad["scores"]["ori_corrected"]["vc"]), ecell(ad["scores"]["ori_corrected"]["benzene"]),
       ecell(ad["scores"]["ori_corrected"]["total_voc"])),
      ("Dates specified in advance (first analysis plan)", "230 monitor-days, 115 dates",
       ecell(std["primary"]["vc"]), ecell(std["primary"]["benzene"]), ""),   # total VOC there used the pre-38 definition
      ("Older monitors, 2011-2022 (ERA5 inputs)", "1,396 monitor-days, 621 dates",
       ecell(c["primary"]["vc"]), ecell(c["primary"]["benzene"]), "")]
ts1 = pd.DataFrame(s1, columns=["Sample", "Monitor-days", "Vinyl chloride rho", "Benzene rho", "Total VOC rho"])
cmp = []
for key in ["ORI minus HYSPLIT | vc", "ORI minus HYSPLIT | benzene", "ORI minus daily AQI (Paducah) | vc",
            "ORI minus daily AQI (Paducah) | benzene"]:
    r = ad["paired"][key]
    cmp.append({"Comparison (earlier ORI minus other), all dates": key.replace(" | ", ", "), "Difference": round(r["diff"], 3),
                "95% CI": f"{r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}"})
r = adr["paired"]["ORI refit minus earlier version | vc"]; r2 = adr["paired"]["ORI refit minus earlier version | benzene"]
cmp.append({"Comparison (earlier ORI minus other), all dates": "refit minus earlier version, vinyl chloride",
            "Difference": round(r["diff"], 3), "95% CI": f"{r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}"})
cmp.append({"Comparison (earlier ORI minus other), all dates": "refit minus earlier version, benzene",
            "Difference": round(r2["diff"], 3), "95% CI": f"{r2['ci95'][0]:+.3f} to {r2['ci95'][1]:+.3f}"})
ts1.to_csv(os.path.join(OUT, "TableS1_earlier_version.csv"), index=False)
pd.DataFrame(cmp).to_csv(os.path.join(OUT, "TableS1b_earlier_version_comparisons.csv"), index=False)


def md_table(t):
    head = "| " + " | ".join(map(str, t.columns)) + " |\n|" + "---|" * len(t.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False))


for t, title in [(t1, "Table 1"), (t2, "Table 2"), (t3, "Table 3"), (ts1, "Table S1"), (pd.DataFrame(cmp), "Table S1b")]:
    print(f"\n{title}\n"); print(md_table(t))
