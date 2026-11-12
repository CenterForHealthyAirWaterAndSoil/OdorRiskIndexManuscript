"""Planned sensitivity (PLUME_PLAN.md, Runs: "year-matched inventories are a later sensitivity check").

The runs use TRI 2024 for every date. HYSPLIT concentrations are linear in each source's emission rate (particle
positions do not depend on the mass they carry; decay and deposition are first order), so each facility x release
x species series in a run's con2stn.txt can be rescaled by (year-matched rate / rate in its CONTROL) without
re-running HYSPLIT. Year-matched: TRI 2023 for 2023 dates, 2024 for 2024 and 2025 dates (TRI 2025 not yet public).
Source: EPA TRI basic data files for Kentucky (data.epa.gov mv_tri_basic_download/<year>_KY), fugitive 5.1 +
stack 5.2. A rescale with every factor = 1 must reproduce predicted_clipped/ exactly (checked below).

Writes predicted_tri_yearmatched/<date>.csv (+ .DONE), then scores them with run_plume.py's own code, unchanged
except the input folder and output names -> plume_results_tri_yearmatched.json.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/path/to/plume-validation")
from validation import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/path/to/plume-validation/validation/runs_clipped"
PRED = "/path/to/plume-validation/validation/results/predicted_clipped"
OUT = os.path.join(HERE, "predicted_tri_yearmatched")
# EPA TRI basic files (data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/<year>_KY/csv), kept beside the scripts
TRI_DIR = os.path.join(HERE, "tri")
LB_TO_G_HR = 453.59237 / 8760.0
# run-directory slug -> TRI facility id (calvert_plume_engine_one_day.FACILITIES)
SLUG_TRIFD = {"westlake_vinyls": "42029WSTLK2468I", "westlake_pvc_plant": "42029PCFCWJOHNS", "arkema_inc": "42029PNNWLALTON",
              "isp_chemicals": "42029GFCHMHIGHW", "lubrizol_advanced_materials": "42029NVNNC2468I"}
TAG_SPECIES = {v["hysplit_tag"]: k for k, v in C.TARGET_SPECIES.items()}
TRI_NAME = {k: v["tri_name"] for k, v in C.TARGET_SPECIES.items()}


def tri_rates(year):
    """(trifd, species, 'stack'|'fugitive') -> g/hr from EPA's TRI basic file for that year."""
    t = pd.read_csv(os.path.join(TRI_DIR, f"tri_{year}_KY.csv"), dtype=str)
    t.columns = [re.sub(r"^\d+\.\s*", "", c).strip().lower() for c in t.columns]
    t = t[t.county.str.upper() == "MARSHALL"]
    out = {}
    for sp, name in TRI_NAME.items():
        for _, r in t[t.chemical.str.upper() == name].iterrows():
            for src, col in (("fugitive", "5.1 - fugitive air"), ("stack", "5.2 - stack air")):
                out[(r.trifd, sp, src)] = float(pd.to_numeric(r[col], errors="coerce") or 0.0) * LB_TO_G_HR
    return out


def read_control(path):
    """Pollutant order and emission rate (g/hr) from a run's CONTROL."""
    L = [l.strip() for l in open(path)]
    n_src = int(L[1]); i = 2 + n_src + 3 + 1 + 2   # start, nsrc, sources, hours, vert, top, nmet, dir, file
    n_pol = int(L[i]); i += 1
    pols = []
    for _ in range(n_pol):
        pols.append((L[i], float(L[i + 1]))); i += 4
    return pols


def run_series(run_dir):
    """{(pol_index, station): [hourly g/m3 ...]} from con2stn.txt (the parser run_backtest.extract uses)."""
    s = {}
    for line in open(os.path.join(run_dir, "con2stn.txt")).readlines()[1:]:
        p = line.split()
        if len(p) >= 15:
            s.setdefault((int(p[11]), p[13]), []).append(float(p[14]))
    return s


def predict(date, rates, use_factor=True):
    site_by_aqs = {m["aqs_site"]: k for k, m in C.MONITOR_SITES.items()}
    agg = {}
    for run_dir in sorted(glob.glob(os.path.join(RUNS, date, "*"))):
        name = os.path.basename(run_dir)
        if name == "bdyfiles" or not os.path.exists(os.path.join(run_dir, "HYSPLIT_OK")):
            continue
        slug, src = name.rsplit("_", 1)
        pols = read_control(os.path.join(run_dir, "CONTROL"))
        for (pi, stn), vals in run_series(run_dir).items():
            tag, rate24 = pols[pi - 1]; sp = TAG_SPECIES[tag]
            f = rates.get((SLUG_TRIFD[slug], sp, src), 0.0) / rate24 if use_factor else 1.0
            key = (site_by_aqs[stn], sp)
            prev = agg.get(key)
            v = np.array(vals) * f
            agg[key] = v if prev is None else prev + v
    rows = [{"date": date, "site": s, "species": sp, "pred_mean_ugm3": float(v.mean() * C.GM3_TO_UGM3)} for (s, sp), v in agg.items()]
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    rates = {y: tri_rates(y) for y in (2023, 2024)}
    dates = sorted(os.path.basename(p)[:10] for p in glob.glob(os.path.join(PRED, "*.DONE")))
    worst = 0.0
    for d in dates:
        orig = pd.read_csv(os.path.join(PRED, f"{d}.csv"))
        same = predict(d, rates[2024], use_factor=False).merge(orig, on=["date", "site", "species"], how="right", suffixes=("", "_orig"))
        worst = max(worst, float((same.pred_mean_ugm3.fillna(0) - same.pred_mean_ugm3_orig).abs().max()))
        ym = predict(d, rates[2023 if d < "2024-01-01" else 2024])
        ym = orig[["date", "site", "species"]].merge(ym, on=["date", "site", "species"], how="left").fillna({"pred_mean_ugm3": 0.0})
        ym.to_csv(os.path.join(OUT, f"{d}.csv"), index=False)
        open(os.path.join(OUT, f"{d}.DONE"), "w").close()
    print(f"{len(dates)} dates rescaled; reproduction check (factor 1) max |diff| = {worst:.2e} ug/m3")
    assert worst < 1e-6, "rescaling with factor 1 does not reproduce predicted_clipped"
    # score with run_plume.py's own statistics code
    src = open(os.path.join(HERE, "run_plume.py")).read()
    src = src.replace('PRED_DIR = "/path/to/plume-validation/validation/results/predicted_clipped"', f'PRED_DIR = "{OUT}"')
    src = src.replace('"plume_results.json"', '"plume_results_tri_yearmatched.json"').replace('"plume_monitor_days.csv"', '"plume_monitor_days_tri_yearmatched.csv"')
    assert f'PRED_DIR = "{OUT}"' in src and "plume_results_tri_yearmatched.json" in src
    exec(compile(src, "run_plume.py (TRI year-matched inputs)", "exec"), {"__file__": os.path.join(HERE, "run_plume.py"), "__name__": "__main__"})


if __name__ == "__main__":
    main()
