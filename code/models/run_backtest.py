"""
Resumable HYSPLIT back-test batch driver.

For each monitor sample date:
  1. Ensure the ARL met file (HRRR, local Central day) exists   [reuses engine]
  2. For each source facility x release type (stack/fugitive), write ONE multi-species
     CONTROL (single source point, all target species that facility emits that way) and
     run hycs_std -> cdump.
  3. con2stn extracts the 10 m air concentration at the 3 monitors for every species/hour.
  4. Aggregate to a predicted 24-h mean (ug/m^3) per (date, site, species), summed over
     facilities and release types. Also keep peak-hour and contact-hours.
  5. Write results/predicted/<date>.csv and a DONE marker (resumable).

We reuse ``calvert_plume_engine_one_day.CalvertCityPlumeEngine`` for the tested pieces
(Central-day windowing, HRRR download, GRIB->ARL, TRI parsing, facility matching). The
multi-species CONTROL writer here mirrors the engine's vetted single-species
``write_deposition_control_file`` structure, extended to N pollutants.

Usage:
  python3 -m validation.run_backtest --pilot           # 8 stratified dates
  python3 -m validation.run_backtest --dates 2025-02-15 2025-03-08
  python3 -m validation.run_backtest --all             # all 32 sample dates
  python3 -m validation.run_backtest --all --force     # ignore DONE markers
"""
import argparse
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import config as C
from validation import extract_monitors

from calvert_plume_engine_one_day import CalvertCityPlumeEngine

HYSPLIT_EXEC = "/path/to/hysplit/exec"
HYCS_STD = os.path.join(HYSPLIT_EXEC, "hycs_std")
BDYFILES = os.path.join(os.path.dirname(HYSPLIT_EXEC), "bdyfiles")

PREDICTED_DIR = os.path.join(C.RESULTS_DIR, "predicted")

# Disk safety. A full-CONUS HRRR ARL is ~12 GB and its source GRIB another ~12 GB, so a date
# needs ~28 GB transient headroom. We delete GRIB right after conversion and the ARL right after
# the date's cdumps are extracted, so steady-state footprint stays ~one date, not 32.
MIN_FREE_GB_MET = 28


def fac_slug(name):
    return name.lower().replace(" ", "_").replace(".", "").replace("&", "and")


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def require_free_gb(min_gb, path, context=""):
    avail = free_gb(path)
    if avail < min_gb:
        raise RuntimeError(
            f"Insufficient disk: {avail:.1f} GB free at {path}, need >= {min_gb} GB "
            f"({context}). Free space or pass --keep-met=off; aborting to avoid filling the disk.")


def _delete_grib(grib_paths):
    """Delete raw GRIB files (and Herbie .idx siblings) after they've been folded into the ARL."""
    freed = 0
    for g in grib_paths or []:
        for p in (g, g + ".idx", str(g) + ".idx"):
            try:
                if os.path.exists(p):
                    freed += os.path.getsize(p)
                    os.remove(p)
            except OSError:
                pass
    if freed:
        print(f"  cleaned raw GRIB (~{freed / 2**30:.1f} GB)")


class Backtest:
    def __init__(self):
        # Engine rooted at the worktree; it owns HRRR download + GRIB->ARL + TRI parsing.
        self.engine = CalvertCityPlumeEngine(C.REPO_DIR)
        os.makedirs(C.MET_DIR, exist_ok=True)
        os.makedirs(PREDICTED_DIR, exist_ok=True)
        # tri_name -> species_key lookup
        self.by_tri = {v["tri_name"]: k for k, v in C.TARGET_SPECIES.items()}

    # -- met --------------------------------------------------------------
    def ensure_met(self, date):
        eng = self.engine
        eng.set_active_date(date)
        # Redirect the ARL to our cache dir (CONTROL writer reads dirname/basename of this).
        met_name = f"MET_{eng.date_obj.strftime('%Y%m%d')}.ARL"
        eng.met_file_path = os.path.join(C.MET_DIR, met_name)
        if os.path.exists(eng.met_file_path) and os.path.getsize(eng.met_file_path) > 0:
            print(f"[{date}] met cached: {met_name}")
            return
        # Disk guard: a full CONUS ARL (~12 GB) plus its source GRIB (~12 GB) transiently
        # coexist during conversion. Abort this date if there isn't headroom, rather than
        # crash mid-write and fill the disk.
        require_free_gb(MIN_FREE_GB_MET, C.MET_DIR, context=f"met build for {date}")
        print(f"[{date}] downloading HRRR + converting to ARL ...")
        grib = eng.download_weather_data()
        eng.convert_grib_to_arl(grib)
        # Delete the raw GRIB immediately - it's a transient intermediate, and keeping 32x of
        # it (one per date) is what filled the disk. The ARL persists until the date is done.
        _delete_grib(grib)

    def cleanup_met(self, date):
        """Remove this date's ARL once its cdumps are extracted. ~12 GB/date reclaimed."""
        if getattr(self, "keep_met", False):
            return
        met = os.path.join(C.MET_DIR, f"MET_{date.replace('-', '')}.ARL")
        if os.path.exists(met):
            try:
                os.remove(met)
                print(f"[{date}] removed ARL (cleanup) - reclaimed ~{os.path.getsize(met)//2**30 if os.path.exists(met) else 12} GB")
            except OSError as e:
                print(f"[{date}] could not remove ARL: {e}")

    # -- emissions --------------------------------------------------------
    def emissions_for(self, date):
        """{facility_name: {'stack'|'fugitive': {species_key: g_hr}}} for target species only."""
        out = {}
        for fac in self.engine.facilities:
            releases = self.engine.get_facility_releases(fac["name"]).get("releases", [])
            for r in releases:
                skey = self.by_tri.get(r["chemical"].upper())
                if skey is None:
                    continue
                for src, rate in (("stack", r.get("stack_g_hr", 0.0)),
                                  ("fugitive", r.get("fugitive_g_hr", 0.0))):
                    if rate and rate > 0:
                        out.setdefault(fac["name"], {}).setdefault(src, {})[skey] = rate
        return out

    # -- CONTROL (multi-species, single source point) ---------------------
    def write_control(self, run_dir, fac, height, species_rates):
        """species_rates: {species_key: g_hr}. Mirrors the engine's dep CONTROL, N pollutants."""
        eng = self.engine
        order = list(species_rates.keys())            # pollutant order == deposition-block order
        start = eng.sim_start_utc.strftime("%y %m %d %H")
        rel_start = eng.sim_start_utc.strftime("%y %m %d %H %M")
        met_dir = os.path.dirname(eng.met_file_path) + "/"
        met_file = os.path.basename(eng.met_file_path)

        lines = [
            start, "1",
            f"{fac['lat']:.4f} {fac['lon']:.4f} {height:.1f}",
            str(C.RUN_HOURS), "0", "10000.0",
            "1", met_dir, met_file,
            str(len(order)),
        ]
        for skey in order:
            sp = C.TARGET_SPECIES[skey]
            lines += [sp["hysplit_tag"], f"{species_rates[skey]:.6f}", "23.0", rel_start]
        # concentration grid
        lines += [
            "1",
            f"{C.GRID_CENTER[0]:.4f} {C.GRID_CENTER[1]:.4f}",
            f"{C.GRID_SPACING:.3f} {C.GRID_SPACING:.3f}",
            f"{C.GRID_SPAN:.3f} {C.GRID_SPAN:.3f}",
            run_dir + "/", "cdump",
            "2", "0", "10",
            "00 00 00 00 00", "00 00 00 00 00", "00 01 00",
        ]
        # deposition blocks (one per pollutant, same order). VOCs: wet params 0.
        lines.append(str(len(order)))
        for skey in order:
            sp = C.TARGET_SPECIES[skey]
            H = sp["henry_M_atm"]
            lines += [
                "0.0 0.0 0.0",
                f"{sp['dry_vd_ms']:.4f} {sp['mol_weight']:.1f} 0.0 0.0 {H:.4f}",
                f"{H:.4f} 0.00E+00 0.00E+00",
                f"{sp['halflife_days']:.2f}",
                "0.0",
            ]
        with open(os.path.join(run_dir, "CONTROL"), "w") as f:
            f.write("\n".join(lines) + "\n")
        return order

    def write_setup(self, run_dir):
        content = ("&SETUP\n  initd = 0,\n  khmax = 24,\n  ndump = 0,\n"
                   f"  ncycl = 1,\n  numpar = {C.NUMPAR},\n  maxpar = {C.MAXPAR},\n/\n")
        for name in ("SETUP.cfg", "SETUP.CFG"):
            with open(os.path.join(run_dir, name), "w") as f:
                f.write(content)

    def run_hysplit(self, run_dir):
        # HYSPLIT resolves landuse/roughness via ../bdyfiles/ASCDATA.CFG (see engine pattern):
        # the bdyfiles symlink lives one level UP, in the date dir (created in run_date()).
        # A run counts as finished only when HYSPLIT exits cleanly and this writes HYSPLIT_OK.
        # A timed-out run can leave a partial cdump; delete it so it is never reused as complete.
        ok_marker = os.path.join(run_dir, "HYSPLIT_OK")
        cdump = os.path.join(run_dir, "cdump")
        if os.path.exists(ok_marker):
            os.remove(ok_marker)
        try:
            res = subprocess.run([HYCS_STD], cwd=run_dir, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, timeout=C.HYSPLIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            if os.path.exists(cdump):
                os.remove(cdump)
            return False
        ok = res.returncode == 0 and os.path.exists(cdump) and os.path.getsize(cdump) > 0
        if ok:
            open(ok_marker, "w").close()
        return ok

    # -- extraction -------------------------------------------------------
    def write_stationfile(self, path):
        with open(path, "w") as f:
            for key, m in C.MONITOR_SITES.items():
                f.write(f"{m['aqs_site']} {m['lat']:.6f} {m['lon']:.6f}\n")

    def extract(self, run_dir, order):
        """con2stn -> {(site_key, species_key): [hourly g/m^3 ...]}."""
        stn = os.path.join(run_dir, "stations.txt")
        self.write_stationfile(stn)
        out = os.path.join(run_dir, "con2stn.txt")
        cmd = [C.CON2STN_PATH, "-icdump", f"-o{out}", f"-s{stn}",
               "-p0", f"-z{C.AIR_LEVEL_INDEX}", "-r0", "-e1", "-xi"]
        subprocess.run(cmd, cwd=run_dir, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, timeout=120)
        site_by_aqs = {m["aqs_site"]: k for k, m in C.MONITOR_SITES.items()}
        series = {}
        with open(out) as f:
            header = f.readline()  # skip
            for line in f:
                parts = line.split()
                if len(parts) < 15:
                    continue
                try:
                    pol = int(parts[11]); station = parts[13]; val = float(parts[14])
                except (ValueError, IndexError):
                    continue
                skey = order[pol - 1] if 1 <= pol <= len(order) else None
                site_key = site_by_aqs.get(station)
                if skey is None or site_key is None:
                    continue
                series.setdefault((site_key, skey), []).append(val)
        return series

    # -- per-date orchestration ------------------------------------------
    def run_date(self, date, force=False):
        marker = os.path.join(PREDICTED_DIR, f"{date}.DONE")
        if os.path.exists(marker) and not force:
            print(f"[{date}] already done (marker present) - skipping")
            self.cleanup_met(date)  # shed any orphan ARL left by an earlier (pre-cleanup) run
            return
        self.ensure_met(date)
        emis = self.emissions_for(date)
        date_runs = os.path.join(C.RUNS_DIR, date)
        os.makedirs(date_runs, exist_ok=True)
        # bdyfiles symlink at the date level; run subdirs reach it via ../bdyfiles/ (matches engine).
        bdylink = os.path.join(date_runs, "bdyfiles")
        if not os.path.exists(bdylink) and os.path.isdir(BDYFILES):
            try:
                os.symlink(BDYFILES, bdylink)
            except OSError:
                pass

        # accumulate hourly g/m^3 per (site, species) across facilities+release types
        agg = {}  # (site, species) -> list-of-lists (one per run) to sum elementwise
        fac_by_name = {f["name"]: f for f in self.engine.facilities}
        for fac_name, by_src in emis.items():
            fac = fac_by_name[fac_name]
            for src, species_rates in by_src.items():
                height = fac["height"] if src == "stack" else 2.0
                run_dir = os.path.join(date_runs, f"{fac_slug(fac_name)}_{src}")
                os.makedirs(run_dir, exist_ok=True)
                if not os.path.exists(os.path.join(run_dir, "HYSPLIT_OK")):
                    order = self.write_control(run_dir, fac, height, species_rates)
                    self.write_setup(run_dir)
                    print(f"[{date}] HYSPLIT {fac_name} ({src}) - {len(order)} species @ {height:.0f}m")
                    if not self.run_hysplit(run_dir):
                        # Never write a date's prediction with a source missing: fail the date so it
                        # gets no DONE marker and is re-run, instead of silently under-predicting.
                        raise RuntimeError(f"HYSPLIT failed for {fac_name} ({src}); date not written")
                else:
                    order = list(species_rates.keys())
                    print(f"[{date}] reuse finished run {fac_name} ({src})")
                series = self.extract(run_dir, order)
                for key, vals in series.items():
                    agg.setdefault(key, []).append(vals)

        # write predicted.csv
        rows = []
        for (site_key, skey), runs in agg.items():
            n = max(len(v) for v in runs)
            summed = [0.0] * n
            for v in runs:
                for i, x in enumerate(v):
                    summed[i] += x
            summed_ug = [x * C.GM3_TO_UGM3 for x in summed]
            mean_ug = sum(summed_ug) / len(summed_ug) if summed_ug else 0.0
            peak_ug = max(summed_ug) if summed_ug else 0.0
            contact = sum(1 for x in summed_ug if x > 0)
            rows.append({
                "date": date, "site": site_key, "species": skey,
                "pred_mean_ugm3": round(mean_ug, 8),
                "pred_peak_ugm3": round(peak_ug, 8),
                "pred_contact_hours": contact,
                "n_hours": len(summed_ug),
            })
        # ensure every (site, species) present, even if all-zero (no emissions reached)
        present = {(r["site"], r["species"]) for r in rows}
        for site_key in C.MONITOR_SITES:
            for skey in C.TARGET_SPECIES:
                if (site_key, skey) not in present:
                    rows.append({"date": date, "site": site_key, "species": skey,
                                 "pred_mean_ugm3": 0.0, "pred_peak_ugm3": 0.0,
                                 "pred_contact_hours": 0, "n_hours": C.RUN_HOURS})
        rows.sort(key=lambda r: (r["site"], r["species"]))
        path = os.path.join(PREDICTED_DIR, f"{date}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        open(marker, "w").close()
        nz = sum(1 for r in rows if r["pred_mean_ugm3"] > 0)
        print(f"[{date}] wrote {path} ({nz}/{len(rows)} nonzero predictions)")
        # cdumps are extracted and predictions saved - the 12 GB ARL is no longer needed.
        self.cleanup_met(date)

    def run_dates(self, dates, force=False):
        for i, d in enumerate(dates, 1):
            print(f"\n===== [{i}/{len(dates)}] {d} =====")
            try:
                self.run_date(d, force=force)
            except Exception as e:
                print(f"[{d}] ERROR: {e}")


def pilot_dates(n=8):
    """Stratified pick: n/2 highest and n/2 lowest days by peak vinyl-chloride across sites."""
    obs, dates = extract_monitors.extract()
    peak = {}
    for row in obs:
        if row["species"] == "vinyl_chloride":
            peak[row["date"]] = max(peak.get(row["date"], 0.0), float(row["value_ugm3"] or 0.0))
    ranked = sorted(peak, key=lambda d: peak[d])
    half = n // 2
    picked = ranked[:half] + ranked[-half:]
    return sorted(set(picked))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="run 8 stratified pilot dates")
    ap.add_argument("--all", action="store_true", help="run all 32 sample dates")
    ap.add_argument("--dates", nargs="+", help="explicit YYYY-MM-DD dates")
    ap.add_argument("--force", action="store_true", help="ignore DONE markers")
    ap.add_argument("--keep-met", action="store_true",
                    help="keep each date's 12 GB ARL after processing (default: delete to save disk)")
    args = ap.parse_args()

    if args.dates:
        dates = args.dates
    elif args.pilot:
        dates = pilot_dates(8)
    elif args.all:
        dates = extract_monitors.sample_dates()
    else:
        ap.error("choose --pilot, --all, or --dates")

    print(f"Running {len(dates)} date(s): {dates}")
    bt = Backtest()
    bt.keep_met = args.keep_met
    bt.run_dates(dates, force=args.force)


if __name__ == "__main__":
    main()
