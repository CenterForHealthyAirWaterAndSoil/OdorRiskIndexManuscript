"""
Disk-light variant of run_backtest.py for long date lists.

The original pipeline downloads 24 full-CONUS HRRR hours (~9 GB), converts them to one ~13 GB ARL,
then runs HYSPLIT, so a date needs ~28 GB free. Here each hour is handled on its own:

  1. Download only the GRIB fields api2arl uses (verified byte-identical ARL output).
  2. Convert that hour with api2arl_v6, exactly as the engine does.
  3. Clip the hour to a 4.0 x 4.5 degree box around Calvert City with HYSPLIT's xtrct_grid (~5 MB).
  4. Append to the date's ARL and delete the hour's GRIB and full ARL before the next hour.

Peak disk is about one HRRR hour (~1 GB). Everything after the met file (CONTROL, HYSPLIT,
con2stn, aggregation) is the unchanged Backtest code. Outputs go to separate folders so the
poster-era runs and predictions are never overwritten.

Usage (run with an interpreter that has herbie, e.g. ~/plume-analysis/.venv/bin/python):
  python -m validation.run_backtest_clipped --dates 2025-02-15
  python -m validation.run_backtest_clipped --dates-file path/to/plume_dates.txt
"""
import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import config as C
from validation import run_backtest as RB

# Clipped outputs live apart from the poster-era results.
C.MET_DIR = os.path.join(C.VALIDATION_DIR, "met_clipped")
C.RUNS_DIR = os.path.join(C.VALIDATION_DIR, "runs_clipped")
RB.PREDICTED_DIR = os.path.join(C.RESULTS_DIR, "predicted_clipped")

BOX_LOWER_LEFT = (35.0, -90.5)
BOX_UPPER_RIGHT = (39.0, -86.0)
N_LEVELS = 41  # all HRRR levels incl. surface
# Pressure-level fields api2arl reads, plus every non-pressure-level message.
GRIB_SUBSET = r"(:(HGT|TMP|UGRD|VGRD|VVEL|RH|SPFH):[0-9]+ mb:)|(:[A-Z0-9]+:(?![0-9]+ mb:))"
XTRCT_GRID = os.path.join(os.path.dirname(RB.HYSPLIT_EXEC), "exec", "xtrct_grid")
MIN_FREE_GB = 3


NIGHT_WINDOW = (23, 7)   # local hours for --night-only downloads (start, end)
NIGHT_ONLY = False


def wait_for_window():
    """In night-only mode, sleep until the download window opens (keeps daytime internet free)."""
    while NIGHT_ONLY:
        h = datetime.datetime.now().hour
        start, end = NIGHT_WINDOW
        if (h >= start or h < end) if start > end else (start <= h < end):
            return
        time.sleep(300)


def wait_for_network(max_hours=6):
    """Block until the HRRR archive answers, instead of failing dates during an outage."""
    import requests
    waited = 0
    while True:
        try:
            requests.head("https://noaa-hrrr-bdp-pds.s3.amazonaws.com/", timeout=15)
            if waited:
                print(f"  internet back after {waited // 60} min", flush=True)
            return
        except Exception:
            if waited == 0:
                print("  no internet; waiting (dates are not skipped)", flush=True)
            if waited >= max_hours * 3600:
                raise RuntimeError(f"no internet for {max_hours} h")
            time.sleep(60); waited += 60


class ClippedBacktest(RB.Backtest):
    def ensure_met(self, date):
        from herbie import Herbie

        eng = self.engine
        eng.set_active_date(date)
        met = os.path.join(C.MET_DIR, f"MET_{eng.date_obj.strftime('%Y%m%d')}.ARL")
        eng.met_file_path = met
        if os.path.exists(met) and os.path.getsize(met) > 0:
            print(f"[{date}] clipped met cached")
            return
        RB.require_free_gb(MIN_FREE_GB, C.MET_DIR, context=f"clipped met for {date}")
        work = os.path.join(C.MET_DIR, f"_work_{date}")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        part = met + ".part"
        if os.path.exists(part):
            os.remove(part)
        env = dict(os.environ, ECCODES_DEFINITION_PATH=eng.eccodes_defs_path)
        box = (f"./\nDATA.ARL\n{BOX_LOWER_LEFT[0]} {BOX_LOWER_LEFT[1]}\n"
               f"{BOX_UPPER_RIGHT[0]} {BOX_UPPER_RIGHT[1]}\n{N_LEVELS}\n")
        t0, mb = time.time(), 0.0
        for k in range(24):
            dt_hour = eng.sim_start_utc + datetime.timedelta(hours=k)
            # A download that dies mid-transfer leaves a partial GRIB, and a corrupt input made
            # xtrct_grid crash (exit 132) on 8 dates. So a failed conversion re-downloads the hour.
            for conv_try in range(3):
                grib = self.download_hour(date, k, dt_hour, work)
                gdir, gname = os.path.dirname(grib), os.path.basename(grib)
                for f in ("DATA.ARL", "extract.bin", "arldata.cfg"):
                    if os.path.exists(os.path.join(gdir, f)):
                        os.remove(os.path.join(gdir, f))
                try:
                    subprocess.run([eng.api2arl_path, f"-i{gname}", "-oDATA.ARL", "-z1"], cwd=gdir, env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=True)
                    subprocess.run([XTRCT_GRID], cwd=gdir, input=box, text=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=True)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"[{date}] hour {k:02d} conversion failed ({e}); re-downloading", flush=True)
                    shutil.rmtree(gdir, ignore_errors=True)
            else:
                raise RuntimeError(f"hour {k:02d} ({dt_hour}Z) failed conversion after 3 downloads")
            mb += os.path.getsize(grib) / 2**20
            with open(part, "ab") as out, open(os.path.join(gdir, "extract.bin"), "rb") as src:
                shutil.copyfileobj(src, out)
            shutil.rmtree(gdir)  # GRIB, idx, full-domain ARL, clipped hour
        shutil.rmtree(work, ignore_errors=True)
        os.replace(part, met)
        print(f"[{date}] clipped met built: {os.path.getsize(met) / 2**20:.0f} MB from {mb:.0f} MB "
              f"downloaded in {(time.time() - t0) / 60:.1f} min")

    def download_hour(self, date, k, dt_hour, work):
        from herbie import Herbie

        for attempt in range(6):
            wait_for_window()
            wait_for_network()
            try:
                # overwrite=True: Herbie otherwise returns any file already at the path, including the
                # partial file a failed attempt left behind.
                grib = Herbie(dt_hour, model="hrrr", product="prs", fxx=0, save_dir=work,
                              verbose=False).download(GRIB_SUBSET, overwrite=True, verbose=False)
                if grib and os.path.exists(grib):
                    return grib
            except Exception as e:
                print(f"[{date}] hour {k:02d} download failed ({str(e)[:120]}); retry {attempt + 1}", flush=True)
            time.sleep(min(30 * 2 ** attempt, 900))
        raise RuntimeError(f"hour {k:02d} ({dt_hour}Z) could not be downloaded")

    def cleanup_met(self, date):
        # Called once the date's predictions are written. PARDUMP (particle positions, ~170 MB/date) is not read
        # by the extraction, and 110 dates of it would fill the disk; cdump and con2stn.txt are kept.
        # Skip the bdyfiles symlink, which points at HYSPLIT's shared install.
        for run_dir in glob.glob(os.path.join(C.RUNS_DIR, date, "*")):
            pardump = os.path.join(run_dir, "PARDUMP")
            if not os.path.islink(run_dir) and os.path.isfile(pardump):
                os.remove(pardump)
        # Clipped files are ~130 MB/date; drop them anyway to keep the footprint flat.
        if getattr(self, "keep_met", False):
            return
        met = os.path.join(C.MET_DIR, f"MET_{date.replace('-', '')}.ARL")
        if os.path.exists(met):
            os.remove(met)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+")
    ap.add_argument("--dates-file")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-met", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="dates to run at once (default 1)")
    ap.add_argument("--night-only", action="store_true", help="download only 11 PM to 7 AM local time")
    args = ap.parse_args()
    global NIGHT_ONLY
    NIGHT_ONLY = args.night_only
    dates = list(args.dates or [])
    if args.dates_file:
        dates += [l.strip() for l in open(args.dates_file) if l.strip()]
    if not dates:
        ap.error("give --dates or --dates-file")
    for d in (C.MET_DIR, C.RUNS_DIR, RB.PREDICTED_DIR):
        os.makedirs(d, exist_ok=True)
    # Downloads dominate run time and appear limited by per-request overhead, so run several dates at once,
    # each in its own process (separate work dirs and run dirs per date; nothing is shared but the log).
    if args.workers > 1 and len(dates) > 1:
        import subprocess
        chunks = [dates[i::args.workers] for i in range(args.workers)]
        extra = (["--force"] if args.force else []) + (["--keep-met"] if args.keep_met else []) + \
                (["--night-only"] if args.night_only else [])
        procs = [subprocess.Popen([sys.executable, "-u", "-m", "validation.run_backtest_clipped", "--workers", "1",
                                   "--dates", *c, *extra], cwd=os.path.dirname(C.VALIDATION_DIR)) for c in chunks if c]
        print(f"Running {len(dates)} date(s) in {len(procs)} parallel workers", flush=True)
        codes = [p.wait() for p in procs]
        print(f"all workers finished (exit codes {codes})", flush=True)
        sys.exit(max(codes))
    print(f"Running {len(dates)} date(s) with clipped met -> {RB.PREDICTED_DIR}")
    bt = ClippedBacktest()
    bt.keep_met = args.keep_met
    failed = []
    for i, d in enumerate(dates, 1):
        t0 = time.time()
        print(f"\n===== [{i}/{len(dates)}] {d} =====", flush=True)
        try:
            bt.run_date(d, force=args.force)
        except Exception as e:
            print(f"[{d}] ERROR: {e}", flush=True)
            failed.append(d)
        print(f"[{d}] total {(time.time() - t0) / 60:.1f} min", flush=True)
    # PLUME_PLAN.md: a date that fails gets one retry; one that fails again is excluded and listed.
    for d in failed:
        print(f"\n===== retry {d} =====", flush=True)
        try:
            bt.run_date(d, force=args.force)
        except Exception as e:
            print(f"[{d}] ERROR on retry, excluded: {e}", flush=True)


if __name__ == "__main__":
    main()
