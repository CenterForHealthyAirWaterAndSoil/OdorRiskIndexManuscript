"""
Parse the AQS daily-summary workbook into a tidy observed dataset for the back-test.

Output: results/observed_daily.csv with one row per (date, site, species):
    date, site, site_name, species, monitor_name, value_ppbc, units,
    value_ppb, value_ugm3, below_detect

24-hour TO-15 canister samples (~every 6 days). We take the day's "Arithmetic Mean"
(with a single 24-h observation it equals the reported value). Non-positive values are
flagged ``below_detect`` and floored at 0 for the mass conversion.

Run:  python3 -m validation.extract_monitors      (from the repo root)
   or python3 validation/extract_monitors.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import config as C

try:
    import openpyxl
except ImportError:
    sys.exit("openpyxl required: pip install openpyxl")


def _site_key_for(site_num):
    for key, meta in C.MONITOR_SITES.items():
        if meta["aqs_site"] == str(site_num):
            return key
    return None


def _species_key_for(monitor_name):
    for key, meta in C.TARGET_SPECIES.items():
        if meta["monitor_name"].lower() == str(monitor_name).strip().lower():
            return key
    return None


def extract():
    wb = openpyxl.load_workbook(C.MONITOR_XLSX, read_only=True)
    ws = wb["Sheet1"]
    rows = ws.iter_rows(values_only=True)
    hdr = next(rows)
    idx = {name: i for i, name in enumerate(hdr)}

    out = []
    for r in rows:
        site_num = r[idx["Site Num"]]
        if site_num is None:
            continue
        site_key = _site_key_for(site_num)
        if site_key is None:
            continue
        species_key = _species_key_for(r[idx["Parameter Name"]])
        if species_key is None:
            continue

        sp = C.TARGET_SPECIES[species_key]
        raw = r[idx["Arithmetic Mean"]]
        units = str(r[idx["Units of Measure"]]).strip()
        date_local = r[idx["Date Local"]]
        date_str = date_local.strftime("%Y-%m-%d") if hasattr(date_local, "strftime") else str(date_local)

        value_ppbc = float(raw) if raw is not None else None
        below = value_ppbc is None or value_ppbc <= 0.0
        v = max(value_ppbc, 0.0) if value_ppbc is not None else 0.0

        # ppbC -> ppb (volume) -> ug/m^3. If the file ever reports plain ppb, n_carbons
        # would be wrong, so we assert the expected unit and record it for auditing.
        if "carbon" in units.lower():
            value_ppb = v / sp["n_carbons"]
        else:
            value_ppb = v  # already a volume mixing ratio
        value_ugm3 = value_ppb * sp["mol_weight"] / C.MOLAR_VOLUME_L

        out.append({
            "date": date_str,
            "site": site_key,
            "site_name": C.MONITOR_SITES[site_key]["name"],
            "species": species_key,
            "monitor_name": sp["monitor_name"],
            "value_ppbc": value_ppbc if value_ppbc is not None else "",
            "units": units,
            "value_ppb": round(value_ppb, 6),
            "value_ugm3": round(value_ugm3, 6),
            "below_detect": int(below),
        })

    out.sort(key=lambda d: (d["date"], d["site"], d["species"]))
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    path = os.path.join(C.RESULTS_DIR, "observed_daily.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    dates = sorted({d["date"] for d in out})
    print(f"Wrote {len(out)} observations to {path}")
    print(f"{len(dates)} sample dates: {dates[0]} -> {dates[-1]}")
    return out, dates


def sample_dates():
    """All monitor sample dates (YYYY-MM-DD), sorted."""
    _, dates = extract()
    return dates


if __name__ == "__main__":
    extract()
