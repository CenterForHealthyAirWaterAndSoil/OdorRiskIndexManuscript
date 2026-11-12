"""EXPLORATORY baseline inputs: ERA5 pressure-level temperature (925, 850 hPa) and 500 hPa wind at the Calvert grid point."""
import cdsapi, os
from concurrent.futures import ThreadPoolExecutor
PT = [37.0, -88.25, 37.0, -88.25]
jobs = []
QUARTERS = {"q1": ["01", "02", "03"], "q2": ["04", "05", "06"], "q3": ["07", "08", "09"], "q4": ["10", "11", "12"]}
for y in ["2023", "2024", "2025"]:
    for q, months in QUARTERS.items():   # quarterly pieces stay under the CDS per-request cost limit
        jobs.append((f"era5_pl_temp_{y}{q}.nc", {"variable": ["temperature"], "pressure_level": ["850", "925"], "year": [y], "month": months}))
        jobs.append((f"era5_pl_wind500_{y}{q}.nc", {"variable": ["u_component_of_wind", "v_component_of_wind"], "pressure_level": ["500"], "year": [y], "month": months}))
def get(job):
    out, req = job
    if os.path.exists(out): return out + " cached"
    req.update({"product_type": ["reanalysis"], "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)], "data_format": "netcdf", "download_format": "unarchived", "area": PT})
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-pressure-levels", req, out)
    return out
with ThreadPoolExecutor(6) as ex:
    for r in ex.map(get, jobs): print("done", r, flush=True)
