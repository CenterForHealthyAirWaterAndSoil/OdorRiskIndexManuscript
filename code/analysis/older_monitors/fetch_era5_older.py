"""Fetch the Calvert ERA5 cell (37.00, -88.25) for 2010-12-30..2018-04-01 from the same two sources as the training panel
and refit_inputs (Copernicus ERA5 single-levels time series for every variable except rain; Open-Meteo models=era5 for
rain), using training/era5_cells.py's own fetchers with the cache directories and request dates redirected to
older_monitors/raw/ (the panel's own caches are not touched)."""
import os, sys, time
REPO = "/path/to/weather-varaible-analysis"; sys.path.insert(0, REPO)
from training import config, era5_cells as E
HERE = os.path.dirname(os.path.abspath(__file__))
E.CDS_DIR = os.path.join(HERE, "raw", "cds_era5"); E.OM_DIR = os.path.join(HERE, "raw", "open_meteo_era5")
config.START_DATE, config.FETCH_END = "2010-12-30", "2018-04-01"
GLAT, GLON = 37.0, -88.25
if __name__ == "__main__":
    what = sys.argv[1]
    if what == "cds":
        E.fetch_cds_cell(GLAT, GLON); print("cds done", flush=True)
    else:
        for y in range(2010, 2019):
            if E.fetch_open_meteo_cell(GLAT, GLON, y): print("om", y, flush=True); time.sleep(8)
