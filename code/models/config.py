"""
Configuration for the HYSPLIT back-test validation (Calvert City, 2025 H1).

Single source of truth for: monitor sites, target species, unit-conversion constants,
per-species HYSPLIT deposition parameters, the dispersion grid, and file paths.

Design notes
------------
* We reuse ``calvert_plume_engine_one_day.CalvertCityPlumeEngine`` for the tested pieces:
  Central-time day windowing (``set_active_date`` / ``sim_start_utc``), HRRR download,
  GRIB->ARL conversion, TRI parsing, and facility matching. This file only adds what the
  validation needs on top.
* Monitor values are 24-hour TO-15 canister means in **ppbC** (parts-per-billion Carbon).
  Convert to a volume mixing ratio with ``ppb = ppbC / n_carbons`` then to mass with
  ``ug_m3 = ppb * MW / MOLAR_VOLUME``.
* HYSPLIT emits g/m^3; multiply by 1e6 for ug/m^3. Comparison is done in ug/m^3.
* Emission rates come straight from TRI air totals (steady-state): the engine already
  computes ``stack_g_hr`` / ``fugitive_g_hr`` as ``lbs_yr * 453.59237 / 8760``.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(VALIDATION_DIR)

MONITOR_XLSX = os.path.join(REPO_DIR, "CalvertDailyVOCS_DateEnding6.30.2025.xlsx")
TRI_CSV = os.path.join(REPO_DIR, "Marshall County Facility Release.csv")

MET_DIR = os.path.join(VALIDATION_DIR, "met")        # ARL cache, one file per run date
RUNS_DIR = os.path.join(VALIDATION_DIR, "runs")      # runs/<date>/<facility_slug>/
RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

CON2STN_PATH = "/path/to/hysplit/exec/con2stn"

# ---------------------------------------------------------------------------
# Monitor sites (from the AQS daily-summary xlsx). Coords are WGS84.
# `aqs_site` is the Site Num as it appears in the workbook.
# ---------------------------------------------------------------------------
MONITOR_SITES = {
    "elementary":    {"name": "Calvert City Elementary", "aqs_site": "18", "lat": 37.026757, "lon": -88.343757},
    "johnson_riley": {"name": "Johnson-Riley Road",      "aqs_site": "20", "lat": 37.041202, "lon": -88.351879},
    "lwd":           {"name": "LWD",                      "aqs_site": "21", "lat": 37.047914, "lon": -88.338323},
}

# ---------------------------------------------------------------------------
# Target species. Each entry maps the three identities that must line up:
#   monitor_name : exact "Parameter Name" string in the AQS xlsx
#   tri_name     : exact "chemical" string (upper-cased) in the TRI CSV
#   hysplit_tag  : <=4 char HYSPLIT pollutant id (unique per species)
#   n_carbons    : carbon atoms, for ppbC -> ppb conversion
#   mol_weight   : g/mol, for ppb -> ug/m^3 conversion AND HYSPLIT dry-dep block
#   dry_vd_ms, henry_M_atm, halflife_days : HYSPLIT gas deposition params
#
# Five species reuse the engine's CHEMICAL_DEPOSITION values; chloroform and
# trichloroethylene are added here (not present in the engine). VOCs are treated as
# negligible depositors (wet params = 0), consistent with the engine's vetted approach.
# Deposition mainly matters as a small plume-mass sink; these VOCs travel largely intact.
#
# henry_M_atm corrected 2026-10-26 to the Sander (2023) compilation medians (PLUME_AUDIT_2026-10-24 item 6):
# vinyl chloride 0.069 -> 0.042, EDC 0.11 -> 0.83, butadiene 0.058 -> 0.014, PERC 0.027 -> 0.058,
# TCE 0.0098 -> 0.098 (benzene and chloroform were right). The value only feeds HYSPLIT's gas wet removal, and
# the HRRR analysis met (fxx=0) carries no precipitation (TPP1 = 0 in every hour of every kept met file, even on
# 2024-03-25, 0.41 in of rain in ERA5), so wet removal never runs: re-running 2024-03-25 with the corrected values
# reproduced all 21 production predictions exactly. Runs before and after this change are therefore identical.
# ---------------------------------------------------------------------------
TARGET_SPECIES = {
    "vinyl_chloride": {
        "monitor_name": "Vinyl chloride", "tri_name": "VINYL CHLORIDE", "hysplit_tag": "VCLR",
        "n_carbons": 2, "mol_weight": 62.5, "dry_vd_ms": 0.003, "henry_M_atm": 0.042,
        "halflife_days": 1.5,
    },
    "benzene": {
        "monitor_name": "Benzene", "tri_name": "BENZENE", "hysplit_tag": "BENZ",
        "n_carbons": 6, "mol_weight": 78.1, "dry_vd_ms": 0.003, "henry_M_atm": 0.18,
        "halflife_days": 9.7,
    },
    "edc": {  # 1,2-dichloroethane; monitor calls it "Ethylene dichloride"
        "monitor_name": "Ethylene dichloride", "tri_name": "1,2-DICHLOROETHANE", "hysplit_tag": "EDC",
        "n_carbons": 2, "mol_weight": 99.0, "dry_vd_ms": 0.003, "henry_M_atm": 0.83,
        "halflife_days": 83.0,
    },
    "butadiene": {
        "monitor_name": "1,3-Butadiene", "tri_name": "1,3-BUTADIENE", "hysplit_tag": "BUTA",
        "n_carbons": 4, "mol_weight": 54.1, "dry_vd_ms": 0.003, "henry_M_atm": 0.014,
        "halflife_days": 1.4,
    },
    "chloroform": {  # added here: not in engine CHEMICAL_DEPOSITION
        "monitor_name": "Chloroform", "tri_name": "CHLOROFORM", "hysplit_tag": "CHCL",
        "n_carbons": 1, "mol_weight": 119.4, "dry_vd_ms": 0.003, "henry_M_atm": 0.25,
        "halflife_days": 150.0,  # OH-limited atmospheric lifetime ~5 months
    },
    "tetrachloroethylene": {
        "monitor_name": "Tetrachloroethylene", "tri_name": "TETRACHLOROETHYLENE", "hysplit_tag": "PERC",
        "n_carbons": 2, "mol_weight": 165.8, "dry_vd_ms": 0.003, "henry_M_atm": 0.058,
        "halflife_days": 117.0,
    },
    "trichloroethylene": {  # added here: not in engine CHEMICAL_DEPOSITION
        "monitor_name": "Trichloroethylene", "tri_name": "TRICHLOROETHYLENE", "hysplit_tag": "TCE",
        "n_carbons": 2, "mol_weight": 131.4, "dry_vd_ms": 0.004, "henry_M_atm": 0.098,
        "halflife_days": 7.0,  # OH-reactive, short-lived
    },
}

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
# Ideal-gas molar volume (L/mol) at 25 C, 1 atm. AQS reports at these reference conditions.
MOLAR_VOLUME_L = 24.45
GM3_TO_UGM3 = 1.0e6

# ---------------------------------------------------------------------------
# Dispersion grid for point extraction.
# Smaller than the dashboard's (span 4.0 / spacing 0.03) because we only need
# concentrations at the monitors, which sit within a few km of the sources. Tuned in the
# pilot for runtime vs. resolution. con2stn interpolates to the exact monitor coordinate.
# ---------------------------------------------------------------------------
GRID_CENTER = (37.0317, -88.3542)   # MAP_CENTER, matches the engine
GRID_SPAN = 1.0                     # degrees (~110 km) - covers plume reach to the monitors
GRID_SPACING = 0.01                 # degrees (~1.1 km) - resolves the 3 nearby monitors
RUN_HOURS = 23                      # local-day window, matches the engine's dep runs
# HYSPLIT particle cap. numpar=-500 releases ~500 particles/h per species, so a 7-species facility
# (Westlake Vinyls) needs ~80,000 over a 23-h day. The poster-era runs used 50,000, which every
# Westlake run hit (emspnt "exceeding puff limit"), dropping its late-day emissions.
NUMPAR = -500
MAXPAR = 150000
# Per-run limit for hycs_std. 600 s was too short for Westlake Vinyls (7 species) once MAXPAR was raised.
HYSPLIT_TIMEOUT_S = 3600
AIR_LEVEL_INDEX = 2                 # con2stn -z: cdump level 1 = 0 m (dep), level 2 = 10 m (air)

# ---------------------------------------------------------------------------
# Event-definition thresholds (Stage 4). Primary = 75th percentile; sweep for robustness.
# ---------------------------------------------------------------------------
PRIMARY_PERCENTILE = 75
SWEEP_PERCENTILES = [50, 60, 70, 75, 80, 90]

# 2025 reporting-year TRI is not public yet (EPA posts preliminary ~late July 2026);
# we use 2024 TRI. Re-score with 2025 preliminary data as a sensitivity check when available.
TRI_VINTAGE = 2024
