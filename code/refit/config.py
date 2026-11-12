"""Fixed settings for the ORI training-panel rebuild (see training/build_training_panel.py)."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "training_panel_v2")
RAW_DIR = os.path.join(OUT_DIR, "raw")

# Local calendar days kept in the panel. 2026-06-06 is left out: the report exports end that afternoon.
START_DATE = "2018-01-01"
END_DATE = "2026-06-05"
# Louisville has almost no reports before this date (20 in all of 2018); rows before it are flagged.
LOU_SPARSE_BEFORE = "2019-03-01"

# Emission sources, from each city's Dual_Model_Proximity_Analysis.py (EMISSION_SOURCES), with three locations
# corrected 2026-10-26 (checklist C1), each checked on satellite imagery:
#   Clairton: was 40.2974, -79.8809 (Clairton's residential grid south of the plant); now the coke-battery rows.
#   Irvin: was 40.3644, -79.8944 (about 3 km north of the mill); now a point placed on the facility buildings on the mill buildings.
#   JBS: was 38.2588, -85.7275 (about 400 m north, by a parking lot); now a point placed on the facility buildings, which matches the
#   plant's TRI coordinates (SWIFT & CO, 1200 Story Ave) to about 80 m.
PGH_SOURCES = [
    {"name": "Clairton_Coke_Works", "lat": 40.3060, "lon": -79.8770},
    {"name": "Edgar_Thomson_Works", "lat": 40.3922, "lon": -79.8550},
    {"name": "Irvin_Works", "lat": 40.337213, "lon": -79.900363},
]
LOU_SOURCES = [
    {"name": "Rubbertown", "lat": 38.2195, "lon": -85.8450},
    {"name": "JBS_Swift_Butchertown", "lat": 38.255157, "lon": -85.728036},
]

# The old Pittsburgh panel's 61 zips (the top 61 by report count; the cut tied at 188 reports, and there is no
# documented reason for it). The rebuild keeps all zips with coordinates and flags these. Louisville's old
# panel had all 36 of its zips.
PGH_ORIGINAL_ZIPS = frozenset({
    "15025", "15034", "15037", "15044", "15045", "15065", "15101", "15102", "15104", "15106", "15108",
    "15110", "15112", "15116", "15120", "15122", "15129", "15131", "15132", "15133", "15136", "15137",
    "15143", "15146", "15147", "15201", "15202", "15203", "15205", "15206", "15207", "15208", "15209",
    "15210", "15211", "15212", "15213", "15214", "15215", "15216", "15217", "15218", "15219", "15220",
    "15221", "15222", "15224", "15226", "15227", "15228", "15229", "15232", "15233", "15234", "15235",
    "15236", "15237", "15238", "15239", "15241", "15243"
})

CITIES = {
    "pgh": {"name": "Pittsburgh", "reports": os.path.join(REPO, "Pittsburgh Data", "smell_reports_pittsburgh.csv"),
            "tz": "America/New_York", "sources": PGH_SOURCES, "fetch_years": range(2018, 2027)},
    "lou": {"name": "Louisville", "reports": os.path.join(REPO, "Louisville Data", "smell_reports.csv"),
            "tz": "America/Kentucky/Louisville", "sources": LOU_SOURCES, "fetch_years": range(2018, 2027)},
}

# Last date requested from either weather source (UTC); one day past END_DATE covers the last local day.
FETCH_END = "2026-06-07"
