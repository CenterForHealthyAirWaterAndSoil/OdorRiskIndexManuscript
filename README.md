# Odor Risk Index manuscript code

Code for a study that asks whether a weather model trained on citizen odor reports in Pittsburgh, Pennsylvania and
Louisville, Kentucky can rank days by measured air toxics in Calvert City, Kentucky, a small town next to a chemical
complex. The model (the Odor Risk Index, ORI) is applied in Calvert City unchanged and compared with EPA canister
samples of vinyl chloride, benzene and other volatile organic compounds, with NOAA's HYSPLIT dispersion model, with a
count of hours downwind of the complex, with standard stagnation and inversion measures, and with the daily Air
Quality Index.

This repository holds code only. The scripts are the ones that produced the numbers and figures in the manuscript,
as they were run. Folder paths that pointed at one computer have been replaced with `/path/to/...`, so set those
before running anything.

## Layout

| Folder | What it does |
|---|---|
| `code/refit/` | Builds the zip code by day training panel from Smell MyCity reports and ERA5 weather, and fits the ORI (`fit_refit.py`; out-of-sample checks, the alert line, calibration). `fit_ifs_comparison.py` and `ifs_panel.py` repeat the fit on ECMWF forecast-model weather. |
| `code/models/` | `odor_forecast_core.py` applies the fitted ORI. `calvert_plume_engine*.py`, `run_backtest*.py`, `config.py` and `extract_monitors.py` run HYSPLIT for each sampling date and read predictions at the monitors. `weather_history.py` is the single source of past weather. |
| `code/analysis/` | Scores at the Calvert City monitors. `score_all_dates_refit.py` gives the main results on the 141 sampling dates (ORI, HYSPLIT, hours downwind, the combined score, the model fitted at Calvert City). `score_aqi.py` is the AQI comparison, `fair_baselines.py` and `paper_review_checks.py` the stagnation and inversion measures (Table S2), `species_sweep_refit.py` every measured chemical, `lead_sensitivity.py` forecast lead time, `drop_louisville_check.py` a training-city sensitivity check. `older_monitors/` builds and scores the 2011 to 2022 samples. `hysplit/` scores HYSPLIT and its emission-year sensitivity runs. `run_extended.py`, `score_all_dates.py` and `score_refit.py` score the earlier version of the ORI reported in Table S1 and the tests specified in advance. |
| `code/poster_figures/` | Main-text figures (weather and odor reports, calibration, transfer to Calvert City, ORI and HYSPLIT by pollutant, study area, appendix figures). `run_all.sh` draws them all. |
| `code/figures/` | Supplementary figures and their captions. |
| `code/make_tables.py` | Result tables. |
| `docs/` | `HYSPLIT_SETTINGS.md` (how the dispersion runs were set up) and `DATA_DICTIONARY.md` (columns of the analysis datasets). |

## Data

No data are stored here. All inputs are public:

- Smell MyCity and Smell Pittsburgh odor reports, CMU CREATE Lab (https://smellpgh.org, https://smellmycity.org)
- ERA5 reanalysis, Copernicus Climate Data Store (https://cds.climate.copernicus.eu), and Open-Meteo's ERA5 and ECMWF IFS archives (https://open-meteo.com)
- HRRR 3 km analyses, NOAA (https://rapidrefresh.noaa.gov/hrrr/)
- VOC canister samples, EPA Air Quality System pre-generated daily files (https://aqs.epa.gov/aqsweb/airdata/download_files.html)
- Toxics Release Inventory basic data files, EPA (https://www.epa.gov/toxics-release-inventory-tri-program)
- Daily AQI by county, EPA AirData
- HYSPLIT 5.4.2, NOAA Air Resources Laboratory (https://www.ready.noaa.gov/HYSPLIT.php)

## Software

Python 3.14 with the packages in `requirements.txt`. The ERA5 downloads need a free Copernicus account for `cdsapi`.
