# Data dictionary

**Conventions.** Dates are local calendar days (America/Chicago at Calvert, America/New_York in Pittsburgh).
Measurements are 24-hour canister samples in ppbC, as AQS reports them; 0 = non-detect. To convert:
ug/m3 = ppbC / carbons x molecular weight / 24.45 (vinyl chloride 2, 62.5; benzene 6, 78.1; 1,3-butadiene 4, 54.1;
EDC 2, 99.0). Monitors: Calvert City Elementary 21-157-0018, Johnson-Riley Road 21-157-0020, LWD 21-157-0021 (inside
the complex); locations and the older monitors are in `monitors.csv`.

## 02_analysis_datasets

**`calvert_monitor_days_2023_2025.csv`**: one row per monitor-day at the three current monitors (508 rows).
- `poster_window_jan_jun_2025`: the poster's own days (excluded from the primary tests).
- `*_ppbC`, `n_species_in_total_voc`: measurements (24-hour canisters, ppbC, 0 = non-detect). `total_voc_ppbC` sums the
  56 species reported in every year 2023-2025, canisters averaged; `total_voc_all_species_ppbC` keeps the original sum
  of every ppbC species except "Total"/"Sum" rows (DEVIATIONS 38).
- ORI, 0 to 100:
  - `ori_deployed_corrected_pct`: the headline ORI (deployed model, its usual IFS inputs, 2024 BLH gap filled);
  - `ori_deployed_as_run_pct`: the pre-specified first run (no BLH for Jan to Jun 2024);
  - `ori_refit_era5_pct`: the refit on ERA5;
  - `ori_deployed_on_era5_pct`: the deployed model on ERA5.
- `hours_downwind_fraction`: share of the day's hours with ERA5 10 m wind within 30 degrees of the complex-to-monitor
  bearing (complex centre 37.0486, -88.3480).
- `combo_ori_plus_downwind_rank`: mean of the within-monitor percentile ranks of the corrected ORI and hours downwind,
  on the 230 scored monitor-days only.
- `calvert_fitted_*_out_of_fold`: ridge model trained on the monitors (11 ERA5 inputs), each year predicted by a fit on
  the other two; output is a predicted within-monitor rank. Scored rows only.
- `trapping_composite_rank`: mean within-monitor rank of the 15 trapping indices (scored rows only).
- `index: ...`: the report-free trapping and stagnation indices themselves (ERA5).
- `hysplit_*_ugm3`: HYSPLIT predicted daily mean at the monitor (TRI 2024 emissions), on the 189 run dates
  (`hysplit_run`); `sum7` adds all seven species.
- `aqi_*`: daily AQI comparators: observed (Paducah, Smithland ozone) and modelled (CAMS). Scored rows only.

**`older_monitor_days_2011_2022.csv`**: the same layout for the seven monitors with samples before 2023 (2,001 rows).
- `set_aside_in_plant`: LWD and TVA Substation, not counted as community monitors.
- `total_voc_ppbC` is blank where fewer than 50 species were reported (before mid-2012).
- `calvert_fitted_*`: the model trained on all 2023 to 2025 data, applied unchanged.

**`monitors.csv`**: AQS metadata for all monitors (name, address, land use, coordinates, distance and bearing from the
complex, nearest facility, sample counts, detection rates, in-plant flag reason).

**`hysplit_date_sets.csv`**: `date`, `set` = first78 (half-sample), second78, new156 (headline), poster33 (poster dates
re-run with the fixed cap), all189.

**`pittsburgh_citywide_daily.csv`**: Pittsburgh daily complaints (all 107 zip codes) and ERA5 weather averaged over the
zip codes, from the refit's training panel (Figs 1 and S1); BLH in true feet.
**`ori_oof_deciles.csv`**: by city, the refit's out-of-fold ORI deciles behind Fig 2 (event %, mean reports per zip-day).

**`calvert_new_dates_with_baseline_indices.csv`**: the 271 monitor-days of the 6 October exploratory comparison (S5).
**`poster_era_57_monitor_days.csv`**: the poster's 57 monitor-days with poster-era scores (cap-50k plume); history only.

## 01_raw_data

- `aqs_calvert_monitors/`, `aqs_older_monitors_2011_2022/`: EPA AQS daily_HAPS and daily_VOCS rows for the monitors
  (EPA's own columns; `Arithmetic Mean` is the daily value). Kentucky DAQ workbook for Jan to Jun 2025.
- `hysplit_predictions/hysplit_daily_predictions_189_dates.csv`: `date`, `monitor`, `species`, `pred_mean_ugm3`
  (daily mean), `pred_peak_ugm3` (highest hour), `pred_contact_hours` (hours above zero), `n_hours`, `poster_date`.
  `SUPERSEDED_..._cap50k.csv`: poster-era runs; do not analyze.
- `weather/`: ERA5 daily inputs for the Calvert cell (2011-2026) and the IFS inputs at each monitor (2023-2025).
- `emissions/`: TRI 2024 releases used by HYSPLIT; TRI 2022 to 2024 Kentucky files for the sensitivities.
- `smell_mycity/`: raw reports (locations deliberately perturbed by Smell MyCity). `aqi/`: daily AQI inputs.
