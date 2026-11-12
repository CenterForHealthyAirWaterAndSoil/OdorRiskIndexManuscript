# Supplementary figure captions (11 November 2026)

Eight supplementary figures (S1 to S8). S1 and S2 are drawn in blue like the main figures; S3 to S8 are still black and gray. Test statistics are given in these
captions, not on the figures. Each figure has a PNG (300 dpi), a PDF and a `_data.csv` of exactly what is plotted.
Within-monitor Spearman rho is computed at each community monitor separately. Unless a caption says otherwise, the
Calvert sample is the 156 sampling dates in 2023-2025 not used on the poster (271 monitor-days at the two community
monitors). Since 11 November 2026 the ORI is the 2026 refit (DEVIATIONS 40); S8 still shows the earlier version.


**Figure S1. All eight weather drivers (Pittsburgh).** Average daily Smell MyCity complaints, binned by each weather
variable. Panels a to d repeat Figure 2.
- N = 3,078 days, 1 January 2018 to 5 June 2026; reports from all 107 Pittsburgh zip codes; daily ERA5 weather averaged
  over the zip codes (the refit's training data).
- Each point is the mean over the days in one bin. The days are sorted by the weather variable and split into 20 bins
  with the same number of days in each (fewer for rain, where many days share the value zero), so every day is included.
- Spearman rho across all 3,078 individual days: (a) wind speed -0.30, (b) boundary layer height -0.42, (c) diurnal
  temperature range +0.53, (d) temperature +0.28, (e) solar radiation +0.23, (f) relative humidity -0.14,
  (g) precipitation -0.10, (h) surface pressure +0.15; all p < 0.001.

**Figure S2. HYSPLIT predictions against measurements, by monitor.** Measured against HYSPLIT-predicted daily vinyl
chloride (top) and benzene (bottom) at each community monitor, on the 141 sampling dates (282 monitor-days). Both axes
are logarithmic and cover the same range, so the solid line marks where the prediction equals the measurement; points
below it are predicted too high. Dotted lines mark a factor of 10 either side.
- Filled points are detections. Open points are non-detects, plotted at half the smallest detected value in that panel.
  Triangles at the left edge are days with a prediction below 0.01 µg/m³.
- Vinyl chloride: Calvert City Elementary rho 0.49, Johnson-Riley Road 0.68 (n = 141 each). Benzene: 0.27 and 0.30.
- Among detected samples, the prediction was within a factor of 10 of the measurement on 74% and 83% of days for vinyl
  chloride (Elementary, Johnson-Riley) and 67% and 67% for benzene.
- Measured values are converted from ppbC to µg/m³.

**Figure S3. Estimates by year and monitor.** Within-monitor Spearman rho with 95% CIs (1,000 bootstrap resamples of
monitor-days) for the ORI (2026 refit; black, top row) and HYSPLIT (gray, bottom row), for vinyl chloride and benzene.
- Both models are scored on the same 271 monitor-days.
- 2025 covers July to December only, because January to June 2025 were the poster's dates.

**Figure S4. Every measured chemical against the ORI (exploratory).** Within-monitor Spearman rho between the ORI and each
of the 25 chemicals measured at the two community monitors, on the test dates on which both were sampled (refit ORI).
- 230 monitor-days for most chemicals. Five were not reported on every date: 1,2,4-trimethylbenzene (78 monitor-days),
  acetonitrile (190), styrene (198), methyl isobutyl ketone (218) and 1,2,4-trichlorobenzene (226).
- Filled black: q < 0.05 (Benjamini-Hochberg across the 25 chemicals). Open: q ≥ 0.05. Gray squares: long-lived
  background gases used as negative controls (three CFCs, carbon tetrachloride, dichloromethane).
- 14 of the 20 other chemicals have q < 0.05. Thirteen are positive, led by 1,2,4-trimethylbenzene 0.64, m/p-xylene
  0.47, ethylene dichloride 0.47, 1,3-butadiene 0.46, benzene 0.44 and vinyl chloride 0.43. Chloromethane is negative
  (-0.20).
- No negative control has q < 0.05; their rho values run from -0.01 to +0.11.

**Figure S5. The ORI against standard trapping indices and locally trained models (exploratory).** Within-monitor Spearman
rho (mean of the two monitors) with 95% CIs from 1,000 date-cluster bootstrap resamples, on the 271 monitor-days. Rows
are ordered by the vinyl chloride score.
- Black: the ORI, transferred with no local data. Gray circles: standard trapping and wind indices, which also need no
  local data. Open diamonds: measures that use the Calvert monitor history.
- Vinyl chloride: hours downwind of the complex (0.70), hours downwind in calm wind (0.58) and the ridge model trained
  on Calvert monitors (0.48) rank above the ORI (0.44, CI 0.30 to 0.56).
- Benzene: the ORI ranks first (0.41, CI 0.27 to 0.53); next is the composite of all indices (0.40).
- Total VOC: hours downwind 0.45, the ridge model 0.43, hours downwind in calm wind 0.38, the ORI 0.37 (CI 0.26 to 0.50).

**Figure S6. HYSPLIT and the difference between the two monitors.** For each of the 115 dates on which both monitors
were sampled, HYSPLIT's predicted Johnson-Riley minus Elementary difference against the measured difference.
- Spearman rho: vinyl chloride 0.65, benzene 0.00.

**Figure S7. HYSPLIT magnitude by species.** Mean HYSPLIT prediction divided by mean measured concentration, by species
and monitor (log scale; 1 means predicted equals measured). Non-detects count as 0. Open circles: Calvert City
Elementary. Gray squares: Johnson-Riley Road.
- Elementary and Johnson-Riley, in that order: vinyl chloride 5.8 and 1.4; benzene 2.1 and 5.4; ethylene dichloride
  0.64 and 0.73; 1,3-butadiene 2.6 and 6.1; chloroform 0.11 and 0.14; tetrachloroethylene 0.65 and 0.08.
- Chloroform and tetrachloroethylene were detected on 1 to 10% of monitor-days, so those ratios rest on few detections.
  Trichloroethylene was never detected and is not shown.
- Over all 189 HYSPLIT dates, the ratios are vinyl chloride 6.1 and 1.4 and ethylene dichloride 0.66 and 0.78
  (HYSPLIT_SETTINGS.md).

**Figure S8. Forecast lead time (earlier version of the ORI).** Within-monitor Spearman rho between archived real
forecasts from the earlier ORI, issued 0 to 5 days ahead, and measured vinyl chloride (black, solid) and benzene (gray,
dashed). Not redone for the refit: archived forecasts exist for it only on the day of issue, where it ranked days as
well as on reanalysis (55 dates from September 2024: vinyl chloride 0.52 vs 0.47, benzene 0.50 vs 0.49).
- 64 monitor-days from 3 September 2024 onward. Forecasts are from Open-Meteo's archives; at leads of 1 day or more,
  boundary layer height comes from the lead-0 run.
- Vinyl chloride: 0.44, 0.45, 0.39 and 0.41 at 0, 1, 3 and 5 days ahead.
- Benzene: 0.68, 0.58, 0.67 and 0.72.
- All p ≤ 0.015.
