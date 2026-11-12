# Figure captions (11 November 2026)

Seven figures for the main text (1 to 7, numbered in the order the paper cites them) and two for the appendix (A1, A2), drawn in blue and white (the ORI is blue; HYSPLIT and comparison bars are white with a dark outline) so they read the
same with any color vision and in grayscale print. Test statistics are given in these captions, not on the figures. Each
figure has a PNG (300 dpi), a PDF and, where it plots numbers, a `_data.csv` of exactly what is plotted. Since
11 November 2026 the ORI in every figure is the 2026 refit, trained on one ERA5 weather source (DEVIATIONS 40); the
earlier version's figures are in the 2026-11-03 package.


**Figure 1. Study area map.**
- (a) The 32 census tracts in Marshall (white), McCracken (light blue) and Livingston (blue) counties. The box marks the area shown in (b).
- (b) The five facilities modeled in HYSPLIT (black squares) and the EPA air monitors:
  - Johnson-Riley Road, 0.55 mi from the complex (blue circle);
  - Calvert City Elementary, 1.5 mi (blue circle);
  - LWD, inside the complex and not used in the community-monitor analyses (gray circle).
- Hollow circles are the earlier VOC monitors used in the 2011-2022 test. Its five community monitors are Atmos Energy,
  Lazy Daz, Bloodworth, and the earlier years at Elementary and Johnson-Riley. TVA Substation, on the plant road, is
  set aside with LWD.

**Figure 2. Weather conditions that trap odor.** Average daily Smell MyCity complaints in Pittsburgh, binned by wind
speed, boundary layer height, diurnal temperature range and temperature.
- N = 3,078 days, 1 January 2018 to 5 June 2026; reports from all 107 Pittsburgh zip codes; daily ERA5 weather averaged
  over the zip codes (the refit's training data).
- Each point is the mean over about 154 days: the days are sorted by the weather variable and split into 20 bins with
  the same number of days in each, so every day is included. Points sit at the mean value of their bin.
- Spearman rho across all 3,078 individual days: wind speed -0.30, boundary layer height -0.42, diurnal temperature
  range +0.53, temperature +0.28; all p < 0.001.

**Figure 3. Calibration in the training cities.** Mean reports per zip-code day (white bars, left axis) and the share of
zip-code days that were odor events (blue line, right axis), by decile of the out-of-sample ORI, in Pittsburgh and
Louisville.
- Out of sample: each calendar month is predicted by a model fit without that month (day of week at its average effect).
- Pittsburgh: 329,346 zip-code days; event days rise from 2.1% in the lowest decile to 31.2% in the highest (12.7% overall).
- Louisville: 95,544 zip-code days; 0.5% to 23.7% (5.4% overall).
- Within the same zip code and year, the model ranks event days above other days 71% of the time in Pittsburgh and 64%
  in Louisville; a seasonal curve alone reaches 51% and 61%.

**Figure 4. Transfer to Calvert City.** Mean measured total VOC, vinyl chloride and benzene on lower-risk (white) vs
higher-risk (blue) forecast days at the two community monitors.
- Sample: 141 sampling days in 2023-2025 on which both monitors have results (282 monitor-days). Days are split at each
  monitor's own median forecast.
- Ratio (mean of the higher half over mean of the lower half): total VOC 1.3x, vinyl chloride 2.9x, benzene 1.4x, all
  p < 0.001 by date permutation.
- At each monitor: Calvert City Elementary 1.1x, 2.8x and 1.4x; Johnson-Riley Road 1.5x, 2.9x and 1.4x (total VOC,
  vinyl chloride and benzene).

**Figure 5. Which model for which pollutant.** Within-monitor Spearman rho with the measured level on the same
282 monitor-days (141 dates, 2023-2025). ORI is blue and HYSPLIT white; in (b), hours downwind is white with diagonal lines and ORI +
hours downwind is blue with diagonal lines. Verdicts come from date-bootstrap 95% CIs of the paired difference. This comparison was
specified after the individual results were known, and its CIs are not adjusted for multiple comparisons.
- (a) HYSPLIT ranks the two chemicals made at the complex better:
  - vinyl chloride 0.59 vs 0.42 (CI of the difference -0.29 to -0.05);
  - EDC 0.64 vs 0.43 (-0.35 to -0.08).
- (a) ORI ranks the chemicals with many sources better: benzene 0.45 vs 0.28 (+0.01 to +0.32) and 1,3-butadiene 0.41 vs
  0.25 (+0.04 to +0.29). Total VOC shows no detectable difference (0.47 vs 0.43, -0.08 to +0.16).
- (b) Hours downwind gives the highest vinyl chloride score: 0.77 vs HYSPLIT 0.59 (-0.27 to -0.09).
- (b) Adding hours downwind to ORI gives 0.67 for vinyl chloride, not clearly different from HYSPLIT (+0.08, -0.002 to
  +0.17), and keeps ORI's benzene score (0.46 vs 0.45).


**Figure 6. Modeled plume transport.** The public HYSPLIT plume model for the Calvert City complex
(nwright083.github.io/Calvert-City-Plume-Analysis), shown for 15 February 2025. It uses 3 km HRRR weather and EPA TRI
emissions for each facility.

**Figure 7. Public odor-risk forecast for Marshall, McCracken and Livingston counties, updated daily.** A 0-100 Odor
Risk Index by census tract, with a multi-day outlook (nwright083.github.io/weather-variable-analysis), shown for
29 July 2026. The site has run the 2026 refit since 11 November 2026.

## Appendix

**Figure A1. Both models scored against the same monitor-days.** The share of monitor-days that were high total-VOC
days (top 25% at that monitor), below (white) vs above (blue) each monitor's median ORI forecast and median
HYSPLIT prediction.
- Sample: the same 282 monitor-days (141 sampling days, 2023-2025) as the main results.
- ORI: 15% vs 36% (2.3 times). HYSPLIT: 15% vs 36% (2.5 times). Both p < 0.001.
- Cochran-Mantel-Haenszel test, stratified by monitor.

**Figure A2. The same comparison for vinyl chloride and benzene.** The same 282 monitor-days as Figure A1. High vinyl
chloride days must also be detections.
- Vinyl chloride: ORI 13% vs 44% (3.4 times); HYSPLIT 8% vs 49% (6.3 times); both p < 0.001.
- Benzene: ORI 24% vs 51% (2.1 times, p < 0.001); HYSPLIT 31% vs 44% (1.4 times, p = 0.03).
