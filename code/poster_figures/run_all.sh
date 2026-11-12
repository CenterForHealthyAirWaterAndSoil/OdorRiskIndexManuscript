#!/bin/sh
# Draw Figures 1-7 and appendix Figures A1-A2 (output folder: posterstyle.OUT, or $POSTER_FIG_OUT if set)
set -e
PY=/path/to/weather-varaible-analysis/.venv/bin/python
cd "$(dirname "$0")"
$PY fig1_weather_drivers.py
$PY fig2_calibration_deciles.py
$PY fig3_transfer_to_calvert.py
$PY fig4_both_models.py                                                                   # appendix Fig A1, A2
$PY poster_screenshot.py fig5_poster_plume_screenshot.png Fig6_modeled_plume_transport        # the poster's plume image
$PY poster_screenshot.py fig6_poster_site_screenshot.png Fig7_public_forecast_site 775   # the poster's site image
$PY fig7_study_area.py                                                                    # Fig 6
"$PY" fig8_which_model_for_which_pollutant.py                                               # Fig 7
