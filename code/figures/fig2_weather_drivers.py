"""Figure 1 data and Figure S1: weather drivers of odor complaints in Pittsburgh (citywide, daily).

Since 2026-11-11 (DEVIATIONS 40) this reads the refit's own training panel, so Figure 1 uses the same weather the
model is trained on: training_panel_v2/zip_day_panel.parquet (Copernicus ERA5, one source per variable, boundary
layer height in true feet; rain from Open-Meteo's ERA5), every Pittsburgh zip code in the panel (107).

  1. Citywide day: sum the zip codes' reports (every report, any rating); average each weather variable over the
     zip codes (each zip code carries its 0.25-degree cell's weather). DTR is the mean of the zip codes' daily
     max minus min. The panel holds complete local days only: N = 3,078 days, 2018-01-01 to 2026-06-05.
  2. Bin each variable (temperature 5 F, solar 20 W/m2, RH 1 %, wind 1 mph, rain 0.05 in, DTR 1 F, BLH 500 ft,
     pressure 1 hPa); drop bins with fewer than 3 days.
  3. Plot mean daily complaints per bin. The daily-level Spearman rho and p (all days, unbinned) are written to the
     data CSV and quoted in the caption, not drawn on the panels.

The earlier version of this script read the merged Open-Meteo file (ECMWF IFS weather, 61 zip codes, boundary layer
height corrected from a double conversion); it is kept as fig2_weather_drivers_merged_file_2026-11-02.py.bak.

Outputs: 06_figures/supplementary/FigS1_weather_drivers_8panel.{png,pdf} + _data.csv,
         02_analysis_datasets/pittsburgh_citywide_daily.csv (read by the main Figure 1)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = "/path/to/weather-varaible-analysis"
SRC = os.path.join(REPO, "training_panel_v2", "zip_day_panel.parquet")
PANEL_OUT = os.path.join(figstyle.PKG, "02_analysis_datasets", "pittsburgh_citywide_daily.csv")

MIN_DAYS_PER_BIN = 3

# key, panel title, x label, bin width. The first four are the main figure, in poster order.
VARIABLES = [
    ("wind_speed", "Wind speed", "Daily mean wind speed (mph)", 1),
    ("boundary_layer_height", "Boundary layer height", "Daily mean boundary layer height (ft)", 500),
    ("diurnal_temperature_range", "Diurnal temperature range", "Daily max minus min temperature (°F)", 1),
    ("temperature", "Temperature", "Daily mean temperature (°F)", 5),
    ("solar_radiation", "Solar radiation", "Daily mean shortwave radiation (W/m²)", 20),
    ("relative_humidity", "Relative humidity", "Daily mean relative humidity (%)", 1),
    ("precipitation", "Precipitation", "Daily rain total (in)", 0.05),
    ("atmospheric_pressure", "Surface pressure", "Daily mean surface pressure (hPa)", 1),
]
MAIN_KEYS = ["wind_speed", "boundary_layer_height", "diurnal_temperature_range", "temperature"]


def citywide_daily():
    p = pd.read_parquet(SRC)
    p = p[p.city == "pgh"]
    daily = p.groupby("date").agg(
        n_zip_codes=("zipcode", "nunique"), complaints=("n_reports", "sum"),
        temperature=("temperature", "mean"), temp_min=("temp_min", "mean"), temp_max=("temp_max", "mean"),
        diurnal_temperature_range=("diurnal_temperature_range", "mean"), precipitation=("precipitation", "mean"),
        wind_speed=("wind_speed", "mean"), relative_humidity=("relative_humidity", "mean"),
        atmospheric_pressure=("atmospheric_pressure", "mean"), solar_radiation=("solar_radiation", "mean"),
        boundary_layer_height=("boundary_layer_height", "mean")).reset_index()
    assert daily[[k for k, *_ in VARIABLES]].notna().all().all(), "a day lacks a weather value"
    return daily


def bin_table(daily, key, width):
    # 20 bins with the same number of days in each (fewer when many days share a value, as with rain); every day is used
    q = pd.qcut(daily[key], 20, duplicates="drop")
    return daily.groupby(q, observed=True).agg(bin_center=(key, "mean"), mean_daily_complaints=("complaints", "mean"),
                                               n_days=("complaints", "size")).reset_index(drop=True)


def draw(daily, keys, stats, name, supplementary):
    spec = {k: (t, xl, w) for k, t, xl, w in VARIABLES}
    nrows = int(np.ceil(len(keys) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(7.0, 2.55 * nrows + 0.2))
    rows = []
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        title, xlabel, width = spec[key]
        b = bin_table(daily, key, width)
        rho, p = stats[key]
        # Black and gray; the daily rho and p go in the caption and the data CSV, not on the panel.
        ax.plot(b["bin_center"], b["mean_daily_complaints"], "-", color="#0072B2", lw=0.8)
        ax.plot(b["bin_center"], b["mean_daily_complaints"], "o", color="#0072B2", ms=3.8,
                mec="black", mew=0.5)
        figstyle.clean(ax)
        ax.set_title(f"{chr(97 + i)}   {title}", loc="left", pad=6)
        if key == "wind_speed":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key == "boundary_layer_height":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel(xlabel)
        if i % 2 == 0:
            ax.set_ylabel("Mean daily complaints")
        ax.set_ylim(0, b["mean_daily_complaints"].max() * 1.08)
        decimals = 2 if width < 1 else 0
        for _, r in b.iterrows():
            rows.append({"panel": chr(97 + i), "variable": key, "x_label": xlabel, "bin_width": width,
                         "bin_center": round(r["bin_center"], decimals),
                         "mean_daily_complaints": round(r["mean_daily_complaints"], 3), "n_days": int(r["n_days"]),
                         "daily_spearman_rho": round(rho, 4), "daily_spearman_p": p, "n_days_total": len(daily)})
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    figstyle.save(fig, name, data=pd.DataFrame(rows), supplementary=supplementary)


def main():
    figstyle.setup()
    daily = citywide_daily()
    n = len(daily)
    print(f"N = {n} complete days ({daily['date'].min()} to {daily['date'].max()})")

    stats = {}
    print(f"{'variable':28s} {'rho':>7s} {'p':>10s} {'binned rho':>11s} bins")
    for key, *_ in VARIABLES:
        rho, p = spearmanr(daily[key], daily["complaints"])
        stats[key] = (rho, p)
        width = {k: w for k, _, _, w in VARIABLES}.get(key, 500)
        b = bin_table(daily, key, width)
        rho_b = spearmanr(b["bin_center"], b["mean_daily_complaints"])[0]
        print(f"{key:28s} {rho:+7.3f} {p:10.2e} {rho_b:+11.3f} {len(b):4d}")

    all_wind = daily.groupby(daily["wind_speed"].round())["complaints"].agg(["mean", "size"])
    print("Wind 12-18 mph bins (mean, days; bins under 3 days are not plotted):")
    print(all_wind.loc[12:].round(1).to_string())

    panel = daily.rename(columns={
        "temperature": "temperature_f", "temp_min": "temp_min_f", "temp_max": "temp_max_f",
        "diurnal_temperature_range": "diurnal_temperature_range_f", "precipitation": "rain_in",
        "wind_speed": "wind_speed_mph", "relative_humidity": "relative_humidity_pct",
        "atmospheric_pressure": "surface_pressure_hpa", "solar_radiation": "shortwave_radiation_wm2",
        "boundary_layer_height": "boundary_layer_height_ft",
    })
    cols = ["date", "n_zip_codes", "complaints", "temperature_f", "temp_min_f", "temp_max_f",
            "diurnal_temperature_range_f", "rain_in", "wind_speed_mph", "relative_humidity_pct",
            "surface_pressure_hpa", "shortwave_radiation_wm2", "boundary_layer_height_ft"]
    os.makedirs(os.path.dirname(PANEL_OUT), exist_ok=True)
    panel[cols].round(4).to_csv(PANEL_OUT, index=False)
    print("saved", PANEL_OUT)

    # 2026-11-02 rebuild: the 4-panel main figure is drawn in the poster's style by 04_code/poster_figures/
    # fig1_weather_drivers.py from the table written above; this script draws only the 8-panel S1.
    draw(daily, [k for k, *_ in VARIABLES], stats, "FigS1_weather_drivers_8panel", supplementary=True)


if __name__ == "__main__":
    main()
