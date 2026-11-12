"""Updated poster Figure 1: weather conditions that trap odor (Pittsburgh citywide daily complaints).

Same layout as the poster panel (4 boxed panels of binned means), in black and gray. The daily Spearman rho and p for
each panel are reported in the caption, not on the figure.
Data: the publication package's corrected citywide daily table (02_analysis_datasets/pittsburgh_citywide_daily.csv,
built by 04_code/figures/fig2_weather_drivers.py): N = 3,078 complete local days, 2018-01-01 to 2026-06-05,
boundary-layer height in true feet (the poster's BLH axis was double-converted, metres x 3.28084^2).
Bins: fixed widths (wind 1 mph, BLH 250 ft, DTR 2 F, temperature 5 F). No day is dropped: at each end of the axis,
bins with fewer than MIN_DAYS days are pooled into one open-ended bin (for example "13 mph and above"), plotted at the
mean of its days.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import posterstyle as S
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SRC = os.path.join(S.PKG, "02_analysis_datasets", "pittsburgh_citywide_daily.csv")
PANELS = [("wind_speed_mph", "Wind Speed", "Wind Speed (mph)", 1),
          ("boundary_layer_height_ft", "Boundary Layer Height", "Boundary Layer Height (ft)", 250),
          ("diurnal_temperature_range_f", "Diurnal Temperature Range", "Diurnal Temperature Range (°F)", 2),
          ("temperature_f", "Temperature", "Temperature (°F)", 5)]


MIN_DAYS = 20


def fixed_bins(x, y, w, min_days=MIN_DAYS):
    """Mean of y in fixed-width bins of x. End bins with < min_days days are pooled inward, so every day is used."""
    k = np.floor(x / w).astype(int)
    ks = np.arange(k.min(), k.max() + 1)
    n = np.array([(k == i).sum() for i in ks])
    lo = 0
    while n[:lo + 1].sum() < min_days:
        lo += 1
    hi = len(ks) - 1
    while n[hi:].sum() < min_days:
        hi -= 1
    out = []
    for j in range(lo, hi + 1):
        m = (k <= ks[j]) if j == lo else (k >= ks[j]) if j == hi else (k == ks[j])
        if m.sum() == 0:
            continue
        pooled = (j == lo and lo > 0) or (j == hi and hi < len(ks) - 1)
        out.append(dict(bin=float(x[m].mean()) if pooled else (ks[j] + 0.5) * w, mean=float(y[m].mean()), size=int(m.sum()),
                        bin_low=float(x[m].min()) if (j == lo) else ks[j] * w,
                        bin_high=float(x[m].max()) if (j == hi) else (ks[j] + 1) * w, pooled_end_bin=bool(pooled)))
    b = pd.DataFrame(out)
    assert b["size"].sum() == len(x)
    return b


def main():
    S.setup_classic()
    d = pd.read_csv(SRC)
    assert len(d) == 3078, len(d)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    rows = []
    for ax, (col, title, xl, w) in zip(axes.flat, PANELS):
        rho, p = spearmanr(d[col], d.complaints)
        b = fixed_bins(d[col].to_numpy(), d.complaints.to_numpy(), w)
        ax.scatter(b["bin"], b["mean"], s=55, color=S.ORI_COL, edgecolor="black", linewidth=0.9, zorder=3)
        ax.set_title(title, fontsize=13, weight="bold", color="black", pad=8)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("Average Daily Complaints", fontsize=11)
        ax.tick_params(labelsize=10)
        if "boundary" in col:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for _, r in b.iterrows():
            rows.append(dict(panel=title, variable=col, bin_center=r["bin"], mean_daily_complaints=round(r["mean"], 3),
                             n_days=int(r["size"]), bin_low=r["bin_low"], bin_high=r["bin_high"],
                             pooled_end_bin=r["pooled_end_bin"], daily_spearman_rho=round(rho, 4), daily_p=p,
                             n_days_total=len(d)))
        print(f"{title:<28} daily rho {rho:+.3f} (p {p:.1e})  bins {len(b)}  days/bin {b['size'].min()}-{b['size'].max()}"
              f"  ends: <{b.bin_high.iloc[0]:g} ({b['size'].iloc[0]} d), >={b.bin_low.iloc[-1]:g} ({b['size'].iloc[-1]} d)")
    fig.tight_layout(h_pad=2.2, w_pad=2.5)
    S.save(fig, "Fig2_weather_conditions_that_trap_odor", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
