"""Figure 2: calibration of the ORI in the training cities, by decile of out-of-fold prediction (2026 refit).

Black and gray: light-gray bars = mean reports per zip-code day (left axis), black line = share of zip-code days that
were odor events (right axis), one panel per city. Statistics are in the caption, not on the figure.
Data: 02_analysis_datasets/ori_oof_deciles.csv, written by 04_code/figures/fig3_ori_calibration.py from the refit's
out-of-fold predictions (each calendar month predicted by a fit without it). Since 2026-11-11 (DEVIATIONS 40); the
earlier figure showed the deployed model on Pittsburgh citywide days, in-sample.
"""
import os

import pandas as pd

import posterstyle as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SRC = os.path.join(S.PKG, "02_analysis_datasets", "ori_oof_deciles.csv")


def main():
    S.setup_classic()
    d = pd.read_csv(SRC)
    assert set(d.city) == {"Pittsburgh", "Louisville"} and len(d) == 20
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    x = list(range(1, 11))
    labels = [f"D{i}" for i in x]
    labels[0], labels[-1] = "D1\n(Lowest)", "D10\n(Highest)"
    for ax, city in zip(axes, ["Pittsburgh", "Louisville"]):
        g = d[d.city == city].sort_values("decile")
        ax.bar(x, g.mean_reports, width=0.65, color="white", edgecolor=S.EDGE, linewidth=0.8, zorder=2)
        ax.set_ylim(0, g.mean_reports.max() * 1.3)
        ax.set_ylabel("Mean reports per zip-code day (bars)", color=S.INK, fontsize=12)
        ax.grid(axis="y", color="#E5E5E5", lw=1.0)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel("Out-of-sample ORI decile (equal-count bins)", fontsize=12)
        ax.set_title(city, loc="left", fontsize=15, weight="bold")
        ax2 = ax.twinx()
        ax2.plot(x, g.event_pct, "-o", color=S.ORI_COL, lw=3, ms=7, zorder=4)
        ax2.set_ylim(0, max(40, g.event_pct.max() * 1.25))
        ax2.set_ylabel("Zip-code days that were odor events (%, line)", color=S.INK, fontsize=12)
        for xi, v in zip(x, g.event_pct):
            ax2.annotate(f"{v:.1f}%", (xi, v), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                         fontsize=9.5, weight="bold", color=S.ORI_COL,
                         bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
        ax.set_xlim(0.3, 10.7)
    axes[0].legend(handles=[Patch(fc="white", ec=S.EDGE, lw=0.8, label="Mean reports per zip-code day (left axis)"),
                            Line2D([], [], color=S.ORI_COL, lw=3, marker="o", ms=7, label="Odor event days, % (right axis)")],
                   loc="upper left", fontsize=10.5, frameon=False)
    fig.tight_layout(w_pad=4)
    S.save(fig, "Fig3_calibration_in_training_city", d)


if __name__ == "__main__":
    main()
