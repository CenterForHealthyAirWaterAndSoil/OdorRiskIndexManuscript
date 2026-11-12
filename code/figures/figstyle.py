"""Shared style for every figure in the publication package (static, print, white background).

Black and gray only, so every figure reads the same with any color vision and in grayscale print. The values match
04_code/poster_figures/posterstyle.py, which draws the main figures:
  ORI    (statistical odor forecast)  black
  PLUME  (HYSPLIT dispersion model)   mid gray, with a dark outline
Further series are told apart by marker shape, open vs filled markers, line style or direct labels, not by shade
alone. Test statistics go in the captions (06_figures/supplementary/CAPTIONS.md), not on the panels.
Every figure is saved as PNG (300 dpi) and PDF (vector), with a CSV of the plotted numbers beside it.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_MAIN = os.path.join(PKG, "06_figures", "main")
FIG_SUPP = os.path.join(PKG, "06_figures", "supplementary")

BLACK, GRAY_DARK, GRAY_MID, GRAY_LIGHT = "#1A1A1A", "#4D4D4D", "#8C8C8C", "#D4D4D4"
EDGE = "#333333"                                        # outline for gray markers and light bars
INK, INK2, MUTED = BLACK, GRAY_DARK, GRAY_MID
GRID, AXIS, SURF = "#E6E6E6", "#BDBDBD", "#ffffff"
ORI, PLUME = BLACK, GRAY_MID
# Gray ramp kept under the old names for the earlier scripts (fig1_study_area.py, fig6_plume_footprints.py),
# which are not part of the current figure set.
SEQ_GRAY = ["#F2F2F2", "#D9D9D9", "#BDBDBD", "#969696", "#737373", "#525252", "#252525"]
SEQ_BLUE = SEQ_ORANGE = SEQ_GRAY

MONITORS = ["elementary", "johnson_riley"]
MONITOR_LABEL = {"elementary": "Calvert City Elementary (1.5 mi)", "johnson_riley": "Johnson-Riley Road (0.55 mi)",
                 "lwd": "LWD (inside complex)"}
MONITOR_SHORT = {"elementary": "Elementary", "johnson_riley": "Johnson-Riley", "lwd": "LWD"}
POLLUTANT_LABEL = {"vc": "Vinyl chloride", "benzene": "Benzene", "total_voc": "Total VOC",
                   "butadiene": "1,3-Butadiene", "edc": "Ethylene dichloride"}


def setup():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "pdf.fonttype": 42, "ps.fonttype": 42,                      # editable text in Illustrator
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name, data=None, supplementary=False):
    """Write <name>.png (300 dpi) and <name>.pdf; write <name>_data.csv when a DataFrame is given."""
    out = FIG_SUPP if supplementary else FIG_MAIN
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    if data is not None:
        data.to_csv(os.path.join(out, f"{name}_data.csv"), index=False)
    plt.close(fig)
    print("saved", os.path.join(out, name))
