"""Shared style and data helpers for the updated R!Louisville poster figures (2026-11-02).

Style follows the poster as presented (Research!Louisville, 5 Oct 2026): DejaVu Sans, light horizontal grid, bold
value labels. Since 3 Nov 2026 the figures are black and gray only (ORI black, HYSPLIT mid gray, comparison bars light
gray with a dark outline), and test statistics are given in the captions rather than printed on the panels.
Every figure is written as PNG (300 dpi), PDF, and a _data.csv of the plotted numbers to
~/school/figures_out/.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.environ.get("POSTER_FIG_OUT", "/path/to/school/figures_out")
EXT = "/path/to/school/R!Louisville/Re-analysis/extended"
PKG = "/path/to/school/Calvert_Publication_Package"
VALIDATION = "/path/to/plume-validation/validation"
PRED = os.path.join(VALIDATION, "results", "predicted_clipped")
RES = ["elementary", "johnson_riley"]
# The paper's ORI since 2026-11-11 (DEVIATIONS 40): the 2026 refit on ERA5 inputs. f_corrected is the earlier version.
ORI_SCORE = "f_refit"

TEAL, TEAL_LIGHT = "#0F9C8E", "#9BD5CD"
RUST, RUST_LIGHT = "#B5541C", "#E8B08C"
GREY_BAR = "#A9B1BE"
INK, GREY = "#1A1A1A", "#666666"
GRID, AXIS = "#E6E6E6", "#BDBDBD"
# Black and gray only (3 Nov 2026): readable with any color vision and in grayscale print. Neighbouring steps are
# 23 Delta-E apart under every color-vision simulation; the light step is low-contrast on white, so light bars get an
# EDGE outline and a value label.
BLACK, GRAY_DARK, GRAY_MID, GRAY_LIGHT = "#1A1A1A", "#4D4D4D", "#8C8C8C", "#D4D4D4"
EDGE = "#333333"
# Since 11 Nov 2026: blue and white. ORI is blue and HYSPLIT (and every comparison bar) is white with a dark outline,
# so the figures read with any color vision and in grayscale print; hatching marks hours downwind.
ORI_COL, HYS_COL = "#0072B2", "white"
# Poster Figures 1-2 (classic matplotlib look)
STEEL, PGH_BAR, PGH_RED = "#3A86C8", "#5B9BD0", "#D94040"


def setup_modern():
    """Poster Figures 3-4: flat bars, light grid, no top/right spines."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11, "axes.labelsize": 12.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white"})


def setup_classic():
    """Poster Figures 1-2: matplotlib defaults (DejaVu Sans, boxed axes)."""
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})


def grid_y(ax):
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=3, color=AXIS)
    ax.tick_params(axis="x", length=4, color=AXIS)


def save(fig, name, data=None):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    if data is not None:
        data.to_csv(os.path.join(OUT, f"{name}_data.csv"), index=False)
    plt.close(fig)
    print("saved", os.path.join(OUT, name))


def fmt_p(p, b=None):
    if b is not None and p <= 1.0 / (b + 1) + 1e-12:
        return f"p < {1.0 / b:.4f}".rstrip("0") if b else "p < 0.001"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


# ------------------------------------------------------------------ data

def with_fixed_total_voc(d):
    """Replace total_voc with the corrected definition (DEVIATIONS 38a): the 56 species reported in every year
    2023-2025, canisters averaged. The original column is kept as total_voc_all_species."""
    tf = pd.read_csv(os.path.join(EXT, "total_voc_fixed.csv"))[["date", "site", "total_voc_fixed"]]
    d = d.merge(tf, on=["date", "site"], how="left", validate="one_to_one")
    assert d.total_voc_fixed.notna().all(), "a monitor-day has no corrected total VOC"
    return d.rename(columns={"total_voc": "total_voc_all_species"}).rename(columns={"total_voc_fixed": "total_voc"})

def calvert_115():
    """Deployed ORI (corrected inputs) at the two residential monitors, sampling days not used on the poster, on the
    115 dates on which both monitors have valid vinyl chloride and benzene (230 monitor-days), as in items 26 and 30."""
    d = pd.read_csv(os.path.join(EXT, "refit_calvert_monitor_days.csv"))
    d = d[d.site.isin(RES) & ~d.poster_window].dropna(subset=["vc", "benzene", "f_corrected"])
    both = d.groupby("date").site.nunique()
    d = d[d.date.isin(both[both == 2].index)].copy()
    assert d.date.nunique() == 115 and len(d) == 230, (d.date.nunique(), len(d))
    return with_fixed_total_voc(d)


def calvert_all():
    """Deployed ORI (corrected inputs) at the two residential monitors on ALL 2023-2025 dates, including the poster's
    Jan-Jun 2025 days (DEVIATIONS 39): the 141 dates on which both monitors have valid vinyl chloride and benzene."""
    d = pd.read_csv(os.path.join(EXT, "refit_calvert_monitor_days.csv"))
    d = d[d.site.isin(RES)].dropna(subset=["vc", "benzene", "f_corrected"])
    both = d.groupby("date").site.nunique()
    d = d[d.date.isin(both[both == 2].index)].copy()
    assert d.date.nunique() == 141 and len(d) == 282, (d.date.nunique(), len(d))
    return with_fixed_total_voc(d)


def plume_monitor_days_full():
    """Every residential monitor-day on a date with a finished, corrected (150k-cap) HYSPLIT run: the 156 new dates
    plus the 33 poster dates re-run (validation/all_plume_dates_full.txt, 189 dates). Never the cap-50k runs."""
    sys.path.insert(0, "/path/to/plume-validation")
    from validation import config as C
    dates = [l.strip() for l in open(os.path.join(VALIDATION, "all_plume_dates_full.txt")) if l.strip()]
    missing = [x for x in dates if not os.path.exists(os.path.join(PRED, f"{x}.DONE"))]
    assert not missing, missing
    pr = pd.concat([pd.read_csv(os.path.join(PRED, f"{x}.csv")) for x in dates])
    pr = pr[pr.site.isin(RES)]
    pw = pr.pivot_table(index=["date", "site"], columns="species", values="pred_mean_ugm3").reset_index()
    pw = pw.rename(columns={sp: f"p_{sp}" for sp in C.TARGET_SPECIES})
    pw["pred_sum7"] = pw[[f"p_{sp}" for sp in C.TARGET_SPECIES]].sum(axis=1)
    md = pd.read_csv(os.path.join(EXT, "monitor_days.csv"))
    md = md[md.site.isin(RES) & md.date.isin(dates)]
    fs = pd.read_csv(os.path.join(EXT, "forecast_standard_source.csv"))[["date", "site", "forecast_std"]]
    rf = pd.read_csv(os.path.join(EXT, "refit_calvert_monitor_days.csv"))[["date", "site", "f_refit"]]
    d = md.merge(pw, on=["date", "site"], how="inner").merge(fs, on=["date", "site"], how="left")
    d = d.merge(rf, on=["date", "site"], how="left", validate="one_to_one")
    assert len(d) == len(md), "a residential monitor-day has no HYSPLIT prediction"
    assert d.forecast_std.notna().all(), "a monitor-day has no corrected ORI forecast"
    assert d.f_refit.notna().all(), "a monitor-day has no refit ORI"
    return with_fixed_total_voc(d.rename(columns={"forecast_std": "f_corrected"}))
