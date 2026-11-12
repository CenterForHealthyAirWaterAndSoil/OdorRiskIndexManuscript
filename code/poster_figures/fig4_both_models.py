"""Appendix Figures A1 and A2 (poster Figure 4, moved to the appendix 3 Nov 2026): both models graded the same way on
the same monitor-days.

Same layout as the poster panel, in black and gray (light gray below-median bars, black ORI / mid-gray HYSPLIT
above-median bars, % of monitor-days that were high-VOC days). Ratios and CMH p values are in the captions. What changed from the poster (57 pooled monitor-days, cap-50k HYSPLIT, Fisher exact test):
  * dates: every date with a finished, corrected HYSPLIT run (150,000-particle cap): the 156 new dates plus the 33
    poster dates re-run (validation/all_plume_dates_full.txt, 189 dates); every residential monitor-day on them
    with a valid measurement (Elementary and Johnson-Riley). The cap-50k poster runs are never used;
  * ORI: the 2026 refit on ERA5 inputs (f_refit; DEVIATIONS 40); HYSPLIT: the 7-species predicted
    sum for total VOC and the species-matched prediction for vinyl chloride and benzene (as run_plume.py);
  * WITHIN-MONITOR: each monitor-day is "above median" if its score is above that monitor's own median, and a
    "high-VOC day" if its measurement is in that monitor's own top 25% (vinyl chloride high days must also be
    detections, as hi_days in run_plume.py). This removes the between-monitor confounding the judges flagged;
  * statistic: ratio of the two shares, and a Cochran-Mantel-Haenszel test stratified by monitor (not Fisher).
Fig A1: total VOC (the poster's metric). Fig A2: vinyl chloride and benzene (the pre-registered endpoints).
"""
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable

import posterstyle as S
import matplotlib.pyplot as plt

SCORE = {"total_voc": "pred_sum7", "vc": "p_vinyl_chloride", "benzene": "p_benzene"}
LABEL = {"total_voc": "total VOC", "vc": "vinyl chloride", "benzene": "benzene"}


def hi_days(d, y):
    q = d.groupby("site")[y].transform(lambda v: v.quantile(0.75))
    return ((d[y] >= q) & (d[y] > 0)) if y == "vc" else (d[y] >= q)


def grade(d, y, score):
    d = d.dropna(subset=[y]).copy()
    d["hi"] = hi_days(d, y)
    d["above"] = d[score] > d.groupby("site")[score].transform("median")
    tabs = []
    for _, g in d.groupby("site"):
        tabs.append(np.array([[(g.above & g.hi).sum(), (g.above & ~g.hi).sum()],
                              [(~g.above & g.hi).sum(), (~g.above & ~g.hi).sum()]], float))
    st = StratifiedTable(tabs)
    pa, pb = d[d.above].hi.mean(), d[~d.above].hi.mean()
    return dict(pct_above=100 * pa, pct_below=100 * pb, ratio=pa / pb, cmh_p=float(st.test_null_odds().pvalue),
                cmh_or=float(st.oddsratio_pooled), n=len(d), n_above=int(d.above.sum()), n_below=int((~d.above).sum()),
                hi_above=int((d.above & d.hi).sum()), hi_below=int((~d.above & d.hi).sum()), high_days=int(d.hi.sum()),
                per_monitor={s: f"{int((g.above & g.hi).sum())}/{int(g.above.sum())} vs "
                                f"{int((~g.above & g.hi).sum())}/{int((~g.above).sum())}" for s, g in d.groupby("site")})


def panel(ax, st, model, y, show_ylabel=True, title=None, ytop=None):
    color = S.ORI_COL     # each panel is titled with its model, so both use white (below) and blue (above)
    lo, hi = st["pct_below"], st["pct_above"]
    ax.bar([0, 1], [lo, hi], width=0.5, color=["white", color], edgecolor=S.EDGE, linewidth=0.8, zorder=2)
    top = ytop if ytop is not None else max(hi, lo)   # shared within a pair of panels (review item 9)
    for xi, v in zip([0, 1], [lo, hi]):
        ax.text(xi, v + top * 0.015, f"{v:.0f}%", ha="center", va="bottom", fontsize=17, weight="bold", color=S.INK)
    ax.set_ylim(0, top * 1.15)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    if model == "ORI":
        ax.set_xticklabels(["Forecast below median", "Forecast above median"])
    else:
        ax.set_xticklabels(["Predicted plume below median", "Predicted plume above median"])
    if show_ylabel:
        ax.set_ylabel(f"% of monitor-days that were\nhigh-{LABEL[y]} days" if y != "total_voc"
                      else "% of monitor-days that were high-VOC days")
    S.grid_y(ax)
    if title:
        ax.set_title(title, loc="left", fontsize=14.5, weight="bold", color=S.INK, pad=10)


def row(model, y, st):
    return dict(model=model, pollutant=y, score=(S.ORI_SCORE if model == "ORI" else SCORE[y]),
                pct_high_below_median=round(st["pct_below"], 2), pct_high_above_median=round(st["pct_above"], 2),
                ratio=round(st["ratio"], 3), cmh_p=st["cmh_p"], cmh_odds_ratio=round(st["cmh_or"], 3),
                monitor_days=st["n"], monitor_days_above=st["n_above"], monitor_days_below=st["n_below"],
                high_days_above=st["hi_above"], high_days_below=st["hi_below"], high_days=st["high_days"],
                per_monitor_above_vs_below=str(st["per_monitor"]))


def main():
    S.setup_modern()
    d = S.plume_monitor_days_full()
    # the paper's sample: the 141 dates on which both community monitors have vinyl chloride and benzene
    d = d.dropna(subset=["vc", "benzene"])
    d = d[d.groupby("date").site.transform("nunique") == 2].copy()
    assert d.date.nunique() == 141 and len(d) == 282, (d.date.nunique(), len(d))
    # all dates, including the poster's days (DEVIATIONS 39)
    print(f"{d.date.nunique()} dates, {len(d)} residential monitor-days ({d.groupby('site').size().to_dict()})")
    rows = []

    # ---- Fig A1: total VOC (poster metric)
    y = "total_voc"
    so, sh = grade(d, y, S.ORI_SCORE), grade(d, y, SCORE[y])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.5, 6.9))
    yt = max(so["pct_above"], so["pct_below"], sh["pct_above"], sh["pct_below"])
    panel(a1, so, "ORI", y, title="ORI model: trapping forecast\nhigher forecast → more high-VOC days", ytop=yt)
    panel(a2, sh, "HYSPLIT", y, title="HYSPLIT model: plume transport\nmore predicted plume → more high-VOC days", ytop=yt)
    fig.suptitle("Both models, graded the same way: high-VOC days (top 25% at each monitor) at the Calvert community "
                 "monitors", fontsize=15.5, weight="bold", color=S.INK, y=1.0)
    fig.tight_layout(w_pad=4.0)
    rows += [row("ORI", y, so), row("HYSPLIT", y, sh)]
    for m, st in (("ORI", so), ("HYSPLIT", sh)):
        print(f"{y:<10} {m:<8} below {st['pct_below']:.1f}% above {st['pct_above']:.1f}% ratio {st['ratio']:.2f} "
              f"CMH OR {st['cmh_or']:.2f} p {st['cmh_p']:.2g} n {st['n']} {st['per_monitor']}")
    S.save(fig, "FigA1_both_models_same_monitor_days", pd.DataFrame(rows))

    # ---- Fig A2: vinyl chloride and benzene
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.6))
    for r, y in enumerate(["vc", "benzene"]):
        so, sh = grade(d, y, S.ORI_SCORE), grade(d, y, SCORE[y])
        name = "Vinyl chloride" if y == "vc" else "Benzene"
        yt = max(so["pct_above"], so["pct_below"], sh["pct_above"], sh["pct_below"])
        panel(axes[r, 0], so, "ORI", y, title=f"{name}: ORI model (trapping forecast)", ytop=yt)
        panel(axes[r, 1], sh, "HYSPLIT", y, title=f"{name}: HYSPLIT model (predicted {LABEL[y]})", ytop=yt)
        rows += [row("ORI", y, so), row("HYSPLIT", y, sh)]
        for m, st in (("ORI", so), ("HYSPLIT", sh)):
            print(f"{y:<10} {m:<8} below {st['pct_below']:.1f}% above {st['pct_above']:.1f}% ratio {st['ratio']:.2f} "
                  f"CMH OR {st['cmh_or']:.2f} p {st['cmh_p']:.2g} n {st['n']} {st['per_monitor']}")
    fig.suptitle("Both models on the same monitor-days: high vinyl chloride and benzene days (top 25% at each monitor)",
                 fontsize=15.5, weight="bold", color=S.INK, y=1.0)
    fig.tight_layout(w_pad=4.0, h_pad=3.0)
    S.save(fig, "FigA2_both_models_vinyl_chloride_and_benzene", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
