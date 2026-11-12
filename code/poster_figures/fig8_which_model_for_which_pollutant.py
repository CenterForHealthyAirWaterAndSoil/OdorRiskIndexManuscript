"""Figure 7 (was Fig 8; renumbered 3 Nov 2026; post hoc, DEVIATIONS 37): which score ranks which pollutant.
Black and gray only: ORI black, HYSPLIT mid gray, hours downwind light gray, ORI + hours downwind hatched.

(a) ORI vs HYSPLIT, within-monitor rho by pollutant, grouped as made at the complex vs many sources.
(b) Vinyl chloride and benzene: ORI, HYSPLIT, hours downwind, ORI + hours downwind.
Source: extended/all_dates_refit_results.json (all dates: 141 dates, 282 community monitor-days; DEVIATIONS 39-40).
The ORI is the 2026 refit on ERA5 inputs; ORI + hours downwind is built from it.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import posterstyle as S



def r2(v):
    """Two decimals from the UNROUNDED rho (review item 5: 0.594997 is 0.59)."""
    return f"{v:.2f}"


def verdict(r):
    lo, hi = r["ci95"]
    if lo > 0:
        return "ORI better"
    if hi < 0:
        return "HYSPLIT better"
    return "no clear difference"


def main():
    S.setup_modern()
    res = json.load(open(os.path.join(S.EXT, "all_dates_refit_results.json")))
    HYS = {"vc": "p_vinyl_chloride", "benzene": "p_benzene", "edc": "p_edc", "butadiene": "p_butadiene",
           "total_voc": "pred_sum7"}
    NAME = {"vc": ("vinyl chloride", "made at the complex"), "edc": ("EDC", "made at the complex"),
            "benzene": ("benzene", "many sources"), "butadiene": ("1,3-butadiene", "many sources"),
            "total_voc": ("total VOC", "many sources")}
    a = {y: dict(res["paired"][f"ORI minus HYSPLIT | {y}"], hysplit_score=HYS[y], pollutant=NAME[y][0], group=NAME[y][1])
         for y in HYS}
    ex = res["exact_rho"]
    order = ["vc", "edc", "benzene", "butadiene", "total_voc"]
    label = {"vc": "Vinyl\nchloride", "edc": "EDC", "benzene": "Benzene", "butadiene": "1,3-\nbutadiene",
             "total_voc": "Total\nVOC"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw=dict(width_ratios=[1.45, 1]))
    rows = []

    # ---- a
    xs = np.array([0, 1, 2.6, 3.6, 4.6])
    w = 0.36
    for x, y in zip(xs, order):
        r = a[y]
        for dx, val, col in ((-w / 2, ex[y][S.ORI_SCORE], S.ORI_COL), (w / 2, ex[y][r["hysplit_score"]], S.HYS_COL)):
            ax.bar(x + dx, val, w, color=col, edgecolor=S.EDGE, linewidth=0.8, zorder=3)
            ax.text(x + dx, max(val, 0) + 0.015, r2(val), ha="center", va="bottom", fontsize=12.5,
                    weight="bold", color=S.INK)
        v = verdict(r)
        rows.append(dict(panel="a", pollutant=r["pollutant"], group=r["group"], ori_rho=ex[y][S.ORI_SCORE],
                         hysplit_rho=ex[y][r["hysplit_score"]], diff_ori_minus_hysplit=r["diff"], ci_low=r["ci95"][0],
                         ci_high=r["ci95"][1], p_swap=r["p_swap"], verdict=v))
    ax.set_xticks(xs)
    ax.set_xticklabels([label[y] for y in order])
    ax.axhline(0, color=S.AXIS, lw=1)
    ax.axvline(1.8, color=S.AXIS, lw=1, ls="--")
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_ylabel("Within-monitor Spearman ρ with measured level")
    ax.text(0.5, 0.74, "Made at the complex", ha="center", fontsize=12.5, weight="bold", color=S.INK)
    ax.text(3.6, 0.74, "Many sources (traffic, fuels, regional)", ha="center", fontsize=12.5, weight="bold", color=S.INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=S.ORI_COL, ec=S.EDGE, lw=0.8, label="ORI (trapping forecast)"),
                       Patch(fc=S.HYS_COL, ec=S.EDGE, lw=0.8, label="HYSPLIT (plume model)")],
              loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=11.5)
    S.grid_y(ax)
    ax.set_title("a.  ORI and HYSPLIT by pollutant",
                 loc="left", fontsize=13.5, weight="bold", color=S.INK)

    # ---- b
    scores = [("ORI", lambda y: ex[y][S.ORI_SCORE], dict(color=S.ORI_COL)),
              ("HYSPLIT", lambda y: ex[y][a[y]["hysplit_score"]], dict(color=S.HYS_COL)),
              ("Hours downwind", lambda y: ex[y]["hd"], dict(color="white", hatch="///")),
              ("ORI + hours\ndownwind", lambda y: ex[y]["combo"], dict(color=S.ORI_COL, hatch="///", white_hatch=True))]
    w2 = 0.2
    for i, y in enumerate(["vc", "benzene"]):
        for j, (lab, f, sty) in enumerate(scores):
            x = i + (j - 1.5) * w2
            val = f(y)
            sty = dict(sty)
            if sty.pop("white_hatch", False):     # ORI's blue with hours downwind's diagonal lines, drawn in white
                bx.bar(x, val, w2 * 0.92, edgecolor="white", linewidth=0, zorder=3, label=lab if i == 0 else None, **sty)
                bx.bar(x, val, w2 * 0.92, facecolor="none", edgecolor=S.EDGE, linewidth=0.8, zorder=4)
            else:
                bx.bar(x, val, w2 * 0.92, edgecolor=S.EDGE, linewidth=0.8, zorder=3, label=lab if i == 0 else None, **sty)
            bx.text(x, val + 0.015, r2(val), ha="center", va="bottom", fontsize=11.5, weight="bold", color=S.INK)
            rows.append(dict(panel="b", pollutant=a[y]["pollutant"], score=lab.replace("\n", " "), rho=val))
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["Vinyl chloride", "Benzene"], fontsize=12.5)
    bx.set_ylim(0, 0.9)
    bx.set_yticks(np.arange(0, 0.81, 0.2))
    bx.set_ylabel("Within-monitor Spearman ρ")
    bx.legend(loc="upper right", frameon=False, fontsize=11, ncol=2)
    S.grid_y(bx)
    bx.set_title("b.  Adding hours downwind to ORI", loc="left", fontsize=13.5, weight="bold", color=S.INK)

    fig.suptitle("Which model for which pollutant (post hoc)",
                 fontsize=15, weight="bold", color=S.INK, y=1.02)
    fig.tight_layout(w_pad=3)
    S.save(fig, "Fig5_which_model_for_which_pollutant", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
