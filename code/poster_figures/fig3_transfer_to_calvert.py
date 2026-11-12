"""Updated poster Figure 3: transfer to Calvert City (measured VOC on lower- vs higher-risk forecast days).

Same layout as the poster panel, now with benzene, in black and gray (light gray = lower-risk half, black =
higher-risk half). The ratios and p values are reported in the caption, not on the figure.
What changed from the poster (57 pooled monitor-days, Jan to Jun 2025, Mann-Whitney):
  * data: the 115 dates in 2023 to 2025 on which both residential monitors have valid vinyl chloride and benzene,
    excluding the poster's own window (230 monitor-days; the sample of DEVIATIONS items 26 and 30);
  * forecast: the paper's ORI, the 2026 refit on ERA5 inputs (`f_refit`; DEVIATIONS 40). Until 2026-11-11 this was
    the deployed ORI with corrected inputs (`f_corrected`), now the earlier version;
  * split: at EACH MONITOR'S OWN median forecast (higher half = above that monitor's median), so both halves hold the
    same mix of the two monitors and the far-higher Johnson-Riley values cannot make the split by themselves;
  * ratio: mean of the higher half / mean of the lower half over monitor-days (the ratio of the two bars);
  * p: joint date permutation of the forecast (the same shuffle of dates at both monitors), 10,000 draws,
    (count + 1) / (B + 1), two-sided on |log ratio|.
Also printed (and written to the data CSV): each monitor's own ratio and their geometric mean, which is the statistic
behind the 2026-10-28 numbers (1.32x / 3.57x / 1.38x).
Units: total VOC in ppbC as reported; vinyl chloride and benzene converted from ppbC to ug/m3 at 25 C
(ug/m3 = ppbC / nC x MW / 24.45), as on the poster.
"""
import numpy as np
import pandas as pd

import posterstyle as S
import matplotlib.pyplot as plt

B = 10000
SEED = 20261102
POLL = [("total_voc", "Total VOC", "Mean measured total VOC (ppbC)", 1.0),
        ("vc", "Vinyl chloride", "Mean measured vinyl chloride (µg/m³)", 62.5 / 2 / 24.45),
        ("benzene", "Benzene", "Mean measured benzene (µg/m³)", 78.11 / 6 / 24.45)]


def stats(d, y, rng):
    piv_f = d.pivot(index="date", columns="site", values=S.ORI_SCORE)
    piv_y = d.pivot(index="date", columns="site", values=y)
    n = len(piv_f)
    F = {s: piv_f[s].to_numpy() for s in S.RES}
    Y = {s: piv_y[s].to_numpy(float) for s in S.RES}
    med = {s: np.median(F[s]) for s in S.RES}

    def pooled(hi):  # hi: dict site -> bool array (n,) or (B, n)
        num_hi = sum((hi[s] * Y[s]).sum(-1) for s in S.RES)
        n_hi = sum(hi[s].sum(-1) for s in S.RES)
        num_lo = sum(((~hi[s]) * Y[s]).sum(-1) for s in S.RES)
        n_lo = sum((~hi[s]).sum(-1) for s in S.RES)
        return num_hi / n_hi, num_lo / n_lo

    hi0 = {s: F[s] > med[s] for s in S.RES}
    m_hi, m_lo = pooled(hi0)
    obs = np.log(m_hi / m_lo)
    perms = np.array([rng.permutation(n) for _ in range(B)])
    hiP = {s: F[s][perms] > med[s] for s in S.RES}
    p_hi, p_lo = pooled(hiP)
    null = np.log(p_hi / p_lo)
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1)
    per = {s: Y[s][hi0[s]].mean() / Y[s][~hi0[s]].mean() for s in S.RES}
    return dict(mean_hi=m_hi, mean_lo=m_lo, ratio=m_hi / m_lo, p=p, n_hi=int(sum(hi0[s].sum() for s in S.RES)),
                n_lo=int(sum((~hi0[s]).sum() for s in S.RES)), ratio_elementary=per["elementary"],
                ratio_johnson_riley=per["johnson_riley"], geo_mean_monitor_ratio=float(np.sqrt(np.prod(list(per.values())))))


def main():
    S.setup_modern()
    rng = np.random.default_rng(SEED)
    d = S.calvert_all()   # all dates (DEVIATIONS 39)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.6))
    rows = []
    for ax, (y, title, ylab, conv) in zip(axes, POLL):
        st = stats(d, y, rng)
        lo, hi = st["mean_lo"] * conv, st["mean_hi"] * conv
        dec = 1 if y == "total_voc" else 2
        ax.bar([0, 1], [lo, hi], width=0.5, color=["white", S.ORI_COL], edgecolor=S.EDGE, linewidth=0.8, zorder=2)
        for xi, v in zip([0, 1], [lo, hi]):
            ax.text(xi, v * 1.012 + hi * 0.01, f"{v:.{dec}f}", ha="center", va="bottom", fontsize=17, weight="bold",
                    color=S.INK)
        ax.set_ylim(0, hi * 1.15)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Lower-risk half of days\n(forecast below median)",
                            "Higher-risk half of days\n(forecast above median)"])
        ax.set_ylabel(ylab)
        S.grid_y(ax)
        ax.set_title(title, loc="left", fontsize=16, weight="bold", color=S.INK, pad=10)
        rows.append(dict(pollutant=y, units=ylab.split("(")[-1].rstrip(")"), mean_lower_half=round(lo, 4),
                         mean_higher_half=round(hi, 4), ratio=round(st["ratio"], 3), p_joint_date_permutation=st["p"],
                         permutations=B, monitor_days_higher=st["n_hi"], monitor_days_lower=st["n_lo"],
                         ratio_elementary=round(st["ratio_elementary"], 3),
                         ratio_johnson_riley=round(st["ratio_johnson_riley"], 3),
                         geometric_mean_of_monitor_ratios=round(st["geo_mean_monitor_ratio"], 3),
                         dates=141, monitor_days=282, forecast="ORI: 2026 refit on ERA5 inputs (f_refit)",
                         split="above vs at-or-below each monitor's own median forecast"))
        print(f"{title:<15} lower {lo:.3f} higher {hi:.3f} ratio {st['ratio']:.3f} p {st['p']:.4f} | EL "
              f"{st['ratio_elementary']:.2f} JR {st['ratio_johnson_riley']:.2f} geo {st['geo_mean_monitor_ratio']:.3f}")
    fig.suptitle("When the model forecast higher risk, the monitors measured more VOCs", fontsize=19, weight="bold",
                 color=S.INK, y=1.0)
    fig.tight_layout(w_pad=3.0)
    S.save(fig, "Fig4_transfer_to_Calvert_City", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
