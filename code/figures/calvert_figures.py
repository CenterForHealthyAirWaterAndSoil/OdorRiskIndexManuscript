"""Calvert City supplementary figures S2-S8 (rebuilt 2026-11-02 for the full HYSPLIT run).

The main figures are the updated poster figures (04_code/poster_figures/). This script reads only the package's own
datasets (02_analysis_datasets, 05_results). Statistics follow the pre-registered plans: within-monitor Spearman rho
(each monitor separately; "both monitors" = mean of the two, on dates both sampled), 95% CIs from 2,000 date-cluster
bootstrap resamples. HYSPLIT panels use the headline sample: the 156 new dates (DEVIATIONS 16 and 22).
Drawn in black and gray (figstyle.py). Test statistics and sample notes are printed when the script runs and are
quoted in 06_figures/supplementary/CAPTIONS.md; they are not drawn on the panels.
Run: ~/weather-varaible-analysis/.venv/bin/python 04_code/figures/calvert_figures.py
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S.setup()
D = os.path.join(S.PKG, "02_analysis_datasets"); R = os.path.join(S.PKG, "05_results")
rng = np.random.default_rng(20261023); B = 2000
UG = {"vc": (2, 62.5), "benzene": (6, 78.1)}            # ppbC -> ug/m3: ppb = ppbC / n_carbons; ug/m3 = ppb * MW / 24.45
to_ug = lambda s, k: s.clip(lower=0) / UG[k][0] * UG[k][1] / 24.45

SITE = {"21-157-0018": "elementary", "21-157-0020": "johnson_riley", "21-157-0021": "lwd"}
md = pd.read_csv(os.path.join(D, "calvert_monitor_days_2023_2025.csv")).rename(columns={
    "vinyl_chloride_ppbC": "vc", "benzene_ppbC": "benzene", "total_voc_ppbC": "total_voc",
    "poster_window_jan_jun_2025": "poster_window", "ori_refit_era5_pct": "odor_forecast_pct",   # the paper's ORI (DEVIATIONS 40)
    "hysplit_vinyl_chloride_ugm3": "p_vinyl_chloride", "hysplit_benzene_ugm3": "p_benzene", "hysplit_sum7_ugm3": "pred_sum7"})
md["site"] = md.aqs_site_id.map(SITE)
md["forecast_std"] = md.odor_forecast_pct
md["year"] = np.where(md.date >= "2025-07-01", "2025 (Jul-Dec)", md.date.str[:4])
for k in UG:
    md[f"{k}_ugm3"] = to_ug(md[k], k)
new = md[(~md.poster_window) & md.site.isin(S.MONITORS)].copy()            # dates not used on the poster
sets = pd.read_csv(os.path.join(D, "hysplit_date_sets.csv"))
pl = md[md.date.isin(set(sets[sets.set == "new156"].date)) & md.site.isin(S.MONITORS)].copy()
assert pl.p_vinyl_chloride.notna().all() and len(pl) == 271, len(pl)
PLUME_SCORE = {"vc": "p_vinyl_chloride", "benzene": "p_benzene", "total_voc": "pred_sum7"}


def rho_site(g, x, y):
    g = g.dropna(subset=[x, y])
    return spearmanr(g[x], g[y])[0] if g[y].nunique() > 1 and len(g) > 3 else np.nan


def rho_both(d, x, y):
    """Mean of the two monitors' rho on dates both monitors sampled (the pre-registered primary statistic)."""
    d = d.dropna(subset=[y]); both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)]
    return float(np.nanmean([rho_site(g, x, y) for _, g in d.groupby("site")]))


def boot(d, fn, B=B):
    byd = {k: v for k, v in d.groupby("date")}; dates = np.array(list(byd))
    vals = [fn(pd.concat([byd[k] for k in rng.choice(dates, len(dates))])) for _ in range(B)]
    return np.nanpercentile(vals, [2.5, 97.5])


def boot_site(g, x, y, B=B):
    g = g.dropna(subset=[x, y]); vals = []
    for _ in range(B):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        vals.append(rho_site(s, x, y))
    return np.nanpercentile(vals, [2.5, 97.5])


def mean_ci(v, B=B):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return v.mean(), *np.percentile(bs, [2.5, 97.5])


# =====================================================================================================
# S2. HYSPLIT predicted vs measured, per monitor
# =====================================================================================================
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
# the paper's sample: every 2023-2025 date on which both community monitors have vinyl chloride and benzene (141 dates)
s2 = md[md.site.isin(S.MONITORS) & md.vc.notna() & md.benzene.notna() & md.p_vinyl_chloride.notna()].copy()
s2 = s2[s2.groupby("date").site.transform("nunique") == 2]
assert s2.date.nunique() == 141 and len(s2) == 282, (s2.date.nunique(), len(s2))
BLUE, LO, HI = "#0072B2", 0.01, 30          # both axes share one range, so the line of equal values runs at 45 degrees
fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.6))
rows = []
for r, key in enumerate(["vc", "benzene"]):
    for c, site in enumerate(S.MONITORS):
        ax = axes[r, c]; S.clean(ax, grid="both")
        g = s2[s2.site == site]; x = g[PLUME_SCORE[key]]; y = g[f"{key}_ugm3"]
        floor = y[y > 0].min() / 2
        nd = (y <= 0).values; low = (x < LO).values
        xx = np.clip(x.values, LO, None); yy = np.where(nd, floor, y.values)
        ax.plot([LO, HI], [LO, HI], color=S.EDGE, linewidth=1.0, zorder=2)
        for f in (10, 0.1):
            ax.plot([LO, HI], [LO * f, HI * f], color="#9A9A9A", linewidth=0.8, linestyle=":", zorder=1)
        m = ~nd & ~low
        ax.scatter(xx[m], yy[m], s=22, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3, label="Detected")
        m = nd & ~low
        ax.scatter(xx[m], yy[m], s=22, facecolor="white", edgecolor=BLUE, linewidth=1.0, zorder=3,
                   label="Not detected (plotted at half the smallest detected value)")
        ax.scatter(xx[low], yy[low], s=24, marker="<", facecolor=np.where(nd[low], "white", BLUE), edgecolor=BLUE,
                   linewidth=1.0, zorder=3, label="Predicted below 0.01 (plotted at left)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(LO * 0.8, HI); ax.set_ylim(LO * 0.8, HI); ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        rho = rho_site(g, PLUME_SCORE[key], key)
        det = ~nd
        print(f"S2 {site} {key}: rho {rho:.2f}, n {len(g)}, non-detects {int(nd.sum())}, predicted < {LO}: {int(low.sum())}, "
              f"within a factor of 10 (detected): {np.mean((x.values[det] / y.values[det] >= 0.1) & (x.values[det] / y.values[det] <= 10)):.2f}")
        if r == 0: ax.set_title(S.MONITOR_LABEL[site], fontsize=9)
        if c == 0: ax.set_ylabel(f"Measured {S.POLLUTANT_LABEL[key].lower()} (µg/m³)")
        ax.set_xlabel(f"HYSPLIT predicted {S.POLLUTANT_LABEL[key].lower()} (µg/m³)")
        rows += [{"date": a, "monitor": site, "pollutant": key, "predicted_ugm3": b, "measured_ugm3": m}
                 for a, b, m in zip(g.date, x, y)]
axes[0, 0].text(0.04, 0.94, "predicted too low", transform=axes[0, 0].transAxes, ha="left", fontsize=7, color="#555555")
axes[0, 0].text(0.97, 0.04, "predicted too high", transform=axes[0, 0].transAxes, ha="right", fontsize=7, color="#555555")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.suptitle("HYSPLIT predictions against measurements, by monitor", x=0.01, ha="left", fontsize=10, fontweight="bold",
             y=0.995)
h, l = axes[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, fontsize=6.5)
S.save(fig, "FigS2_hysplit_predicted_vs_measured", pd.DataFrame(rows), supplementary=True)

# =====================================================================================================
# S3. Consistency across years and monitors
# =====================================================================================================
rows = []
for model, data, x in [("Odor forecast (ORI), all new dates", new, "odor_forecast_pct"),
                       ("HYSPLIT plume, all 156 new dates", pl, None)]:
    for key in ["vc", "benzene"]:
        xx = x or PLUME_SCORE[key]
        for yr in sorted(data.year.unique()):
            for site in S.MONITORS:
                g = data[(data.year == yr) & (data.site == site)]
                ci = boot_site(g, xx, key, B=1000)
                rows.append({"model": model, "pollutant": key, "year": yr, "monitor": S.MONITOR_SHORT[site],
                             "rho": rho_site(g, xx, key), "ci_low": ci[0], "ci_high": ci[1], "n": int(g[key].notna().sum())})
f3 = pd.DataFrame(rows)
fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), sharex=True)
for r, model in enumerate(f3.model.unique()):
    col = S.ORI if r == 0 else S.PLUME
    for c, key in enumerate(["vc", "benzene"]):
        ax = axes[r, c]; S.clean(ax, grid="x"); ax.axvline(0, color=S.AXIS, lw=0.8)
        d = f3[(f3.model == model) & (f3.pollutant == key)].reset_index(drop=True)
        labels = [f"{a}  {b}" for a, b in zip(d.year, d.monitor)]
        y = np.arange(len(d))[::-1]
        ax.errorbar(d.rho, y, xerr=[d.rho - d.ci_low, d.ci_high - d.rho], fmt="o", color=col, ms=6, mec=S.EDGE,
                    mew=0.8, elinewidth=1.3, capsize=0, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5); ax.set_xlim(-0.6, 1.0)
        ax.set_title(f"{model.split(',')[0]}: {S.POLLUTANT_LABEL[key].lower()}", fontsize=9)
        if r == 1: ax.set_xlabel("Within-monitor Spearman ρ (95% CI)")
fig.suptitle("Estimates by year and monitor", x=0.01, ha="left", fontsize=10, fontweight="bold")
print("S3 sample: ORI", len(new), "monitor-days; HYSPLIT", len(pl), "monitor-days")
fig.tight_layout()
S.save(fig, "FigS3_consistency_by_year_and_monitor", f3.round(3), supplementary=True)

# =====================================================================================================
# S4. All-chemical sweep for the odor forecast (exploratory)
# =====================================================================================================
sw = pd.read_csv(os.path.join(R, "ori", "species_sweep_refit.csv")).sort_values("rho")   # refit (species_sweep_refit.py)
NEG = {"Dichlorodifluoromethane", "1,1,2-Trichloro-1,2,2-trifluoroethane", "Freon 114", "Carbon tetrachloride", "Dichloromethane"}
fig, ax = plt.subplots(figsize=(5.4, 6.4)); S.clean(ax, grid="x"); ax.axvline(0, color=S.AXIS, lw=0.8)
y = np.arange(len(sw))
for yi, (_, r) in zip(y, sw.iterrows()):
    neg = r.chemical in NEG; sig = r.q_bh < 0.05
    ax.scatter(r.rho, yi, s=44, marker="s" if neg else "o", zorder=3,
               facecolor=(S.MUTED if neg else S.ORI) if sig or neg else "white",
               edgecolor=S.EDGE if neg else S.ORI, linewidth=1.0 if neg else 1.3)
ax.set_yticks(y); ax.set_yticklabels(sw.chemical, fontsize=7.5); ax.set_xlabel("Within-monitor Spearman ρ with the odor forecast")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.ORI, mec=S.ORI, label="q < 0.05 (Benjamini-Hochberg)"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec=S.ORI, label="q ≥ 0.05"),
                   Line2D([], [], marker="s", ls="", color=S.MUTED, mec=S.EDGE, label="Long-lived background gas (negative control)")],
          loc="lower right", fontsize=7)
ax.set_title("Exploratory: every measured chemical", fontsize=9.5)
print("S4 sample:", sorted(sw.monitor_days.unique()), "monitor-days;", int((sw.q_bh < 0.05).sum()), "of", len(sw), "at q < 0.05")
fig.tight_layout()
S.save(fig, "FigS4_all_chemical_sweep_exploratory", sw.assign(negative_control=sw.chemical.isin(NEG)), supplementary=True)

# =====================================================================================================
# S5. Fair comparison with standard trapping indices and locally trained models (exploratory)
# =====================================================================================================
fb = pd.read_csv(os.path.join(D, "calvert_new_dates_with_baseline_indices.csv"))
# the paper's ORI is the refit (DEVIATIONS 40): replace the deployed forecast in this exploratory comparison
fb = fb.drop(columns=["odor forecast (ours)"]).merge(md[["date", "site", "odor_forecast_pct"]].rename(
    columns={"odor_forecast_pct": "odor forecast (ours)"}), on=["date", "site"], how="left", validate="one_to_one")
assert fb["odor forecast (ours)"].notna().all()
fb_json = json.load(open(os.path.join(R, "ori", "fair_baselines.json")))
idx_cols = list(fb_json["vc"]["rho"].keys())
TRAINED = {"ridge trained on Calvert monitors", "random forest trained on Calvert monitors (rmweather-style)",
           "seasonal pattern only (same month, other years)"}
def wrho(d, col, y):
    return float(np.nanmean([rho_site(g, col, y) for _, g in d.groupby("site")]))
rows = []
for key in ["vc", "benzene", "total_voc"]:
    d0 = fb.dropna(subset=[key])
    for col in idx_cols:
        ci = boot(d0, lambda d: wrho(d, col, key), B=1000)
        rows.append({"pollutant": key, "measure": col, "rho": wrho(d0, col, key), "ci_low": ci[0], "ci_high": ci[1]})
f5b = pd.DataFrame(rows)
order = f5b[f5b.pollutant == "vc"].sort_values("rho").measure.tolist()
fig, axes = plt.subplots(1, 3, figsize=(8.6, 6.0), sharey=True)
for ax, key in zip(axes, ["vc", "benzene", "total_voc"]):
    S.clean(ax, grid="x"); ax.axvline(0, color=S.AXIS, lw=0.8)
    d = f5b[f5b.pollutant == key].set_index("measure").loc[order]
    for yi, (m, r) in enumerate(d.iterrows()):
        ours = m == "odor forecast (ours)"; tr = m in TRAINED
        col = S.ORI if ours else S.MUTED
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=1.3, zorder=2)
        ax.scatter(r.rho, yi, s=40 if ours else 28, marker="D" if tr else "o", zorder=3,
                   facecolor="white" if tr else col, edgecolor=S.ORI if ours else S.GRAY_DARK, linewidth=1.0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7); ax.set_xlim(-0.4, 1.0)
    ax.set_title(S.POLLUTANT_LABEL[key], fontsize=9); ax.set_xlabel("Within-monitor ρ (95% CI)")
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=S.ORI, label="Odor forecast (transferred, no local data)"),
                        Line2D([], [], marker="o", ls="", color=S.MUTED, mec=S.GRAY_DARK, label="Standard index (no local data)"),
                        Line2D([], [], marker="D", ls="", mfc="white", mec=S.GRAY_DARK, label="Uses Calvert monitor history")],
               loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=7)
print("S5 sample:", len(fb), "monitor-days")
fig.tight_layout()
S.save(fig, "FigS5_comparison_with_standard_indices", f5b.round(3), supplementary=True)

# =====================================================================================================
# S6. Spatial test: which monitor is higher on a given day
# =====================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
rows = []
for ax, key in zip(axes, ["vc", "benzene"]):
    S.clean(ax, grid="both")
    d = pl.dropna(subset=[key]); both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)]
    px = d.pivot(index="date", columns="site", values=PLUME_SCORE[key]); py = d.pivot(index="date", columns="site", values=key)
    dx = px.johnson_riley - px.elementary
    dy_ppbc = (py.johnson_riley - py.elementary).round(6)          # round off float noise so equal differences tie
    dy = dy_ppbc / UG[key][0] * UG[key][1] / 24.45
    ax.axhline(0, color=S.AXIS, lw=0.8); ax.axvline(0, color=S.AXIS, lw=0.8)
    ax.scatter(dx, dy, s=26, color=S.PLUME, edgecolor=S.EDGE, linewidth=0.6, zorder=3)
    rho = spearmanr(dx, dy_ppbc)[0]
    print(f"S6 {key}: rho {rho:.2f}, {len(dx)} shared dates")
    ax.set_title(S.POLLUTANT_LABEL[key], fontsize=9)
    ax.set_xlabel("Predicted: Johnson-Riley minus Elementary (µg/m³)"); ax.set_ylabel("Measured difference (µg/m³)")
    rows += [{"date": i, "pollutant": key, "pred_diff_jr_minus_elem": a, "meas_diff_jr_minus_elem": b} for i, a, b in zip(dx.index, dx, dy)]
fig.suptitle("HYSPLIT gets the between-monitor difference right for vinyl chloride, not benzene",
             x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout()
S.save(fig, "FigS6_spatial_difference_test", pd.DataFrame(rows), supplementary=True)

# =====================================================================================================
# S7. Magnitude: mean predicted over mean measured
# =====================================================================================================
mag = json.load(open(os.path.join(R, "hysplit", "plume_results_new156.json")))["secondary"]["magnitude"]
rows = [{"species": sp, "monitor": site, **v} for sp, by in mag.items() for site, v in by.items()]
mg = pd.DataFrame(rows); mg = mg[mg.ratio.notna()]
names = {"vinyl_chloride": "Vinyl chloride", "benzene": "Benzene", "edc": "Ethylene dichloride", "butadiene": "1,3-Butadiene",
         "chloroform": "Chloroform", "tetrachloroethylene": "Tetrachloroethylene", "trichloroethylene": "Trichloroethylene"}
sp_order = [s for s in names if s in set(mg.species)]
fig, ax = plt.subplots(figsize=(5.6, 3.2)); S.clean(ax, grid="x")
ax.axvline(1, color=S.INK2, lw=0.9)
ax.set_ylim(-0.9, len(sp_order) - 0.5)
ax.text(1.06, -0.75, "predicted = measured", fontsize=7, color=S.INK2, va="center")
for site, mk, fc, off in [("elementary", "o", "white", 0.13), ("johnson_riley", "s", S.PLUME, -0.13)]:
    d = mg[mg.monitor == site].set_index("species")
    ys = [len(sp_order) - 1 - sp_order.index(s) + off for s in d.index]
    ax.scatter(d.ratio, ys, marker=mk, s=40, facecolor=fc, edgecolor=S.EDGE, linewidth=1.0, zorder=3,
               label=S.MONITOR_LABEL[site])
ax.set_xscale("log"); ax.set_yticks(range(len(sp_order))); ax.set_yticklabels([names[s] for s in sp_order][::-1])
ax.set_xlabel("Mean predicted ÷ mean measured (log scale)")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.legend(loc="upper left", fontsize=7)
ax.set_title("HYSPLIT magnitude by species (non-detects counted as 0)", fontsize=9.5)
fig.tight_layout()
S.save(fig, "FigS7_hysplit_magnitude_by_species", mg, supplementary=True)

# =====================================================================================================
# S8. Forecast lead time (archived real forecasts)
# =====================================================================================================
ls = json.load(open(os.path.join(R, "ori", "lead_sensitivity.json")))
leads = sorted(int(k.split("_")[1]) for k in ls if k.startswith("lead_"))
rows = [{"lead_days": L, "pollutant": p, "rho": ls[f"lead_{L}"][p]["rho"], "p": ls[f"lead_{L}"][p]["p"],
         "monitor_days": ls[f"lead_{L}"]["monitor_days"]} for L in leads for p in ["vc", "benzene"]]
f9 = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(4.2, 2.8)); S.clean(ax)
for p, col, mk, ls_ in [("vc", S.BLACK, "o", "-"), ("benzene", S.GRAY_MID, "s", "--")]:
    d = f9[f9.pollutant == p]
    ax.plot(d.lead_days, d.rho, marker=mk, ls=ls_, color=col, ms=6.5, mec=S.EDGE, mew=0.8, label=S.POLLUTANT_LABEL[p])
    ax.text(d.lead_days.iloc[-1] + 0.15, d.rho.iloc[-1], S.POLLUTANT_LABEL[p], va="center", fontsize=8, color=S.INK)
ax.set_xticks(leads); ax.set_xlim(-0.3, max(leads) + 1.6); ax.set_ylim(0, 1)
ax.set_xlabel("Forecast lead (days ahead)"); ax.set_ylabel("Within-monitor ρ")
ax.set_title("Archived real forecasts (September 2024 onward)", fontsize=9)
print("S8 sample:", f9.monitor_days.iloc[0], "monitor-days"); print(f9.to_string(index=False))
fig.tight_layout()
S.save(fig, "FigS8_forecast_lead_time", f9, supplementary=True)
print("done")
