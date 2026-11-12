"""Updated poster Figure 5: modeled plume transport (HYSPLIT), from the corrected runs instead of a website screenshot.

Daily-mean ground-level (0-10 m) vinyl chloride, summed over every facility run (stack and fugitive), read from the
cdump files of the corrected (150,000-particle cap) runs. Helpers (cdump reader, grids, facility and monitor
locations) are imported unchanged from the publication package's fig6_plume_footprints.py / fig1_study_area.py.

Date rule (the package's fixed rule, re-applied over the full date set: every date with a finished corrected run and
both residential monitors sampled):
  a) among dates whose predicted VC at Johnson-Riley Road is in the top quartile, the one with the highest measured
     VC at Johnson-Riley Road ("plume toward the monitors");
  b) among dates with VC non-detects at both residential monitors, the one with the lowest max(predicted EL, JR)
     ("plume away from the monitors").
Styled like the poster map: light county fills, red-orange plume shading, rust facility squares, teal monitors.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import posterstyle as S

sys.path.insert(0, os.path.join(S.PKG, "04_code", "figures"))
import fig6_plume_footprints as F6  # noqa: E402
import fig1_study_area as F1  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

BOUNDS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
COLORS = ["#FDD9C4", "#FCB394", "#F98A66", "#EE5B3E", "#D2352A", "#A11B1B"]   # light to dark red-orange
COUNTY_FILL = {"145": "#DCE7F2", "157": "#E8F1E4", "139": "#F5E9DA"}
HALF_KM = 9.0
UPSAMPLE = 20
SHORT = {"elementary": "Elementary", "johnson_riley": "Johnson-Riley"}


def select_dates(d):
    w = d.pivot(index="date", columns="site", values="vc").dropna()
    p = d.pivot(index="date", columns="site", values="p_vinyl_chloride").loc[w.index]
    pool = pd.DataFrame({"vc_JR": w.johnson_riley, "vc_EL": w.elementary, "p_JR": p.johnson_riley, "p_EL": p.elementary})
    q75 = pool.p_JR.quantile(0.75)
    top = pool[pool.p_JR >= q75].sort_values(["vc_JR", "p_JR"], ascending=False)
    nd = pool[(pool.vc_JR == 0) & (pool.vc_EL == 0)].copy()
    nd["p_max"] = nd[["p_JR", "p_EL"]].max(axis=1)
    nd["p_sum"] = nd[["p_JR", "p_EL"]].sum(axis=1)
    nd = nd.sort_values(["p_max", "p_sum"])
    a, b = top.index[0], nd.index[0]
    print(f"pool {len(pool)} dates; (a) {len(top)} top-quartile dates (p_JR >= {q75:.4f}) -> {a}; "
          f"(b) {len(nd)} double non-detect dates -> {b}")
    print(top.head(3).to_string(float_format=lambda v: f"{v:.4g}"))
    rule = {a: f"highest measured VC at Johnson-Riley among {len(top)} dates with predicted VC there >= the pool 75th "
               f"percentile ({q75:.3f} ug/m3); pool = {len(pool)} dates",
            b: f"lowest max(predicted EL, JR) among {len(nd)} dates with VC non-detect at both monitors; "
               f"pool = {len(pool)} dates"}
    return [a, b], pool, rule


def main():
    S.setup_modern()
    d = S.plume_monitor_days_full()
    dates, pool, rule = select_dates(d)
    fac = F1.hysplit_facilities()
    mons = F1.monitor_sites()
    tracts = F1.read_tracts()
    counties = F1.county_outlines(tracts)
    lat_c, lon_c = F6.COMPLEX_CENTER
    kla, klo = F1.km_per_deg(lat_c)
    ext = dict(lat0=lat_c - HALF_KM / kla, lat1=lat_c + HALF_KM / kla, lon0=lon_c - HALF_KM / klo,
               lon1=lon_c + HALF_KM / klo)
    cmap, norm = ListedColormap(COLORS), BoundaryNorm(BOUNDS, len(COLORS))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.6))
    rows = []
    titles = ["plume toward the monitors", "plume away from the monitors"]
    for k, (ax, date) in enumerate(zip(axes, dates)):
        g = F6.daily_mean_grid(date)
        pred, n_hours = F6.predicted(date)
        interp = RegularGridInterpolator((g["lats"], g["lons"]), g["conc"], method="linear")
        step = g["dlat"] / UPSAMPLE
        la = np.arange(ext["lat0"], ext["lat1"] + step, step)
        lo = np.arange(ext["lon0"], ext["lon1"] + step, step)
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        Z = interp(np.column_stack([LA.ravel(), LO.ravel()])).reshape(LA.shape)
        assert Z.max() < BOUNDS[-1], f"{date}: max {Z.max():.2f} above the top bin"
        ax.set_facecolor("#F4F5F7")
        for c, rings, _ in tracts:
            for r in rings:
                ax.add_patch(Polygon(r, closed=True, fc=COUNTY_FILL.get(c, "#EEE"), ec="#8A93A0", lw=0.4, zorder=1))
        ax.contourf(lo, la, Z, levels=BOUNDS, colors=COLORS, alpha=0.85, zorder=2)
        for c, (u, rings) in counties.items():
            for r in rings:
                ax.add_patch(Polygon(r, closed=True, fc="none", ec="#5B6472", lw=1.1, zorder=3))
        for _, f in fac.iterrows():
            vc_src = F6.SPECIES_TAG in f.species.split()
            ax.plot(f.lon, f.lat, marker="s", ms=9, color=S.RUST if vc_src else "white", mec="black", mew=1.0,
                    ls="none", zorder=6)
        meas = pool.loc[date]
        place = {"johnson_riley": dict(xy=(0.03, 0.97), ha="left", va="top"),
                 "elementary": dict(xy=(0.97, 0.04), ha="right", va="bottom")}
        for site in S.RES:
            mlat, mlon = mons[site]["lat"], mons[site]["lon"]
            ax.plot(mlon, mlat, marker="o", ms=12, color=S.TEAL, mec="black", mew=1.3, ls="none", zorder=8)
            ppbc = meas["vc_JR" if site == "johnson_riley" else "vc_EL"]
            ug = F6.ppbc_to_ugm3(ppbc)
            grid_bl = float(interp([[mlat, mlon]])[0])
            p = pred[site]
            rel = (grid_bl - p) / p if p > 0 else 0.0
            assert abs(rel) <= F6.MAX_REL_DIFF, f"{date} {site}: grid {grid_bl:.5g} vs predicted {p:.5g}"
            mtxt = "not detected" if ppbc == 0 else f"{F6.fmt(ug)} µg/m³"
            a = place[site]
            ax.annotate(f"{SHORT[site]} monitor\nMeasured: {mtxt}\nModeled: {F6.fmt(p)} µg/m³", (mlon, mlat),
                        xytext=a["xy"], textcoords="axes fraction", ha=a["ha"], va=a["va"], fontsize=11.5,
                        color=S.INK, zorder=9, linespacing=1.25,
                        bbox=dict(fc="white", ec=S.AXIS, lw=0.8, alpha=0.95, boxstyle="round,pad=0.4"),
                        arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.9, shrinkA=0, shrinkB=7))
            rows.append(dict(panel="ab"[k], date=date, site=site, measured_vc_ppbC=ppbc, measured_vc_ugm3=round(ug, 4),
                             modeled_vc_ugm3=p, grid_check_rel_diff=round(rel, 5), selection_rule=rule[date]))
        ax.set_xlim(ext["lon0"], ext["lon1"])
        ax.set_ylim(ext["lat0"], ext["lat1"])
        F1.set_geo_aspect(ax, lat_c)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(S.AXIS)
        F1.scalebar(ax, ext["lon0"] + 0.012, ext["lat0"] + 0.012, 2, ticks=[0, 1, 2])
        ax.set_title(f"{'ab'[k]}.  {pd.Timestamp(date):%B %-d, %Y}: {titles[k]}", loc="left", fontsize=15,
                     weight="bold", color=S.INK, pad=8)
        print(f"[{date}] {g['n_runs']} runs, panel max {Z.max():.3f} ug/m3; measured JR {meas.vc_JR} EL {meas.vc_EL} ppbC")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.17, wspace=0.06)
    cax = fig.add_axes([0.08, 0.085, 0.42, 0.028])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=BOUNDS, spacing="uniform")
    cb.ax.set_xticklabels([f"{b:g}" for b in BOUNDS])
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Modeled daily-mean vinyl chloride near the ground, 0 to 10 m (µg/m³)", fontsize=12, color=S.INK)
    leg = [Line2D([], [], marker="s", ms=10, color=S.RUST, mec="black", ls="none", label="Vinyl chloride source"),
           Line2D([], [], marker="s", ms=10, color="white", mec="black", ls="none", label="Other modeled source"),
           Line2D([], [], marker="o", ms=11, color=S.TEAL, mec="black", ls="none", label="Community AQS monitor")]
    fig.legend(handles=leg, loc="center left", bbox_to_anchor=(0.56, 0.10), ncol=3, fontsize=11.5, frameon=False,
               handletextpad=0.4, columnspacing=1.4)
    fig.suptitle("Modeled plume transport: NOAA HYSPLIT, 3 km HRRR weather, EPA TRI emissions", fontsize=18,
                 weight="bold", color=S.INK, x=0.02, ha="left", y=0.985)
    S.save(fig, "Fig6_modeled_plume_transport", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
