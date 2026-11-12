"""Figure 6. HYSPLIT daily-mean ground-level vinyl chloride footprints on two contrasting days.

Each panel is the 23-h mean (local day, 00-23 h standard time) of the hourly 0-10 m vinyl chloride (HYSPLIT tag VCLR)
concentration, summed over every facility run (stack and fugitive) for that date, in ug/m3.

Date selection (pool: dates with both residential monitors sampled and a finished HYSPLIT run, i.e. a .DONE marker):
  a) "plume reached the monitors, VC high": among pool dates whose predicted VC at Johnson-Riley Road is in the top
     quartile of the pool (>= 75th percentile), the date with the highest measured VC at Johnson-Riley Road.
  b) "plume elsewhere, VC not detected": among pool dates with VC non-detects at both residential monitors, the date
     with the lowest predicted VC at the more exposed of the two monitors (min over dates of max(pred_EL, pred_JR)).

Display: the 0.01 deg HYSPLIT grid is bilinearly interpolated to 0.0005 deg (the interpolation con2stn uses for the
monitor predictions) and drawn as filled log-spaced bands, so the band under each monitor contains its annotated model
value. The script checks that
the interpolated grid reproduces predicted_clipped at each monitor and stops if it does not.

Inputs (read-only):
  runs_clipped/<date>/<facility>_<release>/cdump and CONTROL   (only dates with predicted_clipped/<date>.DONE)
  results/predicted_clipped/<date>.csv
  /path/to/school/R!Louisville/Re-analysis/extended/plume_monitor_days.csv
  /path/to/weather-varaible-analysis/calvert_tracts.geojson
Output: 06_figures/main/Fig6_plume_footprints.{png,pdf} and Fig6_plume_footprints_data.csv (per-monitor values).

Run: /path/to/weather-varaible-analysis/.venv/bin/python 04_code/figures/fig6_plume_footprints.py
"""
import glob
import os
import struct
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402
import fig1_study_area as f1  # noqa: E402  (shared map helpers, facility and monitor readers)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

MONITOR_DAYS = "/path/to/school/R!Louisville/Re-analysis/extended/plume_monitor_days.csv"
SPECIES_TAG, SPECIES_KEY = "VCLR", "vinyl_chloride"
AIR_LEVEL_M = 10                       # cdump levels are 0 m (deposition) and 10 m (0-10 m air layer)
MW_VC, N_C_VC, MOLAR_VOLUME = 62.5, 2, 24.45
RESIDENTIAL = ["johnson_riley", "elementary"]
COMPLEX_CENTER = (37.0486, -88.3480)    # complex centroid; the monitor distances (0.55 mi, 1.5 mi) are from here
HALF_KM = 9.0                           # panels show the complex +/- 9 km
UPSAMPLE = 20                           # 0.01 deg grid -> 0.0005 deg display raster
BOUNDS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]   # ug/m3, half-decade (log) bins
COLORS = figstyle.SEQ_ORANGE[1:]            # six steps of the shared orange ramp, light to dark
MAX_REL_DIFF = 0.02                         # grid vs predicted_clipped tolerance at the monitors


# ---------------------------------------------------------------- HYSPLIT cdump reader
def _records(buf):
    """Fortran unformatted sequential records, big-endian 4-byte length markers."""
    pos = 0
    while pos < len(buf):
        (n,) = struct.unpack(">i", buf[pos:pos + 4])
        assert struct.unpack(">i", buf[pos + 4 + n:pos + 8 + n])[0] == n, "cdump record marker mismatch"
        yield buf[pos + 4:pos + 4 + n]
        pos += 8 + n


def read_cdump(path, pollutant, level_m):
    """Return (grid dict, array [n_periods, nlat, nlon] in g/m3) for one pollutant and level of a cdump.
    Handles packed (cpack=1) and unpacked files. Verified against con2asc output (all cells, 4 significant digits)."""
    recs = _records(open(path, "rb").read())
    r = next(recs)
    nloc, pack = struct.unpack(">2i", r[24:32])
    for _ in range(nloc):
        next(recs)
    r = next(recs)
    nlat, nlon = struct.unpack(">2i", r[:8])
    dlat, dlon, lat0, lon0 = (round(v, 4) for v in struct.unpack(">4f", r[8:24]))
    r = next(recs)
    (nlev,) = struct.unpack(">i", r[:4])
    levels = list(struct.unpack(f">{nlev}i", r[4:4 + 4 * nlev]))
    r = next(recs)
    (npol,) = struct.unpack(">i", r[:4])
    pols = [r[4 + 4 * k:8 + 4 * k].decode().strip() for k in range(npol)]
    grid = dict(nlat=nlat, nlon=nlon, dlat=dlat, dlon=dlon, lat0=lat0, lon0=lon0, levels=tuple(levels))
    periods, out = [], []
    for r in recs:
        start = struct.unpack(">5i", r[:20])
        stop = struct.unpack(">5i", next(recs)[:20])
        arr = np.zeros((nlat, nlon))
        for _ in range(npol * nlev):
            r = next(recs)
            if r[:4].decode().strip() != pollutant or struct.unpack(">i", r[4:8])[0] != level_m:
                continue
            if pack:
                (nxy,) = struct.unpack(">i", r[8:12])
                v = np.frombuffer(r[12:12 + 8 * nxy], dtype=np.dtype([("i", ">i2"), ("j", ">i2"), ("c", ">f4")]))
                arr[v["j"] - 1, v["i"] - 1] = v["c"]
            else:
                arr[:] = np.frombuffer(r[8:8 + 4 * nlat * nlon], dtype=">f4").reshape(nlat, nlon)
        periods.append((start, stop))
        out.append(arr)
    grid["periods"] = periods
    grid["has_pollutant"] = pollutant in pols
    return grid, np.array(out)


def daily_mean_grid(date):
    """Sum of the VC 0-10 m hourly grids over every facility run for `date`, averaged over the hourly periods, ug/m3."""
    runs = sorted(glob.glob(os.path.join(f1.RUNS, date, "*", "cdump")))
    assert runs, f"no cdump files for {date}"
    total, ref, n_runs_vc = None, None, 0
    for path in runs:
        assert os.path.exists(os.path.join(os.path.dirname(path), "HYSPLIT_OK")), f"unfinished run: {path}"
        g, a = read_cdump(path, SPECIES_TAG, AIR_LEVEL_M)
        key = {k: g[k] for k in ("nlat", "nlon", "dlat", "dlon", "lat0", "lon0", "periods")}
        if ref is None:
            ref = key
        assert key == ref, f"grid or sampling periods differ between runs: {path}"
        hours = {(p[1][3] - p[0][3]) % 24 for p in g["periods"]}
        assert hours == {1}, f"expected hourly averaging periods, got {hours}: {path}"
        if g["has_pollutant"]:
            n_runs_vc += 1
            total = a if total is None else total + a
    n_periods = len(ref["periods"])
    lats = ref["lat0"] + ref["dlat"] * np.arange(ref["nlat"])
    lons = ref["lon0"] + ref["dlon"] * np.arange(ref["nlon"])
    mean_ug = total.sum(axis=0) / n_periods * 1e6
    return dict(lats=lats, lons=lons, conc=mean_ug, n_periods=n_periods, n_runs=len(runs), n_runs_vc=n_runs_vc,
                dlat=ref["dlat"])


# ---------------------------------------------------------------- date selection
def ppbc_to_ugm3(ppbc):
    return ppbc / N_C_VC * MW_VC / MOLAR_VOLUME


def predicted(date):
    p = pd.read_csv(os.path.join(f1.PRED, f"{date}.csv"))
    p = p[p.species == SPECIES_KEY].set_index("site")
    return p.pred_mean_ugm3.to_dict(), p.n_hours.to_dict()


def select_dates():
    done = set(f1.done_dates())
    m = pd.read_csv(MONITOR_DAYS)
    m = m[m.date.isin(done) & m.site.isin(RESIDENTIAL)]
    w = m.pivot(index="date", columns="site", values="vc").dropna()       # both residential monitors sampled
    pool = pd.DataFrame({"vc_JR": w["johnson_riley"], "vc_EL": w["elementary"]})
    pred = {d: predicted(d)[0] for d in pool.index}
    pool["p_JR"] = [pred[d]["johnson_riley"] for d in pool.index]
    pool["p_EL"] = [pred[d]["elementary"] for d in pool.index]
    # predicted_clipped must agree with the p_vinyl_chloride column of the monitor-day table
    chk = m.set_index(["date", "site"]).p_vinyl_chloride
    for d in pool.index:
        for s, col in (("johnson_riley", "p_JR"), ("elementary", "p_EL")):
            assert abs(chk[(d, s)] - pool.loc[d, col]) < 1e-9, f"p_vinyl_chloride != predicted_clipped for {d} {s}"

    q75 = pool.p_JR.quantile(0.75)
    top = pool[pool.p_JR >= q75].sort_values(["vc_JR", "p_JR"], ascending=False)
    date_a = top.index[0]
    nd = pool[(pool.vc_JR == 0) & (pool.vc_EL == 0)].copy()
    nd["p_max"] = nd[["p_JR", "p_EL"]].max(axis=1)
    nd["p_sum"] = nd[["p_JR", "p_EL"]].sum(axis=1)
    nd = nd.sort_values(["p_max", "p_sum"])
    date_b = nd.index[0]

    print(f"Pool: {len(pool)} dates with both residential monitors sampled and a finished HYSPLIT run")
    print(f"(a) top-quartile predicted VC at Johnson-Riley: p_JR >= {q75:.4f} ug/m3 ({len(top)} dates); "
          f"highest measured VC at Johnson-Riley -> {date_a}")
    print(top.head(4).to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"(b) VC non-detect at both monitors ({len(nd)} dates); lowest max(p_EL, p_JR) -> {date_b}")
    print(nd.head(4).to_string(float_format=lambda v: f"{v:.4g}"))
    rule = {date_a: f"a: highest measured VC at Johnson-Riley among {len(top)} pool dates with predicted VC at "
                    f"Johnson-Riley >= pool 75th percentile ({q75:.3f} ug/m3); pool = {len(pool)} dates",
            date_b: f"b: lowest max(predicted EL, predicted JR) among {len(nd)} pool dates with VC non-detect at "
                    f"both monitors; pool = {len(pool)} dates"}
    return [date_a, date_b], pool, rule


# ---------------------------------------------------------------- figure
def fmt(v):
    if v == 0:
        return "0"
    return "<0.01" if v < 0.01 else f"{v:.2f}"


def bearing_of_mass(g, lat_c, lon_c, rmin_km=2.0, rmax_km=15.0):
    """Concentration-weighted mean direction (deg clockwise from north) of the footprint 2-15 km from the complex."""
    kla, klo = f1.km_per_deg(lat_c)
    LA, LO = np.meshgrid(g["lats"], g["lons"], indexing="ij")
    dy, dx = (LA - lat_c) * kla, (LO - lon_c) * klo
    r = np.hypot(dx, dy)
    w = g["conc"] * ((r > rmin_km) & (r < rmax_km))
    return float(np.degrees(np.arctan2((w * dx).sum(), (w * dy).sum())) % 360)


def main():
    figstyle.setup()
    dates, pool, rule = select_dates()
    fac = f1.hysplit_facilities()
    mons = f1.monitor_sites()
    tracts = f1.read_tracts()
    counties = f1.county_outlines(tracts)

    lat_c, lon_c = COMPLEX_CENTER
    kla, klo = f1.km_per_deg(lat_c)
    ext = dict(lat0=lat_c - HALF_KM / kla, lat1=lat_c + HALF_KM / kla,
               lon0=lon_c - HALF_KM / klo, lon1=lon_c + HALF_KM / klo)

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(BOUNDS, cmap.N)

    W, pw, gap, left = 7.2, 3.45, 0.24, 0.03
    ph = pw                                   # square panels: equal km in x and y
    bottom = 0.95
    H = bottom + ph + 0.30
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([left / W, bottom / H, pw / W, ph / H]),
            fig.add_axes([(left + pw + gap) / W, bottom / H, pw / W, ph / H])]

    rows, vmax_seen = [], 0.0
    titles = {0: "plume toward the monitors", 1: "plume away from the monitors"}
    for k, (ax, date) in enumerate(zip(axes, dates)):
        g = daily_mean_grid(date)
        pred, n_hours = predicted(date)
        assert all(n == g["n_periods"] for n in n_hours.values()), "n_hours in predicted_clipped != cdump periods"
        interp = RegularGridInterpolator((g["lats"], g["lons"]), g["conc"], method="linear")

        # display raster: bilinear, as con2stn
        step = g["dlat"] / UPSAMPLE
        la = np.arange(ext["lat0"], ext["lat1"] + step, step)
        lo = np.arange(ext["lon0"], ext["lon1"] + step, step)
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        Z = interp(np.column_stack([LA.ravel(), LO.ravel()])).reshape(LA.shape)
        vmax_seen = max(vmax_seen, float(Z.max()))
        ax.contourf(lo, la, Z, levels=BOUNDS, colors=COLORS, zorder=1)     # below 0.01 ug/m3 is left blank

        for c, rings, props in tracts:
            for r in rings:
                ax.add_patch(Polygon(r, closed=True, fc="none", ec=figstyle.INK2, lw=0.25, alpha=0.55, zorder=2))
        for c, (u, rings) in counties.items():
            for r in rings:
                ax.add_patch(Polygon(r, closed=True, fc="none", ec=figstyle.INK2, lw=0.7, zorder=3))
        ax.set_xlim(ext["lon0"], ext["lon1"])
        ax.set_ylim(ext["lat0"], ext["lat1"])
        f1.set_geo_aspect(ax, lat_c)
        f1.map_axes(ax)

        for _, f in fac.iterrows():
            vc_source = SPECIES_TAG in f.species.split()
            style = dict(f1.FAC_STYLE)
            if not vc_source:
                style.update(mfc=figstyle.SURF, mec=figstyle.INK, mew=0.7)
            ax.plot(f.lon, f.lat, zorder=6, **style)

        # grid summary on the native nodes inside the panel extent
        inside = (((g["lats"] >= ext["lat0"]) & (g["lats"] <= ext["lat1"]))[:, None]
                  & ((g["lons"] >= ext["lon0"]) & (g["lons"] <= ext["lon1"]))[None, :])
        node = np.where(inside, g["conc"], -1.0)
        jmax = np.unravel_index(np.argmax(node), node.shape)
        node_max = float(node[jmax])
        bearing = bearing_of_mass(g, lat_c, lon_c)
        print(f"[{date}] {g['n_runs']} runs ({g['n_runs_vc']} with VC), {g['n_periods']} hourly periods; "
              f"max node in panel {node_max:.3f} ug/m3 at {g['lats'][jmax[0]]:.4f},{g['lons'][jmax[1]]:.4f}; "
              f"footprint bearing {bearing:.0f} deg")

        meas = pool.loc[date]
        # Johnson-Riley: box upper left, elbow leader that reaches the monitor from the west (clear of the
        # facility markers north of it). Elementary: box lower right, straight leader.
        ann = {"johnson_riley": dict(xy_box=(0.03, 0.97), ha="left", va="top", relpos=(0.85, 0.0),
                                     conn="angle,angleA=90,angleB=0,rad=0"),
               "elementary": dict(xy_box=(0.97, 0.05), ha="right", va="bottom", relpos=(0.15, 1.0),
                                  conn="arc3,rad=0")}
        for site in RESIDENTIAL:
            mlat, mlon = mons[site]["lat"], mons[site]["lon"]
            ax.plot(mlon, mlat, zorder=8, **f1.RES_MON_STYLE)
            ppbc = meas["vc_JR" if site == "johnson_riley" else "vc_EL"]
            ug = ppbc_to_ugm3(ppbc)
            grid_bl = float(interp([[mlat, mlon]])[0])
            j = int(round((mlat - g["lats"][0]) / g["dlat"]))
            i = int(round((mlon - g["lons"][0]) / g["dlat"]))
            grid_cell = float(g["conc"][j, i])
            p = pred[site]
            rel = (grid_bl - p) / p if p > 0 else (0.0 if grid_bl == 0 else np.inf)
            assert abs(rel) <= MAX_REL_DIFF, f"{date} {site}: grid {grid_bl:.5g} vs predicted {p:.5g}"
            meas_txt = "ND" if ppbc == 0 else f"{fmt(ug)} µg/m³"
            txt = (f"{figstyle.MONITOR_SHORT[site]}\n"
                   f"Measured: {meas_txt}\n"
                   f"Modeled: {fmt(p)} µg/m³")
            a = ann[site]
            ax.annotate(txt, (mlon, mlat), xytext=a["xy_box"], textcoords="axes fraction", ha=a["ha"],
                        va=a["va"], fontsize=7, color=figstyle.INK, linespacing=1.15, multialignment="left",
                        zorder=9, bbox=dict(fc=figstyle.SURF, ec=figstyle.AXIS, lw=0.5, alpha=0.92, pad=2.5),
                        arrowprops=dict(arrowstyle="-", color=figstyle.INK2, lw=0.5, shrinkA=0, shrinkB=4,
                                        relpos=a["relpos"], connectionstyle=a["conn"]))
            rows.append(dict(panel="ab"[k], date=date, site=site, monitor=figstyle.MONITOR_LABEL[site],
                             lat=mlat, lon=mlon, measured_vc_ppbC=ppbc, measured_vc_ugm3=round(ug, 4),
                             measured_label="ND" if ppbc == 0 else fmt(ug),
                             predicted_clipped_ugm3=p, grid_bilinear_ugm3=round(grid_bl, 6),
                             grid_rel_diff=round(rel, 5), grid_nearest_node_ugm3=round(grid_cell, 6),
                             panel_max_node_ugm3=round(node_max, 4), panel_max_node_lat=round(g["lats"][jmax[0]], 4),
                             panel_max_node_lon=round(g["lons"][jmax[1]], 4), footprint_bearing_deg=round(bearing, 1),
                             n_runs=g["n_runs"], n_runs_with_vc=g["n_runs_vc"], n_hourly_periods=g["n_periods"],
                             selection_rule=rule[date]))
            print(f"   {site:<14} measured {ppbc:g} ppbC = {ug:.3f} ug/m3 | predicted_clipped {p:.5f} | "
                  f"grid bilinear {grid_bl:.5f} ({100 * rel:+.2f}%) | nearest node {grid_cell:.5f}")
        ax.set_title(f"{'ab'[k]}   {date}: {titles[k]}", fontsize=9, pad=4)

    assert vmax_seen < BOUNDS[-1], f"panel maximum {vmax_seen:.2f} exceeds the top bin; extend the colorbar"
    f1.scalebar(axes[1], ext["lon0"] + 0.012, ext["lat0"] + 0.012, 2, ticks=[0, 1, 2])   # same extent in a and b

    # shared colorbar and marker legend
    cax = fig.add_axes([(left + 0.35) / W, 0.50 / H, 3.3 / W, 0.11 / H])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=BOUNDS, spacing="uniform")
    cb.ax.set_xticklabels([f"{b:g}" for b in BOUNDS])
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(figstyle.AXIS)
    cb.ax.tick_params(length=2, width=0.4, color=figstyle.AXIS, labelsize=7.5)
    cb.set_label("Modeled daily-mean vinyl chloride, 0–10 m (µg/m³)", fontsize=8, color=figstyle.INK2, labelpad=3)
    vc_style = dict(f1.FAC_STYLE)
    other_style = dict(f1.FAC_STYLE, mfc=figstyle.SURF, mec=figstyle.INK, mew=0.7)
    leg = [Line2D([], [], label="Vinyl chloride source", **vc_style),
           Line2D([], [], label="Other modeled source", **other_style),
           Line2D([], [], label="Residential AQS monitor", **f1.RES_MON_STYLE)]
    fig.legend(handles=leg, loc="center left", bbox_to_anchor=((left + 4.05) / W, 0.58 / H), ncol=1,
               fontsize=7.5, handletextpad=0.3, labelspacing=0.35)

    figstyle.save(fig, "Fig6_plume_footprints", data=pd.DataFrame(rows))


if __name__ == "__main__":
    main()
