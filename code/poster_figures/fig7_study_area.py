"""Figure 6 (poster Figure 7; renumbered 3 Nov 2026): study area map, now with the air monitors. Black and gray only:
facilities are black squares, current residential monitors black circles, the in-complex monitor a gray circle and
earlier monitors hollow circles.

Panel a is the poster's map (make_assets.study_map: 32 census tracts filled by county, the petrochemical complex,
the TVA Shawnee plant, Calvert City). Panel b zooms on the complex: the five HYSPLIT-modeled facilities (release points
read from the HYSPLIT CONTROL files) and the three current EPA AQS VOC monitors (validation/config.py MONITOR_SITES).

Older monitors (hook): if EXTENDED/older_monitors/monitor_inventory.csv exists, every row with a latitude and longitude
is drawn as a hollow marker in both panels (re-run this script after that file is written).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import posterstyle as S

sys.path.insert(0, os.path.join(S.PKG, "04_code", "figures"))
import fig1_study_area as F1  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, Polygon, Rectangle  # noqa: E402

WS = "/path/to/weather-varaible-analysis"
OLDER = os.path.join(S.EXT, "older_monitors", "monitor_inventory.csv")
COUNTY_COLORS = {"145": "#DCEAF6", "157": "#F7F7F7", "139": "#B9D4EC"}   # white and two blue tints   # light grays; counties are also labeled
COUNTY_NAMES = {"145": "McCracken", "157": "Marshall", "139": "Livingston"}
COMPLEX = (37.0486, -88.3480)
ZOOM = dict(lon0=-88.395, lon1=-88.300, lat0=37.005, lat1=37.075)
MON_LABEL = {"elementary": "Calvert City Elementary\n(1.5 mi)", "johnson_riley": "Johnson-Riley Road\n(0.55 mi)",
             "lwd": "LWD (inside complex)"}


def older_monitors():
    if not os.path.exists(OLDER):
        print("no older-monitor inventory yet:", OLDER)
        return pd.DataFrame()
    o = pd.read_csv(OLDER)
    cols = {c.lower().replace(" ", "_"): c for c in o.columns}
    lat = next(cols[k] for k in ("latitude", "lat") if k in cols)
    lon = next(cols[k] for k in ("longitude", "lon") if k in cols)
    name = next((cols[k] for k in ("local_site_name", "site_name", "name") if k in cols), None)
    out = pd.DataFrame({"lat": pd.to_numeric(o[lat], errors="coerce"), "lon": pd.to_numeric(o[lon], errors="coerce"),
                        "name": o[name].astype(str) if name else ""}).dropna(subset=["lat", "lon"])
    return out.drop_duplicates(subset=["lat", "lon"])


def main():
    S.setup_modern()
    g = json.load(open(os.path.join(WS, "calvert_tracts.geojson")))
    fac = F1.hysplit_facilities()
    mons = F1.monitor_sites()
    old = older_monitors()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    rows = []

    # ---------------- a: the poster map
    for f in g["features"]:
        geom, c = f["geometry"], f["properties"]["COUNTY"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for a in (ax, bx):
                a.add_patch(Polygon(poly[0], closed=True, fc=COUNTY_COLORS.get(c, "#eee"), ec="#555555", lw=0.9))
    ax.autoscale_view()
    ax.set_aspect(1 / 0.8)
    ax.plot(-88.7739, 37.1533, marker="^", ms=13, color="#444", mec="black", mew=1.2, ls="none", zorder=6)
    ax.annotate("Calvert City", (-88.35, 37.033), xytext=(-14, -30), textcoords="offset points", fontsize=12,
                weight="bold", color=S.INK, ha="right", va="top", zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.8, shrinkB=5))
    ax.annotate("Paducah", (-88.60, 37.115), fontsize=11, color=S.GREY, ha="center")
    for c, nm in COUNTY_NAMES.items():
        feats = [f for f in g["features"] if f["properties"]["COUNTY"] == c]
        lat = np.mean([float(f["properties"]["CENTLAT"]) for f in feats])
        lon = np.mean([float(f["properties"]["CENTLON"]) for f in feats])
        ax.text(lon, lat, nm + " Co.", fontsize=11, color="#333333", ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.5))
    ax.add_patch(Rectangle((ZOOM["lon0"], ZOOM["lat0"]), ZOOM["lon1"] - ZOOM["lon0"], ZOOM["lat1"] - ZOOM["lat0"],
                           fc="none", ec=S.INK, lw=1.4, zorder=8))
    for ya, yb in ((ZOOM["lat1"], 1), (ZOOM["lat0"], 0)):
        fig.add_artist(ConnectionPatch((ZOOM["lon1"], ya), (0, yb), coordsA="data", coordsB="axes fraction",
                                       axesA=ax, axesB=bx, color=S.GREY, lw=0.8))
    # panel a shows only the setting; the complex, facilities and monitors are drawn in panel b (zoom box)
    leg = [Line2D([], [], marker="^", ms=12, color="#444", mec="black", ls="none", label="TVA Shawnee plant"),
           Line2D([], [], marker="s", ms=13, mfc="none", mec=S.INK, mew=1.4, ls="none",
                  label="Complex and monitors (panel b)")]
    ax.legend(handles=leg, loc="lower left", fontsize=11, frameon=True)
    ax.set_title("a.  Study area: 32 census tracts around Calvert City, KY", loc="left", fontsize=14, weight="bold",
                 color=S.INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # ---------------- b: zoom on the complex
    for _, f in fac.iterrows():
        bx.plot(f.lon, f.lat, marker="s", ms=11, color=S.BLACK, mec="black", mew=1.0, ls="none", zorder=6)
        rows.append(dict(type="HYSPLIT facility", name=f["name"], lat=f.lat, lon=f.lon, releases=f.releases,
                         species=f.species))
    lab = {"elementary": ((14, -6), "left", "top"), "johnson_riley": ((6, -14), "center", "top"),
           "lwd": ((-8, 12), "right", "bottom")}
    fac_lab = {"arkema_inc": ("Arkema", (0, 11), "center", "bottom"),
               "isp_chemicals": ("ISP Chemicals", (-11, 0), "right", "center"),
               "westlake_pvc_plant": ("Westlake PVC", (11, 3), "left", "bottom"),
               "westlake_vinyls": ("Westlake Vinyls\nand Lubrizol", (11, 6), "left", "bottom")}
    for _, f in fac.iterrows():
        if f.slug in fac_lab:
            t, off, ha, va = fac_lab[f.slug]
            bx.annotate(t, (f.lon, f.lat), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=10.5,
                        color=S.INK, weight="bold", zorder=9, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    for s, m in mons.items():
        col = S.ORI_COL if s in S.RES else S.GRAY_MID      # community monitors in the paper's blue
        bx.plot(m["lon"], m["lat"], marker="o", ms=13, color=col, mec="black", mew=1.3, ls="none", zorder=8)
        off, ha, va = lab[s]
        bx.annotate(MON_LABEL[s], (m["lon"], m["lat"]), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=11, weight="bold", color=S.INK, zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        rows.append(dict(type="AQS VOC monitor", name=m["name"], aqs_site=f"21-157-00{int(m['aqs_site']):02d}",
                         lat=m["lat"], lon=m["lon"]))
    if len(old):
        o = old[(old.lon.between(ZOOM["lon0"], ZOOM["lon1"])) & (old.lat.between(ZOOM["lat0"], ZOOM["lat1"]))]
        bx.plot(o.lon, o.lat, marker="o", ms=11, mfc="none", mec=S.INK, mew=1.6, ls="none", zorder=8)
        old_lab = {"ATMOS ENERGY": ("Atmos Energy\n(2011-2016)", (-12, -2), "right", "center"),
                   "TVA SUBSTATION": ("TVA Substation\n(2011-2019)", (10, -4), "left", "top"),
                   "LAZY DAZ": ("Lazy Daz\n(2011-2014)", (10, 0), "left", "center"),
                   "BLOODWORTH": ("Bloodworth\n(2011-2019)", (10, 0), "left", "center")}
        for _, r in o.iterrows():
            if r["name"].upper() in old_lab:
                t, off, ha, va = old_lab[r["name"].upper()]
                bx.annotate(t, (r.lon, r.lat), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=10,
                            color=S.GREY, zorder=9, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
        for _, r in old.iterrows():
            rows.append(dict(type="earlier AQS VOC monitor (before 2023)", name=r["name"], lat=r.lat, lon=r.lon))
    bx.set_xlim(ZOOM["lon0"], ZOOM["lon1"])
    bx.set_ylim(ZOOM["lat0"], ZOOM["lat1"])
    bx.set_aspect(1 / 0.8)
    bx.set_xticks([])
    bx.set_yticks([])
    for s in bx.spines.values():
        s.set_visible(True)
        s.set_color(S.INK)
        s.set_linewidth(1.4)
    leg = [Line2D([], [], marker="s", ms=11, color=S.BLACK, mec="black", ls="none", label="Modeled facility (HYSPLIT)"),
           Line2D([], [], marker="o", ms=12, color=S.ORI_COL, mec="black", ls="none", label="Community AQS monitor"),
           Line2D([], [], marker="o", ms=12, color=S.GRAY_MID, mec="black", ls="none", label="In-complex AQS monitor")]
    if len(old):
        leg.append(Line2D([], [], marker="o", ms=11, mfc="none", mec=S.INK, mew=1.6, ls="none",
                          label="Earlier VOC monitor (2011-2022)"))
    bx.legend(handles=leg, loc="lower right", fontsize=11, frameon=True)
    F1.scalebar(bx, ZOOM["lon0"] + 0.006, ZOOM["lat0"] + 0.004, 1, ticks=[0, 0.5, 1])
    bx.set_title("b.  The complex, modeled facilities, and air monitors", loc="left", fontsize=14, weight="bold",
                 color=S.INK)
    fig.tight_layout(w_pad=4)
    S.save(fig, "Fig1_study_area_map", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
