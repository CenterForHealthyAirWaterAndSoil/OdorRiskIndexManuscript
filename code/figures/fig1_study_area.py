"""Figure 1. Study area: the three counties (census tracts) and the Calvert City industrial complex.

Panel a: McCracken, Marshall and Livingston counties, KY, with census-tract outlines, Paducah, Calvert City, and a
         locator of Kentucky.
Panel b: the industrial complex: the HYSPLIT source facilities exactly as modeled (coordinates read from the HYSPLIT
         CONTROL files of the finished back-test runs), the three AQS monitors, and Calvert City.

No basemap is downloaded. River and dam geometry is not available offline, so neither is drawn.

Inputs (read-only):
  tracts      /path/to/weather-varaible-analysis/calvert_tracts.geojson
  monitors    /path/to/plume-validation/validation/config.py (MONITOR_SITES)
  facilities  /path/to/plume-validation/validation/runs_clipped/<date>/<facility>_<stack|fugitive>/CONTROL
              (only dates with a .DONE marker in results/predicted_clipped)
  US states   /path/to/hysplit/graphics/states.shp (locator only)
Output: 06_figures/main/Fig1_study_area.{png,pdf} and Fig1_study_area_data.csv (every plotted point).

Run: /path/to/weather-varaible-analysis/.venv/bin/python 04_code/figures/fig1_study_area.py
"""
import glob
import json
import math
import os
import struct
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (figstyle selects the Agg backend first)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, Polygon, Rectangle  # noqa: E402

# ---------------------------------------------------------------- inputs
TRACTS = "/path/to/weather-varaible-analysis/calvert_tracts.geojson"
VALIDATION = "/path/to/plume-validation/validation"
RUNS = os.path.join(VALIDATION, "runs_clipped")
PRED = os.path.join(VALIDATION, "results", "predicted_clipped")
STATES_SHP = "/path/to/hysplit/graphics/states.shp"

COUNTY_NAME = {"145": "McCracken", "157": "Marshall", "139": "Livingston"}
FACILITY_NAME = {"arkema_inc": "Arkema Inc", "isp_chemicals": "ISP Chemicals",
                 "lubrizol_advanced_materials": "Lubrizol Advanced Materials",
                 "westlake_pvc_plant": "Westlake PVC Plant", "westlake_vinyls": "Westlake Vinyls"}
# Approximate town centers, hand-entered from a public gazetteer (rounded to 1e-4 deg); not in any project input.
CITIES = {"Paducah": (37.0834, -88.6000), "Calvert City": (37.0334, -88.3500)}

# Panel b extent: the complex plus all three monitors. Its latitude span is set in main() so that panel b has the
# same physical aspect ratio as panel a.
ZOOM_LON = (-88.383, -88.315)
ZOOM_LAT_CENTER = 37.0405

FAC_STYLE = dict(marker="s", ms=5.5, mfc=figstyle.INK, mec=figstyle.SURF, mew=0.6, ls="none")
RES_MON_STYLE = dict(marker="o", ms=6.5, mfc=figstyle.SURF, mec=figstyle.INK, mew=1.1, ls="none")
LWD_STYLE = dict(marker="o", ms=6.5, mfc=figstyle.MUTED, mec=figstyle.INK, mew=0.6, ls="none")
CITY_STYLE = dict(marker="o", ms=3.5, mfc=figstyle.INK2, mec=figstyle.INK2, ls="none")


# ---------------------------------------------------------------- helpers
def km_per_deg(lat):
    """WGS84 lengths (km) of one degree of latitude and of longitude at latitude `lat`."""
    a, e2 = 6378.137, 0.00669438
    s2 = math.sin(math.radians(lat)) ** 2
    dlat = math.pi / 180 * a * (1 - e2) / (1 - e2 * s2) ** 1.5
    dlon = math.pi / 180 * a * math.cos(math.radians(lat)) / math.sqrt(1 - e2 * s2)
    return dlat, dlon


def set_geo_aspect(ax, lat):
    dlat, dlon = km_per_deg(lat)
    ax.set_aspect(dlat / dlon)


def scalebar(ax, lon, lat, miles, ticks=None):
    """Horizontal scale bar starting at (lon, lat), length `miles`, labeled in miles."""
    dlon_per_mile = 1.609344 / km_per_deg(lat)[1]
    ticks = ticks or [0, miles]
    x = [lon + t * dlon_per_mile for t in ticks]
    ax.plot([x[0], x[-1]], [lat, lat], color=figstyle.INK, lw=1.2, solid_capstyle="butt", zorder=6)
    ylim = ax.get_ylim()
    h = (ylim[1] - ylim[0]) * 0.008
    for xi, t in zip(x, ticks):
        ax.plot([xi, xi], [lat - h, lat + h], color=figstyle.INK, lw=0.8, zorder=6)
        ax.text(xi, lat + 2.2 * h, f"{t:g}", ha="center", va="bottom", fontsize=7, color=figstyle.INK2, zorder=6)
    ax.text(x[-1] + 0.05 * (x[-1] - x[0]), lat, "mi", ha="left", va="center", fontsize=7,
            color=figstyle.INK2, zorder=6)


def north_arrow(ax, x, y, size=0.06):
    """North arrow in axes-fraction coordinates (north is up: plain lon/lat axes)."""
    ax.annotate("", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5", color=figstyle.INK, lw=0.9))
    ax.text(x, y + 0.01, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
            color=figstyle.INK, fontweight="bold")


def map_axes(ax):
    figstyle.clean(ax, grid=None)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():          # a hairline frame for map panels
        s.set_visible(True)
        s.set_color(figstyle.AXIS)
        s.set_linewidth(0.6)


def read_tracts():
    g = json.load(open(TRACTS))
    out = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        out.append((f["properties"]["COUNTY"], [np.asarray(p[0]) for p in polys], f["properties"]))
    return out


def county_outlines(tracts):
    """Dissolve tracts to county outlines with shapely (unary_union)."""
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    out = {}
    for c in COUNTY_NAME:
        geoms = [SPoly(r) for cc, rings, _ in tracts if cc == c for r in rings]
        u = unary_union(geoms)
        parts = list(u.geoms) if hasattr(u, "geoms") else [u]
        out[c] = (u, [np.asarray(p.exterior.coords) for p in parts])
    return out


def read_shp_polygons(path):
    """Minimal ESRI .shp polygon reader (type 5): list of records, each a list of (n, 2) lon/lat rings."""
    b = open(path, "rb").read()
    assert struct.unpack(">i", b[:4])[0] == 9994 and struct.unpack("<i", b[32:36])[0] == 5
    pos, recs = 100, []
    while pos < len(b):
        _, clen = struct.unpack(">2i", b[pos:pos + 8])
        c = b[pos + 8:pos + 8 + 2 * clen]
        pos += 8 + 2 * clen
        if struct.unpack("<i", c[:4])[0] == 0:
            continue
        nparts, npts = struct.unpack("<2i", c[36:44])
        parts = list(struct.unpack(f"<{nparts}i", c[44:44 + 4 * nparts])) + [npts]
        pts = np.frombuffer(c[44 + 4 * nparts:44 + 4 * nparts + 16 * npts], dtype="<f8").reshape(-1, 2)
        recs.append([pts[parts[k]:parts[k + 1]] for k in range(nparts)])
    return recs


def done_dates():
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PRED, "*.DONE")))


def hysplit_facilities():
    """Facility release points exactly as written to the HYSPLIT CONTROL files (line 3: lat lon height).
    Reads every finished (.DONE) date and requires the coordinates to be identical across dates."""
    seen = {}
    dates = done_dates()
    assert dates, "no finished HYSPLIT dates found"
    for d in dates:
        for ctl in sorted(glob.glob(os.path.join(RUNS, d, "*", "CONTROL"))):
            run = os.path.basename(os.path.dirname(ctl))
            slug, release = run.rsplit("_", 1)
            lines = open(ctl).read().split("\n")
            lat, lon, hgt = (float(v) for v in lines[2].split())
            npol = int(lines[9])
            species = [lines[10 + 4 * k].strip() for k in range(npol)]
            key = (slug, release)
            rec = (lat, lon, hgt)
            if key in seen:
                assert seen[key][0] == rec, f"{key} location differs between dates: {seen[key][0]} vs {rec}"
                seen[key][1].update(species)
            else:
                seen[key] = (rec, set(species))
    rows = []
    for slug in sorted({k[0] for k in seen}):
        recs = {rel: seen[(slug, rel)] for rel in ("stack", "fugitive") if (slug, rel) in seen}
        locs = {v[0][:2] for v in recs.values()}
        assert len(locs) == 1, f"{slug}: stack and fugitive release points differ: {locs}"
        lat, lon = locs.pop()
        species = sorted(set().union(*[v[1] for v in recs.values()]))
        rows.append(dict(slug=slug, name=FACILITY_NAME[slug], lat=lat, lon=lon,
                         releases="; ".join(f"{rel} {recs[rel][0][2]:g} m" for rel in recs),
                         species=" ".join(species)))
    print(f"HYSPLIT facilities from CONTROL files of {len(dates)} finished dates:")
    for r in rows:
        print(f"  {r['name']:<28} {r['lat']:.4f} {r['lon']:.4f}  {r['releases']:<22} {r['species']}")
    return pd.DataFrame(rows)


def monitor_sites():
    """MONITOR_SITES from validation/config.py, read as text (ast) so nothing is imported or cached there."""
    import ast
    tree = ast.parse(open(os.path.join(VALIDATION, "config.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MONITOR_SITES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError("MONITOR_SITES not found in validation/config.py")


# ---------------------------------------------------------------- figure
def main():
    from shapely.geometry import Point, Polygon as SPoly
    figstyle.setup()
    tracts = read_tracts()
    counties = county_outlines(tracts)
    fac = hysplit_facilities()
    mons = monitor_sites()
    points = []   # every plotted point -> data CSV

    # extents and a shared physical aspect ratio (height / width) for the two map panels
    xs = np.concatenate([r[:, 0] for _, (u, rr) in counties.items() for r in rr])
    ys = np.concatenate([r[:, 1] for _, (u, rr) in counties.items() for r in rr])
    pad = 0.02
    a_lon, a_lat = (xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad)
    kla, klo = km_per_deg(np.mean(a_lat))
    ratio = (a_lat[1] - a_lat[0]) * kla / ((a_lon[1] - a_lon[0]) * klo)
    kla_b, klo_b = km_per_deg(ZOOM_LAT_CENTER)
    half = 0.5 * ratio * (ZOOM_LON[1] - ZOOM_LON[0]) * klo_b / kla_b
    zoom = dict(lon0=ZOOM_LON[0], lon1=ZOOM_LON[1], lat0=ZOOM_LAT_CENTER - half, lat1=ZOOM_LAT_CENTER + half)

    W, pw, gap, left, bottom = 7.2, 3.42, 0.30, 0.03, 0.42
    ph = pw * ratio
    H = bottom + ph + 0.30
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, pw / W, ph / H])
    bx = fig.add_axes([(left + pw + gap) / W, bottom / H, pw / W, ph / H])

    # ---------------- panel a: counties and tracts
    for c, rings, props in tracts:
        for r in rings:
            ax.add_patch(Polygon(r, closed=True, fc="none", ec=figstyle.AXIS, lw=0.35, zorder=2))
    for c, (u, rings) in counties.items():
        for r in rings:
            ax.add_patch(Polygon(r, closed=True, fc="#f6f5f1", ec="none", zorder=1))
            ax.add_patch(Polygon(r, closed=True, fc="none", ec=figstyle.INK2, lw=0.9, zorder=3))
    ax.set_xlim(*a_lon)
    ax.set_ylim(*a_lat)
    set_geo_aspect(ax, np.mean(a_lat))
    map_axes(ax)
    label_pos = {"145": (-88.745, 36.985), "157": (-88.285, 36.905), "139": (-88.300, 37.245)}
    for c, (lon, lat) in label_pos.items():
        assert counties[c][0].contains(Point(lon, lat)), f"label for county {c} falls outside it"
        ax.text(lon, lat, f"{COUNTY_NAME[c]}\nCounty", ha="center", va="center", fontsize=8, style="italic",
                color=figstyle.INK2, linespacing=1.0, zorder=5)
    city_off = {"Paducah": ((-5, 4), "right", "bottom"), "Calvert City": ((0, -12), "center", "top")}
    for nm, (lat, lon) in CITIES.items():
        ax.plot(lon, lat, zorder=8, **CITY_STYLE)
        off, ha, va = city_off[nm]
        ax.annotate(nm, (lon, lat), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=8,
                    color=figstyle.INK, zorder=8, bbox=dict(fc="#f6f5f1", ec="none", pad=0.4, alpha=0.9))
        points.append(dict(name=nm, type="town center", lat=lat, lon=lon, panel="a",
                           source="hand-entered approximate town center (public gazetteer)"))
    ax.add_patch(Rectangle((zoom["lon0"], zoom["lat0"]), zoom["lon1"] - zoom["lon0"], zoom["lat1"] - zoom["lat0"],
                           fc="none", ec=figstyle.INK, lw=0.9, zorder=7))
    for (xa, ya), (xb, yb) in [((zoom["lon1"], zoom["lat1"]), (0, 1)), ((zoom["lon1"], zoom["lat0"]), (0, 0))]:
        fig.add_artist(ConnectionPatch((xa, ya), (xb, yb), coordsA="data", coordsB="axes fraction",
                                       axesA=ax, axesB=bx, color=figstyle.MUTED, lw=0.5, zorder=0))
    scalebar(ax, a_lon[0] + 0.04, a_lat[0] + 0.035, 10, ticks=[0, 5, 10])
    north_arrow(ax, 0.94, 0.90)

    # locator: Kentucky and neighbors, upper left
    loc = ax.inset_axes([0.015, 0.70, 0.36, 0.28])
    states = read_shp_polygons(STATES_SHP)
    ky = [rec for rec in states if any(len(r) > 3 and SPoly(r).contains(Point(-84.87, 38.20)) for r in rec)]
    assert len(ky) == 1, "Kentucky not found uniquely in states.shp"
    for rec in states:
        for r in rec:
            if r[:, 0].max() < -92 or r[:, 0].min() > -80 or r[:, 1].max() < 35 or r[:, 1].min() > 40.5:
                continue
            loc.add_patch(Polygon(r, closed=True, fc="none", ec=figstyle.AXIS, lw=0.4))
    for r in ky[0]:
        loc.add_patch(Polygon(r, closed=True, fc=figstyle.SURF, ec=figstyle.INK2, lw=0.6))
    for c, (u, rings) in counties.items():
        for r in rings:
            loc.add_patch(Polygon(r, closed=True, fc=figstyle.INK, ec="none"))
    loc.set_xlim(-89.8, -81.7)
    loc.set_ylim(36.3, 39.35)
    set_geo_aspect(loc, 37.8)
    loc.set_xticks([])
    loc.set_yticks([])
    loc.set_facecolor(figstyle.SURF)
    for s in loc.spines.values():
        s.set_color(figstyle.AXIS)
        s.set_linewidth(0.6)
    loc.text(-85.3, 37.55, "Kentucky", ha="center", va="center", fontsize=6.5, color=figstyle.INK2)
    ax.set_title("a   McCracken, Marshall and Livingston counties, KY", fontsize=9, pad=4)

    # ---------------- panel b: complex zoom
    for c, rings, props in tracts:
        for r in rings:
            bx.add_patch(Polygon(r, closed=True, fc="none", ec=figstyle.AXIS, lw=0.5, zorder=1))
    for c, (u, rings) in counties.items():
        for r in rings:
            bx.add_patch(Polygon(r, closed=True, fc="none", ec=figstyle.INK2, lw=0.9, zorder=2))
    bx.set_xlim(zoom["lon0"], zoom["lon1"])
    bx.set_ylim(zoom["lat0"], zoom["lat1"])
    set_geo_aspect(bx, ZOOM_LAT_CENTER)
    map_axes(bx)
    for nm, (lon, lat), c in [("Livingston Co.", (-88.3175, 37.0650), "139"),
                              ("Marshall Co.", (-88.3175, 37.0150), "157")]:
        assert counties[c][0].contains(Point(lon - 0.004, lat)), f"{nm} label is outside the county"
        bx.text(lon, lat, nm, ha="right", va="center", fontsize=7, style="italic", color=figstyle.INK2)
    bx.text(-88.3690, 37.0275, "CALVERT CITY", ha="center", va="center", fontsize=7, color=figstyle.MUTED,
            fontweight="bold")

    # facilities (Westlake Vinyls and Lubrizol are ~60 m apart, so each gets a leader line)
    fac_label = {"arkema_inc": ((-88.3712, 37.0612), "center", "bottom"),
                 "isp_chemicals": ((-88.3712, 37.0474), "right", "center"),
                 "westlake_pvc_plant": ((-88.3500, 37.0545), "center", "bottom"),
                 "westlake_vinyls": ((-88.3270, 37.0555), "center", "bottom"),
                 "lubrizol_advanced_materials": ((-88.3270, 37.0418), "center", "top")}
    fac_text = {"lubrizol_advanced_materials": "Lubrizol Advanced\nMaterials"}
    for _, f in fac.iterrows():
        bx.plot(f.lon, f.lat, zorder=6, **FAC_STYLE)
        (tx, ty), ha, va = fac_label[f.slug]
        bx.annotate(fac_text.get(f.slug, f["name"]), (f.lon, f.lat), xytext=(tx, ty), textcoords="data",
                    ha=ha, va=va, fontsize=7, color=figstyle.INK, linespacing=1.0, zorder=7,
                    arrowprops=dict(arrowstyle="-", color=figstyle.INK2, lw=0.5, shrinkA=1.5, shrinkB=3))
        points.append(dict(name=f["name"], type="HYSPLIT source facility", lat=f.lat, lon=f.lon, panel="b",
                           source=f"HYSPLIT CONTROL ({f.releases}; {f.species})"))

    mon_label = {"elementary": ((7, 0), "left", "center"), "johnson_riley": ((-7, -2), "right", "center"),
                 "lwd": ((0, -7), "center", "top")}
    for key, m in mons.items():
        style = LWD_STYLE if key == "lwd" else RES_MON_STYLE
        bx.plot(m["lon"], m["lat"], zorder=8, **style)
        (dx, dy), ha, va = mon_label[key]
        bx.annotate(figstyle.MONITOR_LABEL[key].replace(" (", "\n("), (m["lon"], m["lat"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=7, color=figstyle.INK, zorder=8,
                    linespacing=1.0, multialignment=ha if ha != "center" else "center",
                    bbox=dict(fc=figstyle.SURF, ec="none", alpha=0.85, pad=0.6))
        points.append(dict(name=m["name"], type="AQS monitor" + (" (inside complex)" if key == "lwd" else
                                                                 " (residential)"),
                           lat=m["lat"], lon=m["lon"], panel="b",
                           source=f"validation/config.py MONITOR_SITES (AQS site {m['aqs_site']})"))
    scalebar(bx, zoom["lon0"] + 0.0035, zoom["lat0"] + 0.0028, 1, ticks=[0, 0.5, 1])
    bx.set_title("b   Industrial complex and AQS monitors", fontsize=9, pad=4)

    leg = [Line2D([], [], label="HYSPLIT source facility", **FAC_STYLE),
           Line2D([], [], label="Residential AQS monitor", **RES_MON_STYLE),
           Line2D([], [], label="AQS monitor inside the complex", **LWD_STYLE),
           Line2D([], [], label="Town center", **CITY_STYLE)]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7.5,
               handletextpad=0.3, columnspacing=1.6)

    for nm, (lat, lon) in {"panel b extent, SW corner": (zoom["lat0"], zoom["lon0"]),
                           "panel b extent, NE corner": (zoom["lat1"], zoom["lon1"])}.items():
        points.append(dict(name=nm, type="zoom box corner", lat=round(lat, 5), lon=lon, panel="a",
                           source="figure layout"))
    df = pd.DataFrame(points)[["name", "type", "lat", "lon", "panel", "source"]]
    figstyle.save(fig, "Fig1_study_area", data=df)


if __name__ == "__main__":
    main()
