"""NOV2026_PLAN.md part C, measurements and monitor inventory (no forecast or weather is read here).

Inputs: raw/aqs_sites.zip, raw/aqs_monitors.zip, raw/extract_{HAPS,VOCS}_YYYY.csv (fetch_aqs_older.py).
Rules copied from run_extended.py:
  * 24-hour samples only; value = Arithmetic Mean; negatives clipped to 0 (as run_plume.py / score inputs do).
  * HAPS pollutants: mean over POCs for a (date, site, parameter) (run_extended.py pivot_table aggfunc="mean").
  * Total VOC: VOCS file, units "Parts per billion Carbon", parameters not starting with Total/Sum, summed.
    run_extended.py sums every row of a (date, site); with one POC per day that is the species sum. Where a site has
    collocated POCs on the same day (TVA Substation 2011-2019), summing every row would double the total, so the
    species sum is taken per POC and then averaged over POCs. Identical to run_extended.py when one POC reports.
Outputs: monitor_inventory.csv, older_monitor_days.csv.
"""
import glob
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
CLAT, CLON = 37.0486, -88.3480
CUTOFF = "2023-01-01"
POLL = {"Vinyl chloride": "vc", "Benzene": "benzene", "1,3-Butadiene": "butadiene", "Ethylene dichloride": "edc"}
# HYSPLIT's five modeled facilities (calvert_plume_engine.FACILITIES coordinates, the ones with plume runs)
MODELED = {"Westlake Vinyls": (37.04909631633214, -88.33221487571784),
           "Westlake PVC Plant": (37.04497026897765, -88.34947586035418),
           "Arkema Inc": (37.05626272127821, -88.36652857343046),
           "ISP Chemicals": (37.04736797, -88.35960388),
           "Lubrizol Advanced Materials": (37.04859077950765, -88.3317972924827)}
ALL_FACILITIES = dict(MODELED, **{
    "CC Metals & Alloys": (37.055574243875, -88.35030650294469),
    "Cymetech Corp": (37.049673766088034, -88.33079423057185),
    "Estron Chemicals": (37.04384046629804, -88.35594862550843),
    "Evonik Corp": (37.04256178133411, -88.34571245940829),
    "Sekisui Specialty Chemicals": (37.043812105671584, -88.35109849432436),
    "Wacker Chemical": (37.04472839351968, -88.3534802959446)})


def hav_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    y = np.sin((lon2 - lon1) * p) * np.cos(lat2 * p)
    x = np.cos(lat1 * p) * np.sin(lat2 * p) - np.sin(lat1 * p) * np.cos(lat2 * p) * np.cos((lon2 - lon1) * p)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def main():
    a = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(os.path.join(RAW, "extract_*_*.csv")))])
    a = a[(a["Sample Duration"] == "24 HOUR") & (a["Date Local"] < CUTOFF)].copy()
    a["site_id"] = a["State Code"] + "-" + a["County Code"] + "-" + a["Site Num"].str.zfill(4)
    a["val"] = pd.to_numeric(a["Arithmetic Mean"]).clip(lower=0)
    a["date"] = a["Date Local"]

    h = a[(a.file == "HAPS") & a["Parameter Name"].isin(POLL)]
    units = h.groupby("Parameter Name")["Units of Measure"].unique().to_dict()
    wide = (h.pivot_table(index=["date", "site_id"], columns="Parameter Name", values="val", aggfunc="mean")
            .rename(columns=POLL).reset_index())
    npoc = h.groupby(["date", "site_id"]).POC.nunique().rename("n_poc_haps").reset_index()
    v = a[(a.file == "VOCS") & (a["Units of Measure"] == "Parts per billion Carbon")
          & ~a["Parameter Name"].str.match(r"^(Total|Sum)")]
    per_poc = v.groupby(["date", "site_id", "POC"]).agg(s=("val", "sum"), n=("Parameter Name", "nunique")).reset_index()
    tv = per_poc.groupby(["date", "site_id"]).agg(total_voc=("s", "mean"), n_species=("n", "max"),
                                                  n_poc_vocs=("POC", "nunique")).reset_index()
    md = wide.merge(npoc, on=["date", "site_id"], how="left").merge(tv, on=["date", "site_id"], how="left")
    for k in POLL.values():
        if k not in md:
            md[k] = np.nan
        md[f"{k}_detected"] = np.where(md[k].isna(), np.nan, (md[k] > 0).astype(float))
    md["units"] = "ppbC (all four HAPS pollutants and total VOC)" if all(
        set(u) == {"Parts per billion Carbon"} for u in units.values()) else str(units)

    s = pd.read_csv(zipfile.ZipFile(os.path.join(RAW, "aqs_sites.zip")).open("aqs_sites.csv"), dtype=str, low_memory=False)
    s["site_id"] = s["State Code"] + "-" + s["County Code"] + "-" + s["Site Number"]
    s = s[s.site_id.isin(md.site_id.unique())].copy()
    s["lat"], s["lon"] = pd.to_numeric(s.Latitude), pd.to_numeric(s.Longitude)
    s["dist_complex_km"] = hav_km(CLAT, CLON, s.lat, s.lon)
    s["bearing_from_complex_deg"] = bearing(CLAT, CLON, s.lat, s.lon)
    for tag, fac in (("modeled", MODELED), ("any", ALL_FACILITIES)):
        dd = pd.DataFrame({k: hav_km(la, lo, s.lat, s.lon) for k, (la, lo) in fac.items()}, index=s.index)
        s[f"nearest_{tag}_facility"] = dd.idxmin(axis=1)
        s[f"nearest_{tag}_facility_km"] = dd.min(axis=1)
    cnt = md.groupby("site_id").agg(first_sample=("date", "min"), last_sample=("date", "max"),
                                    monitor_days=("date", "size"), vc_days=("vc", "count"), benzene_days=("benzene", "count"),
                                    vc_detect_rate=("vc_detected", "mean"), benzene_detect_rate=("benzene_detected", "mean"),
                                    total_voc_days=("total_voc", "count"))
    inv = s.set_index("site_id")[["Local Site Name", "Address", "Land Use", "Location Setting", "lat", "lon",
                                  "Site Established Date", "Site Closed Date", "dist_complex_km",
                                  "bearing_from_complex_deg", "nearest_modeled_facility", "nearest_modeled_facility_km",
                                  "nearest_any_facility", "nearest_any_facility_km"]].join(cnt).reset_index()
    inv = inv.rename(columns={"Local Site Name": "name", "Address": "address", "Land Use": "land_use",
                              "Location Setting": "location_setting", "Site Established Date": "established",
                              "Site Closed Date": "closed"})
    inv["in_plant_candidate"] = ""
    inv.sort_values("dist_complex_km").to_csv(os.path.join(HERE, "monitor_inventory.csv"), index=False)
    md = md.merge(inv[["site_id", "name"]], on="site_id", how="left")
    cols = ["date", "site_id", "name", "vc", "benzene", "edc", "butadiene", "total_voc", "n_species",
            "vc_detected", "benzene_detected", "edc_detected", "butadiene_detected", "n_poc_haps", "n_poc_vocs", "units"]
    md[cols].sort_values(["site_id", "date"]).to_csv(os.path.join(HERE, "older_monitor_days.csv"), index=False)
    print(inv.to_string())
    print("units:", units)
    yr = md.assign(year=md.date.str[:4]).groupby(["site_id", "year"]).agg(days=("date", "size"), vc=("vc", "count"),
                                                                           benzene=("benzene", "count"),
                                                                           total_voc=("total_voc", "count"),
                                                                           n_species=("n_species", "median"))
    yr.to_csv(os.path.join(HERE, "counts_by_monitor_year.csv"))
    print(yr.to_string())


if __name__ == "__main__":
    main()
