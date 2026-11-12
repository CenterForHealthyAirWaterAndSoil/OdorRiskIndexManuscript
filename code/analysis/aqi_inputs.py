"""Inputs for the AQI comparison (AQI_COMPARISON_PLAN.md, DEVIATIONS 30-31). Nothing here touches the VOC data.

- EPA AirData daily AQI by county, daily_88101 (PM2.5) and daily_44201 (ozone), 2023-2025: kept as downloaded zips.
- Open-Meteo air-quality API, domains=cams_global, hourly US AQI at each monitor's coordinates: raw JSON kept.

Output aqi_inputs/aqi_daily.csv: one row per monitor-date with A1 (McCracken County), A1 components and A2
(Smithland ozone) as reported, and M1 (modelled, daily maximum over complete local days) at that monitor. Missing
stays missing. Run: ~/weather-varaible-analysis/.venv/bin/python aqi_inputs.py
"""
import datetime as dt
import gzip
import hashlib
import json
import os
import time
import zipfile

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "aqi_inputs")
YEARS = [2023, 2024, 2025]
AIRDATA = "https://aqs.epa.gov/aqsweb/airdata"
SITES = {"elementary": (37.026757, -88.343757), "johnson_riley": (37.041202, -88.351879)}  # as in score_refit.py
TZ = "America/Chicago"
A1_SITE, A2_SITE = (21, 145, 1027), (21, 139, 3)
HOURLY = ["us_aqi", "us_aqi_pm2_5", "us_aqi_ozone"]


def fetch(url, path):
    if not os.path.exists(path):
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    return path


def airdata(name, year):
    z = fetch(f"{AIRDATA}/{name}_{year}.zip", os.path.join(OUT, f"{name}_{year}.zip"))
    with zipfile.ZipFile(z) as f:
        return pd.read_csv(f.open(f.namelist()[0]), low_memory=False)


def site_rows(d, site):
    st, co, si = site
    return d[(d["State Code"].astype(str) == str(st)) & (d["County Code"] == co) & (d["Site Num"] == si)]


def observed():
    county = pd.concat([airdata("daily_aqi_by_county", y) for y in YEARS])
    a1 = county[(county["State Name"] == "Kentucky") & (county["county Name"] == "McCracken")]
    out = a1.set_index("Date")[["AQI"]].rename(columns={"AQI": "a1_aqi"})
    for code, col in [("88101", "a1_pm25_aqi"), ("44201", "a1_ozone_aqi")]:
        d = pd.concat([airdata(f"daily_{code}", y) for y in YEARS])
        out = out.join(site_rows(d, A1_SITE).groupby("Date Local").AQI.max().rename(col), how="outer")
        if code == "44201":
            out = out.join(site_rows(d, A2_SITE).groupby("Date Local").AQI.max().rename("a2_ozone_aqi"), how="outer")
    return out.rename_axis("date")


def modelled(site, lat, lon):
    path = os.path.join(OUT, f"cams_global_{site}.json.gz")
    if not os.path.exists(path):
        url = ("https://air-quality-api.open-meteo.com/v1/air-quality"
               f"?latitude={lat}&longitude={lon}&hourly={','.join(HOURLY)}"
               f"&start_date={YEARS[0] - 1}-12-31&end_date={YEARS[-1] + 1}-01-01&timezone=GMT&domains=cams_global")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with gzip.open(path, "wt") as f:
            json.dump({"url": url, "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "data": r.json()}, f)
        time.sleep(2)
    with gzip.open(path, "rt") as f:
        rec = json.load(f)["data"]
    h = pd.DataFrame(rec["hourly"])
    loc = pd.to_datetime(h.pop("time")).dt.tz_localize("UTC").dt.tz_convert(TZ)
    h["date"] = loc.dt.strftime("%Y-%m-%d")
    n = h.groupby("date").size()
    ok = h[HOURLY].notna().all(axis=1).groupby(h.date).all()
    complete = set(ok[ok].index) & set(n[(n >= 23) & (n <= 25)].index)
    day = h[h.date.isin(complete)].groupby("date")[HOURLY].max()
    day.columns = ["m1_aqi", "m1_pm25_aqi", "m1_ozone_aqi"]
    return day.assign(site=site, cams_lat=rec["latitude"], cams_lon=rec["longitude"])


def main():
    os.makedirs(OUT, exist_ok=True)
    obs = observed()
    rows = []
    for site, (lat, lon) in SITES.items():
        m = modelled(site, lat, lon)
        rows.append(m.join(obs, how="outer").assign(site=site))
    out = pd.concat(rows).rename_axis("date").reset_index()
    out = out[(out.date >= f"{YEARS[0]}-01-01") & (out.date <= f"{YEARS[-1]}-12-31")]
    out.to_csv(os.path.join(OUT, "aqi_daily.csv"), index=False)
    county = pd.concat([airdata("daily_aqi_by_county", y) for y in YEARS])
    ctx = {"us_counties_with_any_daily_aqi_2023_2025": int(county[["State Code", "County Code"]].drop_duplicates().shape[0]),
           "ky_counties_with_any_daily_aqi_2023_2025": int(county[county["State Name"] == "Kentucky"]["county Name"].nunique()),
           "cams_grid_points": {s: [float(g.cams_lat.iloc[0]), float(g.cams_lon.iloc[0])] for s, g in out.dropna(subset=["cams_lat"]).groupby("site")}}
    json.dump(ctx, open(os.path.join(OUT, "context.json"), "w"), indent=1)
    with open(os.path.join(OUT, "MANIFEST_sha256.txt"), "w") as f:
        for name in sorted(os.listdir(OUT)):
            if name != "MANIFEST_sha256.txt":
                f.write(f"{hashlib.sha256(open(os.path.join(OUT, name), 'rb').read()).hexdigest()}  {name}\n")
    print(json.dumps(ctx, indent=1))
    print(out.groupby("site")[["m1_aqi", "a1_aqi", "a1_pm25_aqi", "a1_ozone_aqi", "a2_ozone_aqi"]].count())


if __name__ == "__main__":
    main()
