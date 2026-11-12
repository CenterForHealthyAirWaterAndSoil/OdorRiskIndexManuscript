"""Download EPA AQS pre-generated daily HAPS and VOCS files (2011-2022) and keep only 24-hour rows from monitors
within 10 km of the Calvert complex centre (NOV2026_PLAN.md part C). Zips are deleted after extraction; URL, size and
SHA-256 of each are logged to raw/download_log.csv."""
import os, sys, zipfile, hashlib, subprocess, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "raw")
CLAT, CLON = 37.0486, -88.3480
s = pd.read_csv(zipfile.ZipFile(os.path.join(RAW, "aqs_sites.zip")).open("aqs_sites.csv"), dtype=str, low_memory=False)
p = np.pi / 180
lat, lon = pd.to_numeric(s.Latitude), pd.to_numeric(s.Longitude)
d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat - CLAT) * p / 2) ** 2 + np.cos(CLAT * p) * np.cos(lat * p) * np.sin((lon - CLON) * p / 2) ** 2))
near = s[d <= 10]
keys = set(zip(near["State Code"], near["County Code"], near["Site Number"]))
log = []
years = [int(y) for y in sys.argv[1:]] or list(range(2011, 2023))
for kind in ["HAPS", "VOCS"]:
    for y in years:
        out = os.path.join(RAW, f"extract_{kind}_{y}.csv")
        if os.path.exists(out): continue
        url = f"https://aqs.epa.gov/aqsweb/airdata/daily_{kind}_{y}.zip"; zp = os.path.join(RAW, f"daily_{kind}_{y}.zip")
        r = subprocess.run(["curl", "-sS", "-f", "-o", zp, url])
        if r.returncode != 0:
            log.append({"url": url, "status": "not available"}); continue
        sha = hashlib.sha256(open(zp, "rb").read()).hexdigest(); size = os.path.getsize(zp)
        z = zipfile.ZipFile(zp); df = pd.read_csv(z.open(z.namelist()[0]), dtype=str, low_memory=False)
        k = list(zip(df["State Code"], df["County Code"], df["Site Num"]))
        df = df[[x in keys for x in k]]
        df = df.assign(file=kind)
        df.to_csv(out, index=False)
        os.remove(zp)
        log.append({"url": url, "status": "ok", "bytes": size, "sha256": sha, "rows_kept": len(df),
                    "rows_24h": int((df["Sample Duration"] == "24 HOUR").sum())})
        print(kind, y, size, len(df), flush=True)
lp = os.path.join(RAW, "download_log.csv")
old = pd.read_csv(lp) if os.path.exists(lp) else pd.DataFrame()
pd.concat([old, pd.DataFrame(log)]).to_csv(lp, index=False)
