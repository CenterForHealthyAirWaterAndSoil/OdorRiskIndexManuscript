"""Corrected total VOC for 2023-2025 (DEVIATIONS 38a).

Sum of the ppbC species reported in every year 2023-2025 (drops acetylene and propylene, reported only in 2023),
summed within each canister (POC) and then averaged over collocated canisters on the same day.
Input: calvert_aqs_extract.csv. Output: total_voc_fixed.csv (date, site, total_voc_fixed, n_species_fixed, n_canisters).
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = {"18": "elementary", "20": "johnson_riley", "21": "lwd"}

a = pd.read_csv(os.path.join(HERE, "calvert_aqs_extract.csv"), dtype=str)
v = a[(a.file == "VOCS") & (a["Units of Measure"] == "Parts per billion Carbon")
      & ~a["Parameter Name"].str.match(r"^(Total|Sum)") & (a["Sample Duration"] == "24 HOUR")].copy()
v["val"] = pd.to_numeric(v["Arithmetic Mean"]).clip(lower=0)
v["yr"] = v["Date Local"].str[:4]
years = sorted(v.yr.unique())
assert years == ["2023", "2024", "2025"], years
every_year = v.groupby("Parameter Name").yr.nunique()
keep = sorted(every_year[every_year == len(years)].index)
dropped = sorted(set(v["Parameter Name"]) - set(keep))
assert dropped == ["Acetylene", "Propylene"], dropped
v = v[v["Parameter Name"].isin(keep)]
v["site"] = v["Site Num"].astype(int).astype(str).map(SITE)
per_can = v.groupby(["Date Local", "site", "POC"]).agg(tv=("val", "sum"), n=("Parameter Name", "nunique")).reset_index()
out = per_can.groupby(["Date Local", "site"]).agg(total_voc_fixed=("tv", "mean"), n_species_fixed=("n", "max"),
                                                  n_canisters=("POC", "nunique")).reset_index()
out = out.rename(columns={"Date Local": "date"}).sort_values(["date", "site"])
out.to_csv(os.path.join(HERE, "total_voc_fixed.csv"), index=False)
print(len(keep), "species kept; dropped", dropped)
print(out.groupby("site").agg(days=("date", "size"), multi_canister_days=("n_canisters", lambda x: int((x > 1).sum()))))
print(out.groupby(out.date.str[:4]).total_voc_fixed.mean().round(2))
