"""Exploratory sweep of every chemical against the paper's ORI, the 2026 refit (DEVIATIONS 40).

The code of corrected_and_species.py part 3, unchanged except that the score is f_refit (the refit on ERA5 inputs)
instead of the deployed model's corrected forecast: the same non-poster dates at the two community monitors, chemicals
detected on at least 10% of monitor-days with at least 100 values, joint date permutation (2,000), Benjamini-Hochberg
across chemicals. Seed 20261111. Output: species_sweep_refit.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = ["elementary", "johnson_riley"]
rng = np.random.default_rng(20261111)


def wrho(d, y, fc):
    return float(np.mean([spearmanr(g[fc], g[y])[0] for _, g in d.groupby("site") if g[y].nunique() > 1 and g[fc].nunique() > 1]))


def joint_perm(d, y, fc, B):
    d = d[d.site.isin(RES)].dropna(subset=[y, fc]); both = d.groupby("date").site.nunique(); d = d[d.date.isin(both[both == 2].index)]
    if d[y].nunique() < 3: return None
    piv = d.pivot(index="date", columns="site", values=fc); dates = piv.index.values; idx = {k: i for i, k in enumerate(dates)}
    obs = wrho(d, y, fc); c = 0; di = d.date.map(idx).values; sv = d.site.values; arr = {s: piv[s].values for s in RES}
    for _ in range(B):
        p = rng.permutation(len(dates)); c += abs(wrho(d.assign(**{fc: [arr[s][p[i]] for s, i in zip(sv, di)]}), y, fc)) >= abs(obs)
    return {"rho": round(obs, 3), "p": round((c + 1) / (B + 1), 4), "monitor_days": int(len(d)), "detected_share": round(float((d[y] > 0).mean()), 2)}


m = pd.read_csv("refit_calvert_monitor_days.csv")
new = m[~m.poster_window & m.site.isin(RES)]
a = pd.read_csv("calvert_aqs_extract.csv", dtype=str); a["val"] = pd.to_numeric(a["Arithmetic Mean"])
a["site"] = a["Site Num"].str.zfill(4).map({"0018": "elementary", "0020": "johnson_riley", "0021": "lwd"}); a["date"] = a["Date Local"]
a = a[a.site.isin(RES) & ~a["Parameter Name"].str.match(r"^(Total|Sum)")]
a["src"] = np.where(a.file == "HAPS", 0, 1)
a = a.sort_values("src").drop_duplicates(["date", "site", "Parameter Name"])
wide = a.pivot_table(index=["date", "site"], columns="Parameter Name", values="val").reset_index()
sp = new[["date", "site", "f_refit"]].merge(wide, on=["date", "site"])
res = []
for c in [c for c in wide.columns if c not in ("date", "site")]:
    if (sp[c] > 0).mean() < 0.10 or sp[c].notna().sum() < 100: continue
    r = joint_perm(sp, c, "f_refit", 2000)
    if r: res.append({"chemical": c, **r})
t = pd.DataFrame(res).sort_values("p").reset_index(drop=True); k = len(t)
t["q_bh"] = (t.p * k / (t.index + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1).round(4)
t.to_csv("species_sweep_refit.csv", index=False)
print(k, "chemicals;", int((t.q_bh < 0.05).sum()), "at q < 0.05;", int((t.rho > 0).sum()), "positive")
print(t.sort_values("rho", ascending=False).to_string(index=False))
