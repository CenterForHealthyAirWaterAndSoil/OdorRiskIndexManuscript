# How the HYSPLIT runs were set up

Every value below was read from the code and run files that produced the 189 scored dates. The code is in
`code/models/` (`04_code/models/` in the full package), abbreviated: `cfg` = `config.py`, `rb` = `run_backtest.py`,
`rbc` = `run_backtest_clipped.py`, `eng` = `calvert_plume_engine_one_day.py`. The values were checked against the
CONTROL, SETUP.cfg and MESSAGE files kept from the runs.

## What a run is

- One run is one calendar day in Calvert City, from local midnight, simulated for 23 hours with NOAA HYSPLIT.
- Sources: each of the 5 facilities releases its EPA TRI 2024 air emissions at a constant rate. Stack emissions are
  released at 15 m and fugitive emissions at 2 m. That gives 9 runs per day, because Arkema has a stack release only.
- Species: up to 7 per run (vinyl chloride, benzene, 1,2-dichloroethane (EDC), 1,3-butadiene, chloroform,
  tetrachloroethylene (PERC), trichloroethylene (TCE)).
- Output: for each monitor and species, the hourly air concentration from all 9 runs is summed and then
  averaged over the day.
- What is scored: the daily mean (ug/m3) at each monitor. Peak hour and "contact hours" (hours above zero)
  are saved but not scored.

## Settings

| Setting | Value | Note (source) |
|---|---|---|
| HYSPLIT | v5.4.2 (2025-05-14), `hycs_std` | `rb` 37-38; MESSAGE |
| Weather | NOAA HRRR 3 km, `prs` product, analysis hours only (fxx = 0), downloaded with Herbie | Real observed-quality weather, not a forecast (`rbc` 140-141) |
| Weather hours | 24 hourly analyses from local midnight | `rbc` 101-102 |
| Weather domain | Clipped to 35.0-39.0 N, 90.5-86.0 W (about 119 x 162 points), all 41 levels | Saves disk. Checked against full-domain runs (Spearman 0.9987) and a half-size box (within 1.8%). `rbc` 38-40, 98-115 |
| Day definition | Starts at local midnight, America/Chicago, so CDT in summer | `eng` 439, 574-589 |
| Run length | 23 h. Emissions also run 23 h, starting at local midnight | `cfg` 123; `rb` 152, 158 |
| Vertical motion, model top | Weather model's own vertical velocity (option 0); 10,000 m | `rb` 152 |
| Turbulence, stability | Kantha-Clayson; stability from wind and temperature profiles (KBLS = 2), because HRRR has no flux fields | HYSPLIT's own defaults; logged in MESSAGE |
| Particle mode | 3D particles (initd = 0); maximum particle age 24 h | `rb` 186-187 |
| Particles | numpar = -500, about 500 released per hour per species; maxpar = 150,000 | Large enough that no run reached it. `cfg` 127-128 |
| Release heights | Stack 15 m, fugitive 2 m, for every facility | TRI has no stack data, so these are generic (`eng` 186; `rb` 272) |
| Emission rate | g/h = (lb/yr) x 453.59237 / 8760, constant all day. Stack = TRI section 5.2; fugitive = TRI section 5.1 | The stack/fugitive split is TRI's own (`eng` 615-622) |
| Emission year | TRI 2024 for every date (primary) | `cfg` 141 |
| Concentration grid | 0.01 x 0.01 degrees (about 1.1 x 0.9 km), 1.0 x 1.0 degrees wide, centred at 37.0317, -88.3542 | `cfg` 120-122; `rb` 160-165 |
| Vertical layer | Air concentration averaged over 0-10 m above ground | `rb` 166; `cfg` 131 |
| Averaging | 1-hour averages, giving 23 values per day | `rb` 167 |
| Value at a monitor | `con2stn` with bilinear interpolation (`-xi`) to the exact monitor coordinate, 0-10 m level (`-z2`) | `rb` 224-225 |
| Daily outputs | Mean of the 23 hours (scored), peak hour, contact hours | `rb` 293-301 |
| Units | HYSPLIT g/m3 x 1e6 = ug/m3 | `cfg` 112 |
| Monitor units | Monitor data are ppbC. For amounts: ppb = ppbC / carbons, ug/m3 = ppb x MW / 24.45. Rankings need no conversion | `cfg` 111 |
| Quality checks | 1 h time limit per run (3600 s). A run counts only if HYSPLIT exits cleanly (HYSPLIT_OK file). A date gets its DONE file only when all 9 runs pass; a failure fails the whole date, which gets one retry | `cfg` 130; `rb` 192-211, 279-282; `rbc` 209-215 |
| Completeness | All 189 dates: 9 of 9 HYSPLIT_OK, and no particle-cap warnings | Run log (full package: `05_results/hysplit/RUNLOG.md`) |

Facilities and TRI 2024 rates (g/h, stack + fugitive) (CONTROL files; locations within 85-620 m of EPA's facility
registry; full package: `05_results/audits/PLUME_AUDIT_2026-10-24.md`)

| Facility | Lat, lon | What it releases in the model |
|---|---|---|
| Westlake PVC Plant | 37.0450, -88.3495 | Vinyl chloride 2,031 + 201: 87% of modeled vinyl chloride |
| Westlake Vinyls | 37.0491, -88.3322 | All 7 species. Vinyl chloride 77 + 263, EDC 295 + 751, benzene 340 + 454, butadiene 197 + 38; chloroform, PERC and TCE small |
| ISP Chemicals | 37.0474, -88.3596 | Benzene 21 + 782 |
| Lubrizol Advanced Materials | 37.0486, -88.3318 | Benzene 44 + 406 |
| Arkema | 37.0563, -88.3665 | EDC 0.08 (stack only; negligible) |

Species (`cfg` 69-105). Every species is a gas.

| Species | MW | Half-life (days) | Dry deposition (m/s) |
|---|---|---|---|
| Vinyl chloride | 62.5 | 1.5 | 0.003 |
| Benzene | 78.1 | 9.7 | 0.003 |
| EDC | 99.0 | 83 | 0.003 |
| 1,3-Butadiene | 54.1 | 1.4 | 0.003 |
| Chloroform | 119.4 | 150 | 0.003 |
| PERC | 165.8 | 117 | 0.003 |
| TCE | 131.4 | 7.0 | 0.004 |

## What we know is imperfect

- No wet removal: the HRRR analysis files carry no precipitation, so rain-out never ran. Dry deposition at 0.003 m/s
  is active.
- The grid is coarse for Johnson-Riley. The monitor is about 470 m from the PVC plant, inside a roughly 1 km grid
  cell next to the plant. Its predicted amounts are a smeared near-source average.
- Emissions are simplified. Each facility's annual total is spread evenly over every hour, with generic release
  heights. Real releases vary by hour and by process.
- A one-hour daylight-time offset. HYSPLIT days follow clock time and EPA samples follow standard time, so they
  share 23 of 24 hours in winter and 22 of 24 in summer.
- Emission year matters for amounts, not rankings. Checked on the 156 dates outside the poster window:

  | Emissions | Vinyl chloride rho | Benzene rho |
  |---|---|---|
  | TRI 2024 (primary) | 0.595 | 0.257 |
  | Year-matched | 0.569 | 0.267 |
  | Prior-year, which is what a forecaster would know | 0.560 | 0.256 |

  The rankings hold with every emission year. The alternatives rescale the saved output, which is valid because
  HYSPLIT is linear in the emission rate.
- Amounts are off; rankings hold. Mean predicted over mean measured, by monitor, all 189 dates:

  | Species | Elementary | Johnson-Riley |
  |---|---|---|
  | Vinyl chloride | 6.1x | 1.4x |
  | Benzene | 1.9x | 5.1x |
  | EDC | 0.7x | 0.8x |

  Non-detects count as 0. Use HYSPLIT to rank days, not to estimate ug/m3.

## Re-running one date (on the original computer)

```
cd ~/plume-validation
cp validation/results/predicted_clipped/2025-03-08.csv /tmp/2025-03-08_scored.csv   # keep the scored version
rm -f validation/runs_clipped/2025-03-08/*/HYSPLIT_OK    # otherwise finished runs are reused, not re-run
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -u -m validation.run_backtest_clipped \
    --dates 2025-03-08 --force --keep-met
```

The command writes:
- `validation/met_clipped/MET_20250308.ARL`, the clipped weather (about 130 MB). It is reused if it is already there;
  92 dates still have theirs. Otherwise it is downloaded, which takes about 80 minutes.
- `validation/runs_clipped/2025-03-08/<facility>_<stack|fugitive>/`, holding CONTROL, SETUP.cfg, cdump, con2stn.txt,
  MESSAGE and HYSPLIT_OK.
- `validation/results/predicted_clipped/2025-03-08.csv` and its `.DONE` file.
