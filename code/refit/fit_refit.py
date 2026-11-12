"""Odor Risk Index refit on training panel v2, exactly as refit/REFIT_PLAN.md (frozen 2026-10-28) specifies.

Reads training_panel_v2/zip_day_panel.parquet and writes training_panel_v2/fit/. It scores nothing at Calvert and
changes no deployed coefficient. Any missing input stops the run; nothing is filled in. The run refuses to start if
REFIT_PLAN.md no longer matches its frozen hash.

Run from the repo root:  .venv/bin/python -m training.fit_refit
Plan section numbers (s6.2 and so on) and decision numbers (M1-M20, DEVIATIONS item 23) are cited in comments.
"""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score

import odor_forecast_core as core
from training import config
from training import reports as R
from training import source_descriptions as SD

PANEL = os.path.join(config.OUT_DIR, "zip_day_panel.parquet")
OUT = os.path.join(config.OUT_DIR, "fit")
PLAN = os.path.expanduser("~/school/R!Louisville/Re-analysis/refit/REFIT_PLAN.md")
PLAN_HASH = os.path.expanduser("~/school/R!Louisville/Re-analysis/refit/REFIT_PLAN_HASH.txt")

SEED = 20261027
N_FOLDS = 10
N_BOOT = 1000
BLH_FITTED_SCALE = 3.28084   # the deployed BLH coefficient was fitted on BLH this many times too large (M10b)

WEATHER = ("temp_c", "temp_c2", "diurnal_temperature_range", "relative_humidity", "wind_speed",
           "boundary_layer_height", "solar_radiation", "precipitation", "pressure_anomaly")
WEATHER_RAW = ("temperature", "temperature_squared", "diurnal_temperature_range", "relative_humidity", "wind_speed",
               "boundary_layer_height", "solar_radiation", "precipitation", "atmospheric_pressure")
SOURCE = ("multi_source_exposure", "wind_align_weighted")
DOW = ("dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "holiday")
TOP_ZIPS = {"pgh": ("15212", "15217", "15218", "15221", "15025"),
            "lou": ("40206", "40211", "40203", "40204", "40212")}
FORWARD_YEARS = (2022, 2023, 2024, 2025)
CALVERT_YEAR = 2025
# s5 expected signs (M18): +1 positive, -1 negative, 0 no single expectation.
EXPECTED_SIGN = {"boundary_layer_height": -1, "wind_speed": -1, "wind_align_weighted": 1, "multi_source_exposure": 1,
                 "diurnal_temperature_range": 1, "pressure_anomaly": 1, "temp_c": 0, "temp_c2": 0,
                 "relative_humidity": 0, "solar_radiation": 0, "precipitation": 0}
SEWER_RULES = (SD.ODOR_TYPES["sewage / sewer"], SD.NAMED["lou"]["MSD / Morris Forman (sewer, wastewater)"])


class FitError(RuntimeError):
    """The fit cannot proceed without guessing (missing input, empty column, non-convergence, changed plan)."""


# ------------------------------------------------------------------------------------------------ specification

@dataclass(frozen=True)
class Spec:
    """One fit. Defaults are the main fit (s2-s5)."""
    name: str
    population: str = "main"        # main | report_days | sparse_included | orig61 | bridging
    event: str = "city_mean"        # city_mean | zip_mean | two_reports | negative_control
    burden: str = "burden_hourly_mean_excl1"
    weather: tuple = WEATHER
    source: tuple = SOURCE
    intercepts: str = "city_year"   # city_year | city_year_month | city_const | zip_fe
    dow: bool = True
    cities: tuple = ("pgh", "lou")
    drop_zips: tuple = ()
    interact: tuple = ()            # terms multiplied by the Louisville indicator
    cv: bool = True                 # month-block CV metrics (False: coefficients only)


MAIN = Spec("main")


def sensitivity_specs():
    """s6.7, in the plan's table order. Every other setting is the main fit's."""
    specs = [
        Spec("report_days", population="report_days"),
        Spec("zip_mean_event", event="zip_mean"),
        Spec("city_year_month", intercepts="city_year_month", cv=False),
        Spec("sparse_included", population="sparse_included"),
        Spec("rain_left_out", weather=tuple(t for t in WEATHER if t != "precipitation")),
        Spec("original_61_zips", population="orig61"),
        Spec("two_reports", event="two_reports"),
        Spec("zip_fixed_effects", intercepts="zip_fe", source=("wind_align_weighted",), cv=False),
    ]
    for city in ("pgh", "lou"):
        specs += [Spec(f"leave_out_{z}", drop_zips=(z,), cv=False) for z in TOP_ZIPS[city]]
    specs += [Spec("leave_out_15212_40206", drop_zips=("15212", "40206"), cv=False),
              Spec("negative_control", event="negative_control"),
              Spec("sewer_removed", burden="burden_sewer_removed"),
              Spec("bridging", population="bridging", event="city_mean", burden="burden_hourly_mean",
                   weather=WEATHER_RAW, source=("multi_source_exposure", "wind_align_weighted_circmean"),
                   intercepts="city_const", dow=False)]
    return specs


# ------------------------------------------------------------------------------------------------ data

def load_panel(path=PANEL):
    d = pd.read_parquet(path)
    d["zipcode"] = d.zipcode.astype(str)
    d["date"] = d.date.astype(str)
    dates = pd.to_datetime(d.date)
    d["week"] = (dates - pd.to_timedelta(d.dow, unit="D")).dt.strftime("%Y-%m-%d")
    d["quarter"] = d.year.astype(str) + "Q" + ((d.month - 1) // 3 + 1).astype(str)
    d["doy"] = dates.dt.dayofyear
    return d


def require_complete(frame, columns, what):
    """Stop if any input is missing. The fit never fills a value in."""
    bad = frame[list(columns)].isna().sum()
    bad = bad[bad > 0]
    if len(bad):
        raise FitError(f"{what}: missing values in {bad.to_dict()}; nothing is filled in")


def population(d, spec):
    keep = d.city.isin(spec.cities)
    if spec.population in ("main", "report_days", "orig61"):
        keep &= ~d.sparse_reporting_period
    if spec.population == "report_days":
        keep &= d.has_report
    elif spec.population == "orig61":
        keep &= d.zip_in_original_set
    elif spec.population == "bridging":
        keep &= d.has_report & d.zip_in_original_set
    elif spec.population not in ("main", "sparse_included"):
        raise FitError(f"unknown population {spec.population}")
    keep &= ~d.zipcode.isin(spec.drop_zips)
    return d[keep].reset_index(drop=True)


def events(p, spec):
    """Event column and the thresholds used (s3; M9, M16). Thresholds are recomputed on the fit population."""
    if spec.event == "city_mean":
        thr = p.groupby("city")[spec.burden].mean()
        return (p[spec.burden] > p.city.map(thr)).astype(int).to_numpy(), {k: float(v) for k, v in thr.items()}
    if spec.event == "zip_mean":
        thr = p.groupby("zipcode")[spec.burden].transform("mean")
        return (p[spec.burden] > thr).astype(int).to_numpy(), {"rule": "burden above the zip's own mean"}
    if spec.event == "two_reports":
        return (p.n_reports_excl1 >= 2).astype(int).to_numpy(), {"rule": "n_reports_excl1 >= 2"}
    if spec.event == "negative_control":
        return (p.n_reports_value1 >= 1).astype(int).to_numpy(), {"rule": "n_reports_value1 >= 1"}
    raise FitError(f"unknown event rule {spec.event}")


def equal_city_weights(city):
    """w = N / (n_cities x N_city): each city carries equal total weight and the weights sum to N (s4)."""
    city = pd.Series(city)
    n_city = city.map(city.value_counts())
    return (len(city) / (city.nunique() * n_city)).to_numpy(float)


def temperature_centre(d):
    """Mean daily temperature of the main fit population (s5, M18); used to centre temperature in every fit."""
    return float(population(d, MAIN).temperature.mean())


def add_centred_temperature(p, t0):
    p = p.copy()
    p["temp_c"] = p.temperature - t0
    p["temp_c2"] = p.temp_c ** 2
    return p


# ------------------------------------------------------------------------------------------------ design

@dataclass
class Design:
    X: sp.csr_matrix          # columns scaled by `scale` (continuous terms only)
    names: list
    scale: np.ndarray         # the column divisors; coefficients on the original scale = beta / scale
    continuous: list          # indices of continuous columns
    dow_idx: list             # indices of the day-of-week and holiday columns
    intercept_idx: list       # indices of intercept-type columns (city-year, const, zip)
    within_idx: list          # columns that vary within a zip and year: weather, alignment, harmonics, interactions


def _dummies(labels, prefix, drop=()):
    codes, uniq = pd.factorize(pd.Series(labels), sort=True)
    keep = [i for i, u in enumerate(uniq) if u not in drop]
    remap = -np.ones(len(uniq), int)
    remap[keep] = np.arange(len(keep))
    rows = np.flatnonzero(remap[codes] >= 0)
    m = sp.csr_matrix((np.ones(len(rows)), (rows, remap[codes][rows])), shape=(len(labels), len(keep)))
    return m, [f"{prefix}:{uniq[i]}" for i in keep]


def design(p, spec):
    cont_names = list(spec.weather) + list(spec.source)
    cont = p[cont_names].to_numpy(float) if cont_names else np.zeros((len(p), 0))
    lou = (p.city == "lou").to_numpy(float)
    for t in spec.interact:
        cont = np.column_stack([cont, lou * p[t].to_numpy(float)])
        cont_names.append(f"lou_x_{t}")
    scale = cont.std(axis=0) if cont.shape[1] else np.zeros(0)
    if (scale == 0).any():
        raise FitError(f"{spec.name}: a continuous term is constant: {np.array(cont_names)[scale == 0].tolist()}")
    blocks = [sp.csr_matrix(cont / scale)] if cont.shape[1] else []
    names = list(cont_names)

    cy = p.city + ":" + p.year.astype(str)
    if spec.intercepts == "city_year":
        m, n = _dummies(cy, "cy")
    elif spec.intercepts == "city_year_month":
        m, n = _dummies(p.city + ":" + p.year_month, "cym")
    elif spec.intercepts == "city_const":
        m = sp.csr_matrix(np.column_stack([np.ones(len(p)), lou]))
        n = ["const", "lou"]
    elif spec.intercepts == "zip_fe":
        mz, nz = _dummies(p.zipcode, "zip")
        first = {f"{c}:{p.loc[p.city == c, 'year'].min()}" for c in p.city.unique()}
        my, ny = _dummies(cy, "cy", drop=first)   # one reference year per city: zip indicators span the city
        m, n = sp.hstack([mz, my]).tocsr(), nz + ny
    else:
        raise FitError(f"unknown intercepts {spec.intercepts}")
    start = len(names)
    blocks.append(m)
    names += n
    intercept_idx = list(range(start, len(names)))

    dow_idx = []
    if spec.dow:
        dm = np.column_stack([(p.dow == k).to_numpy(float) for k in range(1, 7)] + [p.is_holiday.to_numpy(float)])
        dow_idx = list(range(len(names), len(names) + 7))
        blocks.append(sp.csr_matrix(dm))
        names += list(DOW)
    X = sp.hstack(blocks).tocsr()
    full_scale = np.ones(len(names))
    full_scale[:len(scale)] = scale
    within = [i for i, nm in enumerate(cont_names) if nm != "multi_source_exposure"]
    return Design(X, names, full_scale, list(range(len(scale))), dow_idx, intercept_idx, within)


# ------------------------------------------------------------------------------------------------ estimation

def loglik(y, w, eta):
    return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))


def fit_logit(X, y, w, offset=None, max_iter=100, tol=1e-10):
    """Weighted logistic regression by Newton-Raphson with step halving. Unpenalized."""
    X = sp.csr_matrix(X)
    n, k = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    # Standard GLM start (as in R's glm): one weighted least-squares step from mu0 = (y + 0.5) / 2. Starting at
    # beta = 0 fails when a large offset puts every starting probability near 0 or 1.
    mu0 = (y + 0.5) / 2
    z = np.log(mu0 / (1 - mu0)) - off
    W0 = w * mu0 * (1 - mu0)
    try:
        beta = np.linalg.solve((X.T @ sp.diags(W0) @ X).toarray(), X.T @ (W0 * z))
    except np.linalg.LinAlgError as e:
        raise FitError("singular design") from e
    eta = X @ beta + off
    ll = loglik(y, w, eta)
    for it in range(1, max_iter + 1):
        mu = expit(eta)
        g = X.T @ (w * (y - mu))
        H = (X.T @ sp.diags(w * mu * (1 - mu)) @ X).toarray()
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError as e:
            raise FitError(f"singular Hessian at iteration {it}") from e
        t = 1.0
        while True:
            nb = beta + t * step
            neta = X @ nb + off
            nll = loglik(y, w, neta)
            if nll >= ll - 1e-9 * abs(ll) or t < 1e-6:
                break
            t /= 2
        beta, eta, gain = nb, neta, nll - ll
        ll = nll
        if np.max(np.abs(t * step)) < tol or (abs(gain) < 1e-12 * abs(ll) and it > 2):
            mu = expit(eta)
            H = (X.T @ sp.diags(w * mu * (1 - mu)) @ X).toarray()
            return {"beta": beta, "mu": mu, "H": H, "loglik": ll, "iterations": it, "converged": True}
    raise FitError(f"no convergence after {max_iter} iterations")


def _cluster_meat(X, score_resid, groups):
    codes, uniq = pd.factorize(pd.Series(groups))
    G = len(uniq)
    S = (sp.csr_matrix((score_resid, (codes, np.arange(len(codes)))), shape=(G, len(codes))) @ X)
    S = S.toarray() if sp.issparse(S) else np.asarray(S)
    return S.T @ S, G


def vcov_cluster(fit, X, y, w, groups):
    """CR1 cluster-robust covariance (one-way)."""
    n, k = X.shape
    Hinv = np.linalg.inv(fit["H"])
    meat, G = _cluster_meat(X, w * (y - fit["mu"]), groups)
    return G / (G - 1) * (n - 1) / (n - k) * Hinv @ meat @ Hinv


def vcov_twoway(fit, X, y, w, g1, g2):
    """Two-way cluster-robust covariance (Cameron, Gelbach and Miller): V1 + V2 - V12, each with its CR1 factor.
    Returns (V, n_negative_eigenvalues_clipped)."""
    g12 = (pd.Series(g1).astype(str) + "|" + pd.Series(g2).astype(str)).to_numpy()
    V = (vcov_cluster(fit, X, y, w, g1) + vcov_cluster(fit, X, y, w, g2) - vcov_cluster(fit, X, y, w, g12))
    V = (V + V.T) / 2
    ev, vec = np.linalg.eigh(V)
    neg = int((ev < 0).sum())
    if neg:
        V = vec @ np.diag(np.clip(ev, 0, None)) @ vec.T
    return V, neg


def to_original_scale(beta, V, scale):
    b = beta / scale
    return b, V / np.outer(scale, scale)


# ------------------------------------------------------------------------------------------------ metrics

class StratifiedC:
    """Concordance within strata: every event paired with every non-event in the same stratum; ties count 1/2.
    Precomputes the sort once so bootstrap row weights can be applied cheaply (`value(weights)`)."""

    def __init__(self, y, score, strata):
        y = np.asarray(y).astype(bool)
        codes = pd.factorize(pd.Series(strata))[0]
        order = np.lexsort((np.asarray(score, float), codes))
        self.order, self.ev = order, y[order]
        st, sc = codes[order], np.asarray(score, float)[order]
        n = len(order)
        new_st = np.r_[True, st[1:] != st[:-1]]
        new_tie = new_st | np.r_[True, sc[1:] != sc[:-1]]
        self.stratum_id = np.cumsum(new_st) - 1
        tie_id = np.cumsum(new_tie) - 1
        tie_start = np.flatnonzero(new_tie)
        self.tie_lo = tie_start[tie_id]
        self.tie_hi = np.r_[tie_start[1:], n][tie_id]
        self.st_lo = np.flatnonzero(new_st)[self.stratum_id]

    def value(self, weights=None):
        w = np.ones(len(self.order)) if weights is None else np.asarray(weights, float)[self.order]
        we, wn = w * self.ev, w * ~self.ev
        cum = np.r_[0.0, np.cumsum(wn)]
        below = cum[self.tie_lo] - cum[self.st_lo]
        tied = cum[self.tie_hi] - cum[self.tie_lo]
        num = np.sum(we * (below + 0.5 * tied))
        den = np.sum(np.bincount(self.stratum_id, we) * np.bincount(self.stratum_id, wn))
        return float(num / den) if den > 0 else float("nan")


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def calibration(y, p):
    """Calibration-in-the-large (intercept with slope fixed at 1) and calibration slope (logistic recalibration)."""
    lp = _logit(p)
    ones = np.ones(len(y))
    itl = fit_logit(sp.csr_matrix(ones[:, None]), y, ones, offset=lp)["beta"][0]
    slope = fit_logit(sp.csr_matrix(np.column_stack([ones, lp])), y, ones)["beta"][1]
    return float(itl), float(slope)


def city_metrics(y, p, strata, ranking_only=False, rank=None):
    """rank: the within-zip-and-year part of the out-of-fold linear predictor (DEVIATIONS item 24). Without it,
    zip-stratified concordance ranks by p, which is right only when one model scored every row of a stratum."""
    out = {"n": int(len(y)), "events": int(y.sum()), "event_rate": float(y.mean()),
           "zip_strat_c": StratifiedC(y, p if rank is None else rank, strata).value(),
           "auc": float(roc_auc_score(y, p))}
    if rank is not None:
        out["zip_strat_c_literal"] = StratifiedC(y, p, strata).value()
    if not ranking_only:
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        out["logloss"] = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
        out["brier"] = float(np.mean((p - y) ** 2))
        out["cal_in_the_large"], out["cal_slope"] = calibration(y, p)
    return out


def f1_threshold(y, p, w):
    """Threshold maximising weighted F1 (predict an event when p >= threshold). Returns (threshold, F1)."""
    o = np.argsort(-p, kind="stable")
    ys, ps, ws = y[o], p[o], w[o]
    tp, fp = np.cumsum(ws * ys), np.cumsum(ws * (1 - ys))
    pos = tp[-1]
    last = np.r_[ps[1:] != ps[:-1], True]
    f1 = 2 * tp[last] / (tp[last] + fp[last] + pos)
    i = int(np.argmax(f1))
    return float(ps[last][i]), float(f1[i])


def operating(y, p, thr):
    pred = p >= thr
    tp, fp, fn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": float(prec), "recall": float(rec),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan"),
            "share_flagged": float(pred.mean())}


# ------------------------------------------------------------------------------------------------ folds

def month_folds(year_month, seed=SEED, k=N_FOLDS):
    """s6.1: months shuffled within each year and dealt round-robin into k folds, the deal continuing across years."""
    rng = np.random.default_rng(seed)
    months = sorted(set(year_month))
    fold, i = {}, 0
    for year in sorted({m[:4] for m in months}):
        for m in rng.permutation([m for m in months if m[:4] == year]):
            fold[m] = i % k
            i += 1
    return np.array([fold[m] for m in year_month])


def shuffled_folds(labels, seed=SEED, k=N_FOLDS):
    """Leakage-diagnostic folds (weeks or single dates): units shuffled and dealt into k folds."""
    units = sorted(set(labels))
    perm = np.random.default_rng(seed).permutation(len(units))
    fold = {units[j]: i % k for i, j in enumerate(perm)}
    return np.array([fold[u] for u in labels])


# ------------------------------------------------------------------------------------------------ fitting helpers

@dataclass
class Prepared:
    spec: Spec
    p: pd.DataFrame
    y: np.ndarray
    w: np.ndarray
    D: Design
    thresholds: dict
    strata: np.ndarray


def prepare(d, spec, t0):
    p = add_centred_temperature(population(d, spec), t0)
    used = [c for c in spec.weather + spec.source + (spec.burden,) if c in p.columns]
    require_complete(p, used + ["city", "zipcode", "date", "year", "dow", "is_holiday"], spec.name)
    y, thr = events(p, spec)
    if spec.intercepts == "zip_fe":   # a zip with no event carries no information in a fixed-effects logit
        has = pd.Series(y).groupby(p.zipcode.to_numpy()).transform("max").to_numpy() > 0
        p, y = p[has].reset_index(drop=True), y[has]
    w = equal_city_weights(p.city)
    strata = (p.zipcode + "|" + p.year.astype(str)).to_numpy()
    return Prepared(spec, p, y, w, design(p, spec), thr, strata)


def fit_rows(P, rows, offset=None):
    """Fit on a row subset, dropping columns that are all zero there. Returns (fit, kept column indices)."""
    X = P.D.X[rows]
    keep = np.flatnonzero(np.asarray((X != 0).sum(axis=0)).ravel() > 0)
    off = None if offset is None else offset[rows]
    f = fit_logit(X[:, keep], P.y[rows], P.w[rows], off)
    return f, keep


def predict_rows(P, fit, keep, train, test, offset=None, intercept=None):
    """Linear predictors for test rows: (actual day-of-week effects, average effect, within-zip-and-year part).

    The within part (weather, alignment, harmonics, interactions, plus any fixed offset) is what can rank days inside
    one zip and year; it leaves out each fold's re-estimated intercepts and proximity term (DEVIATIONS item 24).

    intercept: a constant replacing the test rows' own intercept columns (forward-by-year and leave-one-city-out,
    where the test rows' intercepts were not estimated)."""
    X = P.D.X
    beta = np.zeros(X.shape[1])
    beta[keep] = fit["beta"]
    Xt = X[test]
    est = np.zeros(X.shape[1], bool)
    est[keep] = True
    if intercept is None:
        missing = np.asarray((Xt[:, np.flatnonzero(~est)] != 0).sum(axis=0)).ravel()
        if missing.any():
            raise FitError(f"{P.spec.name}: test rows need a column the training rows never had")
    lp = Xt @ beta + (0 if offset is None else offset[test])
    if intercept is not None:
        lp = lp - Xt[:, P.D.intercept_idx] @ beta[P.D.intercept_idx] + intercept
    lp_avg = lp
    if P.D.dow_idx:
        shares = np.asarray(X[train][:, P.D.dow_idx].mean(axis=0)).ravel()
        b = beta[P.D.dow_idx]
        lp_avg = lp - Xt[:, P.D.dow_idx] @ b + shares @ b
    w = P.D.within_idx
    lp_within = (Xt[:, w] @ beta[w] if w else np.zeros(Xt.shape[0])) + (0 if offset is None else offset[test])
    return lp, lp_avg, lp_within


def crossval(P, folds, offset=None):
    """Out-of-fold linear predictors: actual and average day-of-week effects, and the within-zip-and-year part."""
    n = len(P.y)
    lp_act, lp_avg, lp_w = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        f, keep = fit_rows(P, tr, offset)
        lp_act[te], lp_avg[te], lp_w[te] = predict_rows(P, f, keep, tr, te, offset)
    if np.isnan(lp_avg).any():
        raise FitError(f"{P.spec.name}: rows without an out-of-fold prediction")
    return lp_act, lp_avg, lp_w


def metrics_by_city(P, prob, ranking_only=False, rows=None, rank=None):
    rows = np.ones(len(P.y), bool) if rows is None else rows
    out = {}
    for c in ("pgh", "lou"):
        m = rows & (P.p.city == c).to_numpy()
        if m.any():
            out[c] = city_metrics(P.y[m], prob[m], P.strata[m], ranking_only, None if rank is None else rank[m])
    return out


def coefficient_table(P, fit, keep, Vs):
    """Coefficients on the original scale with each SE in Vs ({label: covariance on the fitted scale})."""
    names = [P.D.names[i] for i in keep]
    scale = P.D.scale[keep]
    b = fit["beta"] / scale
    out = pd.DataFrame({"term": names, "coef": b, "sd": np.where(scale == 1, np.nan, scale),
                        "per_sd": np.where(scale == 1, np.nan, fit["beta"])})
    for label, V in Vs.items():
        out[f"se_{label}"] = np.sqrt(np.diag(V)) / scale
    if "se_twoway" in out:
        out["z"] = out.coef / out.se_twoway
        out["p_twoway"] = 2 * stats.norm.sf(np.abs(out.z))
        out["ci_low"], out["ci_high"] = out.coef - 1.959964 * out.se_twoway, out.coef + 1.959964 * out.se_twoway
    return out


def full_fit(P, clusterings=("twoway", "date")):
    rows = np.ones(len(P.y), bool)
    f, keep = fit_rows(P, rows)
    X = P.D.X[:, keep]
    V, info = {}, {}
    for c in clusterings:
        if c == "twoway":
            V[c], info["twoway_negative_eigenvalues"] = vcov_twoway(f, X, P.y, P.w, P.p.zipcode.to_numpy(),
                                                                    P.p.year_month.to_numpy())
        else:
            col = {"date": "date", "week": "week", "month": "year_month", "quarter": "quarter", "zip": "zipcode"}[c]
            V[c] = vcov_cluster(f, X, P.y, P.w, P.p[col].to_numpy())
    return f, keep, V, info


def wald(coefs, V_orig, terms):
    idx = [list(coefs.term).index(t) for t in terms]
    b = coefs.coef.to_numpy()[idx]
    stat = float(b @ np.linalg.solve(V_orig[np.ix_(idx, idx)], b))
    return {"terms": list(terms), "chi2": stat, "df": len(idx), "p": float(stats.chi2.sf(stat, len(idx)))}


# ------------------------------------------------------------------------------------------------ bootstrap

def month_draws(months, n_boot=N_BOOT, seed=SEED):
    """1,000 resamples of calendar months (with replacement); returns month labels and a (n_boot, M) count matrix."""
    labels = np.array(sorted(set(months)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    counts = np.stack([np.bincount(r, minlength=len(labels)) for r in idx])
    return labels, counts


def paired_bootstrap(P, prob_a, prob_b, draws, rows=None, rank_a=None, rank_b=None):
    """Per city: A minus B in zip-stratified concordance (ranked by rank_a / rank_b when given) and log-loss,
    with month-bootstrap 95% CIs."""
    rank_a = prob_a if rank_a is None else rank_a
    rank_b = prob_b if rank_b is None else rank_b
    labels, counts = draws
    lookup = {m: i for i, m in enumerate(labels)}
    rows = np.ones(len(P.y), bool) if rows is None else rows
    out = {}
    for c in ("pgh", "lou"):
        m = rows & (P.p.city == c).to_numpy()
        if not m.any():
            continue
        y, s, mi = P.y[m], P.strata[m], np.array([lookup[x] for x in P.p.year_month.to_numpy()[m]])
        ca, cb = StratifiedC(y, rank_a[m], s), StratifiedC(y, rank_b[m], s)
        lla = -(y * np.log(np.clip(prob_a[m], 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - prob_a[m], 1e-15, 1)))
        llb = -(y * np.log(np.clip(prob_b[m], 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - prob_b[m], 1e-15, 1)))
        dsum = np.bincount(mi, lla - llb, minlength=len(labels))
        nsum = np.bincount(mi, minlength=len(labels))
        dc, dll = [], []
        for cnt in counts:
            wt = cnt[mi].astype(float)
            dc.append(ca.value(wt) - cb.value(wt))
            dll.append((cnt @ dsum) / (cnt @ nsum))
        out[c] = {"diff_zip_strat_c": ca.value() - cb.value(),
                  "ci_zip_strat_c": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
                  "diff_logloss": float(np.mean(lla - llb)),
                  "ci_logloss": [float(np.percentile(dll, 2.5)), float(np.percentile(dll, 97.5))]}
    return out


# ------------------------------------------------------------------------------------------------ sewer rebuild (M17)

def sewer_flags(text, smell_value):
    """Value 2-5 reports whose text matches the sewer odor type or the MSD named source (source_descriptions.py)."""
    t = text.str.lower()
    hit = t.str.contains(SEWER_RULES[0], regex=True) | t.str.contains(SEWER_RULES[1], regex=True)
    return (hit & (smell_value >= 2)).to_numpy()


def burden_from_reports(kept):
    """zip x date burden_hourly_mean_excl1 from a set of reports, with the panel's own target code."""
    t = R.zip_day_targets(R.zip_hour_table(kept))
    return t[["zipcode", "date", "burden_hourly_mean_excl1"]]


def louisville_burden(lou_rows, remove_sewer=True):
    """Louisville burden rebuilt from the raw reports, optionally without sewer reports. Returns (burden, n_removed)."""
    cfg = config.CITIES["lou"]
    rep = R.load_reports(cfg["reports"], cfg["tz"])
    raw = pd.read_csv(cfg["reports"], usecols=["smell description", "additional comments"], dtype=str)
    if len(raw) != len(rep):
        raise FitError("report text and report rows do not line up")
    # Blank text is matched as an empty string (it never matches a sewer rule), as in source_descriptions.py.
    text = raw["smell description"].fillna("") + " | " + raw["additional comments"].fillna("")
    flag = pd.Series(sewer_flags(text, rep.smell_value), index=rep.index)
    kept, _ = R.select_reports(rep, set(lou_rows.zipcode), config.START_DATE, config.END_DATE)
    n_removed = 0
    if remove_sewer:
        drop = flag.loc[kept.index].to_numpy()
        n_removed = int(drop.sum())
        kept = kept[~drop]
    b = lou_rows[["zipcode", "date"]].merge(burden_from_reports(kept), on=["zipcode", "date"], how="left",
                                            validate="one_to_one")
    # A zip-day with no remaining report has burden 0 by definition (the panel's own rule), not a filled-in value.
    return b.burden_hourly_mean_excl1.fillna(0.0).to_numpy(), n_removed


def add_sewer_burden(d):
    d = d.copy()
    lou = d.city == "lou"
    check, _ = louisville_burden(d[lou], remove_sewer=False)
    if np.max(np.abs(check - d.loc[lou, "burden_hourly_mean_excl1"].to_numpy())) != 0:
        raise FitError("rebuilding Louisville's burden from the raw reports does not reproduce the panel")
    rebuilt, n_removed = louisville_burden(d[lou], remove_sewer=True)
    d["burden_sewer_removed"] = d.burden_hourly_mean_excl1
    d.loc[lou, "burden_sewer_removed"] = rebuilt
    return d, n_removed


# ------------------------------------------------------------------------------------------------ deployed model (M10)

def deployed_lp(p, blh_scale=1.0, with_const=True):
    """Linear predictor of COEFFS_POOLED_PROXIMITY as deployed (raw temperature and pressure, BLH in true feet)."""
    c = core.COEFFS_POOLED_PROXIMITY
    cols = {"temperature": "temperature", "temperature_squared": "temperature_squared",
            "solar_radiation": "solar_radiation", "relative_humidity": "relative_humidity", "wind_speed": "wind_speed",
            "precipitation": "precipitation", "diurnal_temperature_range": "diurnal_temperature_range",
            "boundary_layer_height": "boundary_layer_height", "atmospheric_pressure": "atmospheric_pressure",
            "multi_source_exposure": "multi_source_exposure", "wind_align_weighted": "wind_align_weighted"}
    if set(c) - {"const"} != set(cols):
        raise FitError(f"deployed coefficients have unexpected terms: {sorted(c)}")
    require_complete(p, cols.values(), "deployed model inputs")
    lp = sum(c[k] * p[v].to_numpy(float) * (blh_scale if k == "boundary_layer_height" else 1.0) for k, v in cols.items())
    return lp + (c["const"] if with_const else 0.0)


# ------------------------------------------------------------------------------------------------ main run

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_plan_frozen():
    want = open(PLAN_HASH).read().split()[0]
    got = _sha256(PLAN)
    if want != got:
        raise FitError(f"REFIT_PLAN.md changed since it was frozen ({got[:12]} != {want[:12]}); log a deviation first")
    return want


def _prob(lp):
    return expit(lp)


def run(out=OUT, log=print):
    t_start = time.time()
    plan_sha = check_plan_frozen()
    os.makedirs(out, exist_ok=True)
    d = load_panel()
    t0 = temperature_centre(d)
    d, n_sewer = add_sewer_burden(d)
    draws = month_draws(d.year_month[~d.sparse_reporting_period])
    coef_rows, metric_rows, results = [], [], {}

    def record_metrics(name, scheme, mets, dow_mode="average"):
        for c, m in mets.items():
            metric_rows.append({"fit": name, "folds": scheme, "dow": dow_mode, "city": c, **m})

    # ---- main fit (s5) with every SE (s6.3)
    log("main fit")
    P = prepare(d, MAIN, t0)
    f, keep, V, info = full_fit(P, ("twoway", "date", "week", "month", "quarter", "zip"))
    main_coef = coefficient_table(P, f, keep, V)
    coef_rows.append(main_coef.assign(fit="main"))
    V2 = to_original_scale(f["beta"], V["twoway"], P.D.scale[keep])[1]
    results["main"] = {"n": len(P.y), "events": int(P.y.sum()), "thresholds": P.thresholds,
                       "iterations": f["iterations"], **info,
                       "weather_block_wald": wald(main_coef, V2, WEATHER),
                       "vif": vif(P.p, WEATHER)}
    folds = month_folds(P.p.year_month.to_numpy())
    lp_act, lp_avg, lw_main = crossval(P, folds)
    p_main = _prob(lp_avg)
    main_mets = metrics_by_city(P, p_main, rank=lw_main)
    record_metrics("main", "month", main_mets)
    record_metrics("main", "month", metrics_by_city(P, _prob(lp_act), rank=lw_main + (lp_act - lp_avg)),
                   dow_mode="actual")
    pd.DataFrame({"city": P.p.city, "zipcode": P.p.zipcode, "date": P.p.date, "fold": folds, "event": P.y,
                  "p_oof": p_main, "p_oof_actual_dow": _prob(lp_act)}).to_parquet(
        os.path.join(out, "oof_predictions_main.parquet"), index=False)

    # ---- leakage diagnostics (s6.1)
    leak = {}
    for scheme, labels in (("week", P.p.week.to_numpy()), ("date", P.p.date.to_numpy())):
        log(f"main fit, {scheme} folds")
        _, lpw, lww = crossval(P, shuffled_folds(labels))
        mets = metrics_by_city(P, _prob(lpw), rank=lww)
        record_metrics("main", scheme, mets)
        leak[scheme] = mets
    results["leakage"] = {c: {"month_c": main_mets[c]["zip_strat_c"], "week_c": leak["week"][c]["zip_strat_c"],
                              "date_c": leak["date"][c]["zip_strat_c"],
                              "date_minus_month": leak["date"][c]["zip_strat_c"] - main_mets[c]["zip_strat_c"]}
                          for c in ("pgh", "lou")}

    # ---- baselines (s6.2)
    base_probs, base_rank = {}, {}
    for name, spec in (("B0", replace(MAIN, name="B0", weather=(), source=())),
                       ("B1", replace(MAIN, name="B1", weather=(), source=("multi_source_exposure",))),
                       ("B2", replace(MAIN, name="B2", weather=("harm_sin", "harm_cos"),
                                      source=("multi_source_exposure",)))):
        log(f"baseline {name}")
        dd = d.assign(harm_sin=np.sin(2 * np.pi * d.doy / 365.25), harm_cos=np.cos(2 * np.pi * d.doy / 365.25))
        PB = prepare(dd, spec, t0)
        _, lpb, base_rank[name] = crossval(PB, folds)
        base_probs[name] = _prob(lpb)
        record_metrics(name, "month", metrics_by_city(PB, base_probs[name], rank=base_rank[name]))
    results["main_minus_B2"] = paired_bootstrap(P, p_main, base_probs["B2"], draws, rank_a=lw_main,
                                                rank_b=base_rank["B2"])

    # ---- alert threshold (s6.4, M14)
    log("alert threshold (nested)")
    thr, f1 = f1_threshold(P.y, p_main, P.w)
    per_city = {c: dict(zip(("threshold", "f1"), f1_threshold(P.y[m], p_main[m], np.ones(m.sum()))))
                for c in ("pgh", "lou") for m in [(P.p.city == c).to_numpy()]}
    nested_pred = np.zeros(len(P.y), bool)
    nested_thr = {}
    for k in range(N_FOLDS):
        outer_tr = folds != k
        inner = np.full(len(P.y), np.nan)
        for j in range(N_FOLDS):
            if j == k:
                continue
            tr, te = outer_tr & (folds != j), folds == j
            fj, kj = fit_rows(P, tr)
            inner[te] = predict_rows(P, fj, kj, tr, te)[1]
        m = outer_tr
        tk, _ = f1_threshold(P.y[m], _prob(inner[m]), P.w[m])
        nested_thr[k] = tk
        nested_pred[folds == k] = p_main[folds == k] >= tk
    results["threshold"] = {"pooled_equal_city_weight": thr, "pooled_f1": f1, "per_city": per_city,
                            "nested_fold_thresholds": nested_thr,
                            "nested_operating": {c: operating(P.y[m], nested_pred[m].astype(float), 0.5)
                                                 for c in ("pgh", "lou") for m in [(P.p.city == c).to_numpy()]},
                            "deployed_threshold_for_reference": 0.2113}

    # ---- deployed model (s6.5, M10)
    log("deployed model")
    results["deployed_a"] = metrics_by_city(P, _prob(deployed_lp(P.p)), ranking_only=True)
    record_metrics("deployed_as_is", "none (in-sample dates)", results["deployed_a"])
    PD = prepare(d, replace(MAIN, name="deployed_slopes", weather=(), source=()), t0)
    dep_probs, dep_rank = {}, {}
    for label, scale in (("true_feet", 1.0), ("fitted_scale", BLH_FITTED_SCALE)):
        off = deployed_lp(PD.p, blh_scale=scale, with_const=False)
        _, lpd, dep_rank[label] = crossval(PD, folds, offset=off)
        dep_probs[label] = _prob(lpd)
        record_metrics(f"deployed_slopes_{label}", "month", metrics_by_city(PD, dep_probs[label],
                                                                            rank=dep_rank[label]))
    results["main_minus_deployed_slopes"] = paired_bootstrap(P, p_main, dep_probs["true_feet"], draws,
                                                             rank_a=lw_main, rank_b=dep_rank["true_feet"])

    # ---- transport and city-level checks (s6.6, M8)
    log("forward by year")
    names = np.array(P.D.names)
    fwd = []
    for Y in FORWARD_YEARS:
        tr, te = (P.p.year < Y).to_numpy(), (P.p.year == Y).to_numpy()
        fy, ky = fit_rows(P, tr)
        beta = dict(zip(names[ky], fy["beta"]))
        prev = {c: beta[f"cy:{c}:{Y - 1}"] for c in ("pgh", "lou")}
        for rule, icpt in (("A_own_previous_year", None), ("B_two_city_mean", (prev["pgh"] + prev["lou"]) / 2)):
            for c in ("pgh", "lou"):
                tc = te & (P.p.city == c).to_numpy()
                ic = prev[c] if icpt is None else icpt
                _, lp, _ = predict_rows(P, fy, ky, tr, tc, intercept=ic)
                fwd.append({"test_year": Y, "rule": rule, "city": c,
                            **city_metrics(P.y[tc], _prob(lp), P.strata[tc])})
    results["forward"] = fwd
    log("leave one city out and per-city fits")
    loco = []
    for fit_city, test_city in (("pgh", "lou"), ("lou", "pgh")):
        tr, te = (P.p.city == fit_city).to_numpy(), (P.p.city == test_city).to_numpy()
        fc, kc = fit_rows(P, tr)
        icpt = dict(zip(names[kc], fc["beta"]))[f"cy:{fit_city}:{CALVERT_YEAR}"]
        _, lp, _ = predict_rows(P, fc, kc, tr, te, intercept=icpt)
        loco.append({"fit_city": fit_city, "test_city": test_city, **city_metrics(P.y[te], _prob(lp), P.strata[te])})
        PC = prepare(d, replace(MAIN, name=f"city_{fit_city}", cities=(fit_city,)), t0)
        fcc, kcc, Vc, _ = full_fit(PC)
        coef_rows.append(coefficient_table(PC, fcc, kcc, Vc).assign(fit=f"city_{fit_city}_only"))
    results["leave_one_city_out"] = loco
    log("city x weather interaction")
    PI = prepare(d, replace(MAIN, name="interaction", interact=WEATHER + ("wind_align_weighted",)), t0)
    fi, ki, Vi, _ = full_fit(PI, ("twoway",))
    ci = coefficient_table(PI, fi, ki, Vi)
    coef_rows.append(ci.assign(fit="city_interaction"))
    Vi2 = to_original_scale(fi["beta"], Vi["twoway"], PI.D.scale[ki])[1]
    results["interaction_wald"] = wald(ci, Vi2, [f"lou_x_{t}" for t in WEATHER + ("wind_align_weighted",)])
    results["replacement_checks"] = replacement_checks(results)

    # ---- sensitivities (s6.7)
    sens = {}
    for spec in sensitivity_specs():
        log(f"sensitivity {spec.name}")
        PS = prepare(d, spec, t0)
        fs, ks, Vs, _ = full_fit(PS)
        cs = coefficient_table(PS, fs, ks, Vs)
        coef_rows.append(cs.assign(fit=spec.name))
        sens[spec.name] = {"n": len(PS.y), "events": int(PS.y.sum()), "thresholds": PS.thresholds}
        if spec.cv:
            fs_folds = month_folds(PS.p.year_month.to_numpy())
            _, lps, lws = crossval(PS, fs_folds)
            ps = _prob(lps)
            record_metrics(spec.name, "month", metrics_by_city(PS, ps, rank=lws))
            if spec.name == "original_61_zips":
                shared = (P.p.city == "pgh").to_numpy() & P.p.zip_in_original_set.to_numpy()
                key = PS.p.zipcode + "|" + PS.p.date
                pos = pd.Series(np.arange(len(PS.y)), index=key)
                sub = pos.loc[(P.p.zipcode + "|" + P.p.date)[shared]].to_numpy()
                sens[spec.name]["pgh_61_shared_zip_strat_c"] = {
                    "main_fit": StratifiedC(P.y[shared], lw_main[shared], P.strata[shared]).value(),
                    "original_61_fit": StratifiedC(PS.y[sub], lws[sub], PS.strata[sub]).value()}
    results["sensitivities"] = sens
    results["sewer_reports_removed"] = n_sewer
    coefs = pd.concat(coef_rows, ignore_index=True)
    results["readings"] = readings(coefs)

    # ---- deployment-form coefficients (not deployed; s7)
    results["deployment_form"] = deployment_form(main_coef, P, t0)

    write_outputs(out, coefs, pd.DataFrame(metric_rows), results, plan_sha, t0, time.time() - t_start)
    log(f"done in {(time.time() - t_start) / 60:.1f} min")
    return results


def vif(p, terms):
    c = np.corrcoef(p[list(terms)].to_numpy(float), rowvar=False)
    return dict(zip(terms, np.diag(np.linalg.inv(c)).round(2).tolist()))


def replacement_checks(results):
    """s6.6: (a) main minus B2 zip-stratified concordance CI above 0; (b) rule-B forward concordance > 0.5 every year."""
    out = {}
    for c in ("pgh", "lou"):
        a = results["main_minus_B2"][c]["ci_zip_strat_c"][0] > 0
        fw = [r["zip_strat_c"] for r in results["forward"] if r["city"] == c and r["rule"] == "B_two_city_mean"]
        b = all(x > 0.5 for x in fw)
        out[c] = {"a_beats_climatology": bool(a), "b_forward_above_half": bool(b), "pass": bool(a and b)}
    out["both_pass"] = all(out[c]["pass"] for c in ("pgh", "lou"))
    return out


def readings(coefs):
    """s6.7 reading rules for zip fixed effects, leave-one-zip-out, negative control; s5 expected signs."""
    main = coefs[coefs.fit == "main"].set_index("term")
    terms = [t for t in WEATHER + ("wind_align_weighted",) if t in main.index]
    out = {"expected_signs": [], "zip_fixed_effects": [], "negative_control": [], "leave_one_zip_out": []}
    for t in list(terms) + ["multi_source_exposure"]:
        e = EXPECTED_SIGN[t]
        s = int(np.sign(main.coef[t]))
        out["expected_signs"].append({"term": t, "expected": e, "estimate": float(main.coef[t]),
                                      "ci": [float(main.ci_low[t]), float(main.ci_high[t])],
                                      "matches": None if e == 0 else bool(s == e)})
    fe = coefs[coefs.fit == "zip_fixed_effects"].set_index("term")
    nc = coefs[coefs.fit == "negative_control"].set_index("term")
    for t in terms:
        b, lo, hi = main.coef[t], main.ci_low[t], main.ci_high[t]
        out["zip_fixed_effects"].append({"term": t, "main": float(b), "fe": float(fe.coef[t]),
                                         "reporting_sensitive": bool(np.sign(fe.coef[t]) != np.sign(b)
                                                                     or not lo <= fe.coef[t] <= hi)})
        same = np.sign(nc.coef[t]) == np.sign(b)
        big = abs(nc.coef[t]) >= 0.5 * abs(b)
        excl0 = nc.ci_low[t] > 0 or nc.ci_high[t] < 0
        out["negative_control"].append({"term": t, "main": float(b), "negative_control": float(nc.coef[t]),
                                        "reporting_behaviour": bool(same and big and excl0)})
        loz = coefs[coefs.fit.str.startswith("leave_out_") & (coefs.term == t)]
        dev = (loz.coef - b).abs() / main.se_twoway[t]
        i = dev.idxmax()
        out["leave_one_zip_out"].append({"term": t, "max_change_in_se": float(dev.max()),
                                         "fit": loz.fit[i]})
    return out


def deployment_form(main_coef, P, t0):
    """Main-fit coefficients in the deployed raw-temperature form, with the Calvert constant of s7 (not deployed)."""
    b = main_coef.set_index("term").coef
    shares = np.asarray(P.D.X[:, P.D.dow_idx].mean(axis=0)).ravel()
    dow_avg = float(shares @ b[list(DOW)].to_numpy())
    icpt = (b[f"cy:pgh:{CALVERT_YEAR}"] + b[f"cy:lou:{CALVERT_YEAR}"]) / 2
    bt, bt2 = b["temp_c"], b["temp_c2"]
    raw = {"const": float(icpt + dow_avg + bt2 * t0 ** 2 - bt * t0), "temperature": float(bt - 2 * bt2 * t0),
           "temperature_squared": float(bt2)}
    for t in WEATHER[2:] + SOURCE:
        raw[t] = float(b[t])
    return {"note": "NOT deployed. Refit main fit in the deployed raw-temperature form; pressure_anomaly is the "
                    "anomaly against each location's own ERA5-cell mean (s7).",
            "temperature_centre_F": t0, "calvert_intercept_two_city_2025_mean": float(icpt),
            "average_day_of_week_and_holiday_effect": dow_avg, "coefficients": raw}


# ------------------------------------------------------------------------------------------------ outputs

def write_outputs(out, coefs, metrics, results, plan_sha, t0, seconds):
    coefs.to_csv(os.path.join(out, "coefficients.csv"), index=False)
    metrics.to_csv(os.path.join(out, "oof_metrics.csv"), index=False)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=_json_default)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = None
    manifest = {"run_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "plan_sha256": plan_sha, "script_sha256": _sha256(os.path.abspath(__file__)),
                "report_script_sha256": _sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_report.py")),
                "panel_sha256": _sha256(PANEL), "git_commit": commit, "seed": SEED, "folds": N_FOLDS,
                "bootstrap_resamples": N_BOOT, "temperature_centre_F": t0, "minutes": round(seconds / 60, 1),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__}
    with open(os.path.join(out, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    from training.fit_report import write_report
    write_report(out, coefs, metrics, results, manifest)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    run()
    sys.exit(0)
