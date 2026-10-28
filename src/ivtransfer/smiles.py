"""SVI smiles per expiry and a global power-law SSVI surface per day (paper Section 3.3).

Two SSVI fits are run on the same inputs. ``fit_ssvi_penalised`` is the fit of the original DeepVol library (L-BFGS-B with soft
penalties); ``fit_ssvi_constrained`` enforces eta (1 + |rho|) <= 2 and 0 < gamma < 0.5 exactly through its parameterisation and is
the one whose fit cost and risk-neutral density enter the sample weights. The penalised fit is kept because the density step runs
only on the expiries for which it produced a usable density (see ``surfaces.fit_day``), so dropping it would change which samples
get a martingale check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_w(k, a, b, rho, m, sigma):
    """Total implied variance of a raw SVI smile (Gatheral 2004)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_vol(k, T, a, b, rho, m, sigma):
    return np.sqrt(np.maximum(1e-9, svi_w(k, a, b, rho, m, sigma)) / T)


def _svi_objective(params, k, T, target_vol):
    a, b, rho, m, sigma = params
    if sigma <= 0 or b < 0 or abs(rho) > 1:
        return np.inf * np.ones_like(k)
    return (svi_vol(k, T, a, b, rho, m, sigma) - target_vol) ** 2


def fit_svi(df_expiry: pd.DataFrame):
    """Least-squares SVI fit of one expiry's implied vols (columns k, T, iv_clean). Returns SVIParams or None."""
    if df_expiry.empty:
        return None
    k = df_expiry['k'].values; T = float(df_expiry['T'].iloc[0]); target_vol = df_expiry['iv_clean'].values
    x0 = [max(1e-6, np.min(target_vol) ** 2 * T * 2), 0.1, -0.5, 0.0, 0.1]
    bounds = ([1e-6, 0, -0.999, -np.inf, 1e-3], [np.inf, np.inf, 0.999, np.inf, np.inf])
    try:
        res = least_squares(_svi_objective, x0, bounds=bounds, args=(k, T, target_vol), method='trf', max_nfev=1000)
    except Exception:
        return None
    return SVIParams(*res.x) if res.success else None


def atm_metrics(p: SVIParams, T: float):
    """ATM implied vol, skew and curvature of an SVI smile at k = 0."""
    w0 = svi_w(0.0, p.a, p.b, p.rho, p.m, p.sigma)
    wp = p.b * (p.rho + (0.0 - p.m) / np.sqrt((0.0 - p.m) ** 2 + p.sigma ** 2))
    wpp = p.b * (p.sigma ** 2) / (((0.0 - p.m) ** 2 + p.sigma ** 2) ** 1.5)
    iv = np.sqrt(max(w0, 0.0) / max(T, 1e-12))
    skew = 0.5 * wp / np.sqrt(max(T * w0, 1e-12))
    curv = 0.5 / np.sqrt(max(T, 1e-12)) * (-(wp ** 2) / (4 * (w0 ** 1.5 + 1e-18)) + (wpp / (2 * np.sqrt(w0 + 1e-18))))
    return float(iv), float(skew), float(curv)


# ---------------- SSVI ----------------
EPS_T = 1.0 / 365.0
EPS_SIG = 1e-12


def w_ssvi(k, theta, rho, eta, gamma):
    """Power-law SSVI total variance (Gatheral and Jacquier 2014)."""
    ph = eta * (theta ** (-gamma))
    return 0.5 * theta * (1.0 + rho * ph * k + np.sqrt((ph * k + rho) ** 2 + (1.0 - rho ** 2)))


def theta_atm(expiry_slice: pd.DataFrame, F: float) -> float:
    """ATM total variance of an expiry: vol of the strike closest to the forward."""
    idx = (expiry_slice['strike'] - F).abs().idxmin()
    s = float(expiry_slice.loc[idx, 'mid_iv' if 'mid_iv' in expiry_slice.columns else 'iv_clean'])
    T = float(expiry_slice.loc[idx, 'T'])
    return max(s, EPS_SIG) ** 2 * max(T, EPS_SIG)


def monotone_theta(thetas: dict, asof: pd.Timestamp) -> dict:
    """Running maximum of theta across maturities (no calendar arbitrage in the ATM term structure)."""
    exps = sorted(thetas.keys(), key=lambda e: (e - asof).days)
    th = np.maximum.accumulate(np.array([thetas[e] for e in exps]))
    return {e: t for e, t in zip(exps, th)}


def ssvi_weights(df: pd.DataFrame, cap_percentile: float = 90.0) -> np.ndarray:
    """Fit weights: inverse spread capped at its 90th percentile, liquidity bumps, extreme |k| down-weighted."""
    spread = df.get('spread', pd.Series(1.0, index=df.index))
    spread = spread.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-6)
    w = 1.0 / spread
    w = np.minimum(w, np.nanpercentile(w, cap_percentile))
    if 'oi' in df.columns:
        w *= (1.0 + 0.5 * df['oi'].rank(pct=True).fillna(0.0))
    if 'volume' in df.columns:
        w *= (1.0 + 0.25 * df['volume'].rank(pct=True).fillna(0.0))
    k_abs = df['k_fwd'].abs() if 'k_fwd' in df.columns else df['k'].abs() if 'k' in df.columns else pd.Series(0.0, index=df.index)
    w *= (1.0 - 0.5 * (k_abs > np.nanpercentile(k_abs, 99.0)))
    return w.values


def _fit_frame(df, thetas, forwards, cap_percentile=90.0):
    d = df[df['T'] > EPS_T].copy()
    d['theta'] = d['expiry'].map(thetas)
    d['w_mkt'] = (d['mid_iv' if 'mid_iv' in d.columns else 'iv_clean'] ** 2) * d['T']
    d['wgt'] = ssvi_weights(d, cap_percentile)
    k_col = 'k_fwd' if 'k_fwd' in d.columns else 'k'
    if k_col not in d.columns:
        d['F'] = d['expiry'].map(forwards); d['k_fwd'] = np.log(d['strike'] / d['F']); k_col = 'k_fwd'
    return d, k_col


def _penalised_objective(params, k, w_mkt, theta, wgt):
    rho, eta, gamma = params
    err = np.mean(wgt * (w_ssvi(k, theta, rho, eta, gamma) - w_mkt) ** 2)
    pen = 0.0
    if abs(rho) >= 1.0: pen += 1e6 * (abs(rho) - 0.999) ** 2
    if gamma <= 0.0: pen += 1e6 * (abs(gamma) + 0.01) ** 2
    if gamma > 0.5: pen += 1e6 * (gamma - 0.5) ** 2
    if eta <= 0: pen += 1e6 * (abs(eta) + 0.01) ** 2
    gj = eta * (1.0 + abs(rho))
    if gj > 2.0: pen += 1000.0 * (gj - 2.0) ** 2
    return err + pen


def fit_ssvi_penalised(df, thetas, forwards, cap_percentile=90.0):
    """Original library fit: one (rho, eta, gamma) for the day, L-BFGS-B from (-0.4, 0.5, 0.4). Returns (rho, eta, gamma, cost)."""
    d, k_col = _fit_frame(df, thetas, forwards, cap_percentile)
    args = tuple(d[c].values.astype(float) for c in (k_col, 'w_mkt', 'theta', 'wgt'))
    res = minimize(_penalised_objective, [-0.4, 0.5, 0.4], args=args, method='L-BFGS-B', bounds=[(-0.999, 0.999), (0.001, 5.0), (0.001, 0.5)])
    rho, eta, gamma = res.x
    return float(rho), float(eta), float(gamma), float(res.fun)


# constrained fit: rho = 0.999 tanh(a), eta = 2 sigmoid(b) / (1 + |rho|), gamma = 0.001 + 0.499 sigmoid(c); three starting points
STARTS = ((-0.5, 0.8, 0.4), (-0.2, 0.4, 0.2), (-0.8, 0.95, 0.45))      # (rho, s = eta (1 + |rho|) / 2, gamma)


def _unpack(x):
    rho = 0.999 * np.tanh(x[0]); eta = 2.0 / (1.0 + np.exp(-x[1])) / (1.0 + np.sqrt(rho * rho + 1e-8))
    return float(rho), float(eta), float(0.001 + 0.499 / (1.0 + np.exp(-x[2])))


def _pack(rho, s, gamma):
    s = np.clip(s, 1e-3, 0.999); g_ = np.clip((gamma - 0.001) / 0.499, 1e-3, 0.999)
    return np.array([np.arctanh(np.clip(rho, -0.99, 0.99) / 0.999), np.log(s / (1 - s)), np.log(g_ / (1 - g_))])


def ssvi_arrays(df, thetas, forwards, cap_percentile=90.0):
    """Rows and weights of the fit, restricted to rows with a finite theta, k and weight."""
    d, k_col = _fit_frame(df, thetas, forwards, cap_percentile)
    a = [d[c].to_numpy(float) for c in (k_col, 'w_mkt', 'theta', 'wgt', 'T')]
    ok = np.all([np.isfinite(v) for v in a], axis=0) & (a[2] > 0) & (a[3] > 0)
    return [v[ok] for v in a]


def rmse_vol(arrays, rho, eta, gamma):
    """Fit error in vol points (RMSE of sqrt(w / T))."""
    k, wm, th, wg, T = arrays
    if len(k) == 0:
        return np.nan
    w = w_ssvi(k, th, rho, eta, gamma)
    return float(np.sqrt(np.mean((np.sqrt(np.maximum(w, 1e-12) / T) - np.sqrt(np.maximum(wm, 0) / T)) ** 2)) * 100)


def fit_ssvi_constrained(df, thetas, forwards, cap_percentile=90.0):
    """Arbitrage-free SSVI fit, errors relative to each expiry's theta. Returns (dict or None, arrays).
    fit_cost = weighted mean of ((w_model - w_mkt) / theta)^2."""
    arr = ssvi_arrays(df, thetas, forwards, cap_percentile); k, wm, th, wg, T = arr
    if len(k) < 5:
        return None, arr
    sw = np.sqrt(wg / wg.mean())
    res = lambda x: sw * (w_ssvi(k, th, *_unpack(x)) - wm) / th
    best = None
    for r0, s0, g0 in STARTS:
        o = least_squares(res, _pack(r0, s0, g0), method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
        if best is None or o.cost < best.cost:
            best = o
    rho, eta, gamma = _unpack(best.x)
    return {'rho': rho, 'eta': eta, 'gamma': gamma, 'fit_cost': float(np.mean(res(best.x) ** 2)),
            'rmse_vol': rmse_vol(arr, rho, eta, gamma), 'gj': eta * (1 + abs(rho))}, arr
