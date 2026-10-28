"""Black-Scholes prices and implied volatilities (paper Section 3.3)."""
from __future__ import annotations

from math import exp, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

SIGMA_THRESHOLD = 1e-5


def bs_price(is_call, S, K, r, q, sigma, T):
    """Vectorised Black-Scholes price with continuous dividend yield q. Very small or invalid sigma / T / S / K -> intrinsic value."""
    is_call, S, K, sigma, T = (np.asarray(a) for a in (is_call, S, K, sigma, T))
    mask_small = sigma < SIGMA_THRESHOLD
    mask_valid = (sigma > 0) & (T > 0) & (S > 0) & (K > 0)
    prices = np.zeros_like(S, dtype=float)
    intr_c = np.maximum(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    intr_p = np.maximum(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
    if not mask_valid.any():
        return np.where(is_call, intr_c, intr_p)
    m = mask_small & mask_valid
    prices[m] = np.where(is_call[m], intr_c[m], intr_p[m])
    calc = mask_valid & ~mask_small
    if calc.any():
        ratio = S[calc] / K[calc]
        ratio = np.where(ratio > 0, ratio, 1e-10)
        sq = np.sqrt(T[calc])
        d1 = (np.log(ratio) + (r - q + 0.5 * sigma[calc] ** 2) * T[calc]) / np.maximum(sigma[calc] * sq, 1e-10)
        d2 = d1 - sigma[calc] * sq
        call = S[calc] * np.exp(-q * T[calc]) * norm.cdf(d1) - K[calc] * np.exp(-r * T[calc]) * norm.cdf(d2)
        put = K[calc] * np.exp(-r * T[calc]) * norm.cdf(-d2) - S[calc] * np.exp(-q * T[calc]) * norm.cdf(-d1)
        prices[calc] = np.where(is_call[calc], call, put)
    prices[~mask_valid] = np.where(is_call[~mask_valid], intr_c[~mask_valid], intr_p[~mask_valid])
    return prices


def implied_vol(is_call, S, K, r, q, T, price, lo=1e-6, hi=5.0):
    """Reference implied volatility for one option (Brent's method). NaN when the price is below 0.9 x intrinsic or no root exists
    in [lo, hi], [lo, 10], [lo, 20] or [lo, 50]. Used to check ``implied_vol_vec``."""
    if S <= 0 or K <= 0 or T <= 0 or price < 0:
        return np.nan
    intrinsic = (max(0.0, S * exp(-q * T) - K * exp(-r * T)) if is_call else max(0.0, K * exp(-r * T) - S * exp(-q * T))) * 0.9
    if price < intrinsic:
        return np.nan

    def f(sig):
        p = bs_price(np.array([is_call]), np.array([S]), np.array([K]), r, q, np.array([sig]), np.array([T]))[0]
        return p - price if np.isfinite(p) else np.nan
    try:
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            return np.nan
        if flo * fhi > 0:
            for hi2 in (10.0, 20.0, 50.0):
                fhi2 = f(hi2)
                if np.isfinite(fhi2) and flo * fhi2 <= 0:
                    hi = hi2
                    break
            else:
                return np.nan
        return brentq(f, lo, hi, maxiter=200, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_vec(is_call, S, K, r, q, T, price, lo=1e-6, hi=5.0):
    """Implied vols of many options with one S, T, r, q: same rules and root as ``implied_vol``, found by 60 bisection steps."""
    is_call = np.asarray(is_call, bool); K = np.asarray(K, float); price = np.asarray(price, float)
    n = len(price); out = np.full(n, np.nan)
    if n == 0 or S <= 0 or T <= 0:
        return out
    T_ = np.full(n, float(T))
    intrinsic = np.where(is_call, np.maximum(0.0, S * exp(-q * T) - K * exp(-r * T)),
                         np.maximum(0.0, K * exp(-r * T) - S * exp(-q * T))) * 0.9
    ok = (K > 0) & (price >= 0) & ~(price < intrinsic)
    Ks = np.where(K > 0, K, 1.0); dS, dK, sqT = S * exp(-q * T), Ks * np.exp(-r * T_), sqrt(T)
    intr_c, intr_p = np.maximum(0.0, dS - dK), np.maximum(0.0, dK - dS)

    def f(sig):   # bs_price(...) - price for S, K, T > 0 (ndtr is what norm.cdf evaluates)
        sg = np.where(sig < SIGMA_THRESHOLD, 1.0, sig)
        d1 = (np.log(S / Ks) + (r - q + 0.5 * sg ** 2) * T_) / np.maximum(sg * sqT, 1e-10); d2 = d1 - sg * sqT
        p = np.where(is_call, dS * ndtr(d1) - dK * ndtr(d2), dK * ndtr(-d2) - dS * ndtr(-d1))
        return np.where(sig < SIGMA_THRESHOLD, np.where(is_call, intr_c, intr_p), p) - price
    flo = f(np.full(n, lo)); hi_v = np.full(n, hi); fhi = f(hi_v)
    ok &= np.isfinite(flo) & np.isfinite(fhi)
    need = ok & (flo * fhi > 0)
    for hi2 in (10.0, 20.0, 50.0):
        if not need.any():
            break
        f2 = f(np.full(n, hi2)); take = need & np.isfinite(f2) & (flo * f2 <= 0)
        hi_v[take] = hi2; need &= ~take
    ok &= ~need
    a, b, fa = np.full(n, lo), hi_v.copy(), flo.copy()
    for _ in range(60):
        m = 0.5 * (a + b); fm = f(m); left = fa * fm <= 0
        b = np.where(left, m, b); a = np.where(left, a, m); fa = np.where(left, fa, fm)
    out[ok] = (0.5 * (a + b))[ok]
    return out
