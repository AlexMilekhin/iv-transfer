"""Risk-neutral density of an SSVI slice (Breeden-Litzenberger) and its martingale error, used in the sample weights (Section 3.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import norm

K_MIN_MULT, K_MAX_MULT, N_POINTS = 0.1, 4.0, 5000


def _ssvi_power_law(k, theta, rho, eta, gamma):
    z = eta / (theta ** gamma) * k
    return (theta / 2) * (1 + rho * z + np.sqrt((z + rho) ** 2 + (1 - rho ** 2)))


def _bs_call(S, K, T, r, sigma):
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def density_moments(T, theta, rho, eta, gamma, S0, r0):
    """Moments of the risk-neutral density of one expiry; None if T < 0.005 or the grid holds < 95% of the probability."""
    if T < 0.005:
        return None
    F = S0 * np.exp(r0 * T)
    K_grid = np.linspace(S0 * K_MIN_MULT, S0 * K_MAX_MULT, N_POINTS); h = K_grid[1] - K_grid[0]
    w_grid = _ssvi_power_law(np.log(K_grid / F), theta, rho, eta, gamma)
    calls = _bs_call(S0, K_grid, T, r0, np.sqrt(np.maximum(1e-9, w_grid) / T))
    d2C = (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / (h ** 2)
    f_grid = np.maximum(0, np.exp(r0 * T) * d2C)
    K_pdf = K_grid[1:-1]
    total = simpson(y=f_grid, x=K_pdf)
    if total < 0.95:
        return None
    pdf = f_grid / total
    mean = simpson(y=K_pdf * pdf, x=K_pdf)
    var = simpson(y=((K_pdf - mean) ** 2) * pdf, x=K_pdf); vol = np.sqrt(var)
    return {'F': F, 'rnd_mean': mean, 'martingale_error': abs(mean - F) / F,
            'rnd_vol': np.sqrt(simpson(y=((K_pdf / F - 1.0) ** 2) * pdf, x=K_pdf) / T),
            'rnd_skew': simpson(y=((K_pdf - mean) ** 3) * pdf, x=K_pdf) / (vol ** 3),
            'rnd_kurtosis': simpson(y=((K_pdf - mean) ** 4) * pdf, x=K_pdf) / (vol ** 4)}


def extract(ssvi_rows: pd.DataFrame, S0: float, r0: float) -> pd.DataFrame:
    """Density moments per expiry from rows with columns expiry, T, theta, ssvi_rho, ssvi_eta, ssvi_p."""
    out = []
    for _, row in ssvi_rows.iterrows():
        res = density_moments(row['T'], row['theta'], row['ssvi_rho'], row['ssvi_eta'], row['ssvi_p'], S0, r0)
        if res:
            out.append({'expiry': row['expiry'], 'T': row['T'], 'F': res['F'], 'rnd_mean': res['rnd_mean'],
                        'martingale_error': res['martingale_error'], 'rnd_vol': res['rnd_vol'], 'rnd_skew': res['rnd_skew'],
                        'rnd_kurtosis': res['rnd_kurtosis']})
    return pd.DataFrame(out) if out else pd.DataFrame()
