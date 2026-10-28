"""Fast unit tests (seconds): python -m pytest tests"""
import numpy as np
import pandas as pd

from ivtransfer import pricing, smiles, universe
from ivtransfer.config import TrainConfig
from ivtransfer.model import n_parameters
from ivtransfer.train import masks


def test_vectorised_iv_matches_brent():
    rng = np.random.default_rng(1); n = 300
    K = 100 * np.exp(rng.normal(0, 0.3, n)); call = rng.random(n) < 0.5; T, r, q = 0.25, 0.10, 0.02
    px = pricing.bs_price(call, np.full(n, 100.0), K, r, q, rng.uniform(0.05, 1.5, n), np.full(n, T)) * rng.uniform(0.7, 1.2, n)
    v1 = pricing.implied_vol_vec(call, 100.0, K, r, q, T, px)
    v2 = np.array([pricing.implied_vol(bool(c), 100.0, k, r, q, T, p) for c, k, p in zip(call, K, px)])
    assert np.array_equal(np.isnan(v1), np.isnan(v2)) and np.nanmax(np.abs(v1 - v2)) < 1e-6


def test_svi_recovers_a_smile():
    k = np.linspace(-0.3, 0.3, 15); T = 0.1; true = (0.002, 0.05, -0.4, 0.02, 0.15)
    df = pd.DataFrame({'k': k, 'T': T, 'iv_clean': smiles.svi_vol(k, T, *true)})
    p = smiles.fit_svi(df)
    assert p is not None and np.max(np.abs(smiles.svi_vol(k, T, p.a, p.b, p.rho, p.m, p.sigma) - df['iv_clean'])) < 5e-3   # within half a vol point (default tolerances of the original library)


def test_constrained_ssvi_is_arbitrage_free():
    rng = np.random.default_rng(0); rows = []
    for T, th in ((0.05, 0.004), (0.1, 0.008), (0.2, 0.017)):
        for k in np.linspace(-0.25, 0.2, 12):
            w = smiles.w_ssvi(k, th, -0.6, 1.8, 0.4) * (1 + 0.01 * rng.standard_normal())
            rows.append({'expiry': pd.Timestamp('2025-01-01') + pd.Timedelta(days=int(T * 365)), 'T': T, 'k': k, 'iv_clean': np.sqrt(w / T), 'strike': 100 * np.exp(k)})
    df = pd.DataFrame(rows); thetas = {e: g['iv_clean'].iloc[len(g) // 2] ** 2 * g['T'].iloc[0] for e, g in df.groupby('expiry')}
    fx, _ = smiles.fit_ssvi_constrained(df, thetas, {})
    assert fx is not None and fx['eta'] * (1 + abs(fx['rho'])) <= 2 + 1e-12 and 0 < fx['gamma'] < 0.5


def test_network_size():
    assert n_parameters(TrainConfig()) == 3464


def test_ex_ante_universe():
    rows = [(n, y) for y in range(2020, 2027) for n in ('A', 'B')] + [('C', 2024), ('C', 2025), ('D', 2026)]
    U = universe.build(pd.DataFrame(rows, columns=['name', 'year']), (2025, 2026))
    add = set(map(tuple, U.loc[U['added'], ['name', 'year']].to_numpy()))
    test = set(map(tuple, U.loc[U['xa_test'], ['name', 'year']].to_numpy()))
    assert add == {('C', 2026)}                              # in the 2025 list, not in 2026's: a 2026 test name only
    assert ('D', 2026) not in test and ('C', 2026) in test   # entered on 2026 liquidity: not tested in 2026


def test_walk_forward_masks():
    d = pd.to_datetime(pd.date_range('2019-01-02', '2025-12-31', freq='W-WED'))
    D = {'asof': pd.Series(d), 'afut': pd.Series(d + pd.Timedelta(days=14)), 'tk': np.array(['X'] * len(d))}
    TR, VA, TE, ok = masks(D, 2025, min_n=1)
    assert D['afut'][TR].max() <= pd.Timestamp('2023-12-31') and D['asof'][VA].min() > pd.Timestamp('2023-12-31')
    assert D['afut'][VA].max() <= pd.Timestamp('2024-12-31') and set(D['asof'][TE].dt.year) == {2025}
