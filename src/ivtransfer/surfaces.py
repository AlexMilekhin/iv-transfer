"""From option quotes to daily smile fits, one underlying-year ("unit") at a time (paper Section 3.3).

For every grid date of a unit: clean the quotes, compute implied vols, fit an SVI smile per expiry (ATM vol, skew, curvature),
fit a power-law SSVI surface for the day (penalised and constrained fits) and the risk-neutral density of each expiry.
Each unit is written as svi / ssvi / rnd / errors parquet files plus a ``.done`` marker, so a run can be resumed.
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from . import density, smiles
from .config import GRID_WEEKDAYS, SPECS, Layout
from .pricing import implied_vol_vec

logger = logging.getLogger('ivtransfer')
MIN_OPTIONS_PER_EXPIRY = 5
CHAIN_V2_COLS = ['date', 'underlying', 'expiration', 'strike', 'call_put', 'avg', 'trades', 'S_avg', 'r']


# ======================= quotes =======================
def quotes_chain_v2(path: str, ticker: str) -> Optional[pd.DataFrame]:
    """Exchange chain file (Brazil, India, Hong Kong, Canada): one price per option and day (average trade price or settlement),
    used as bid = ask = mid (equal spread weights); number of trades as volume; the underlying's price S and the rate r per row."""
    if not os.path.exists(path):
        return None
    q = pd.read_parquet(path, columns=CHAIN_V2_COLS, filters=[('underlying', '==', ticker)])
    return pd.DataFrame({'date': q.date, 'expiration': q.expiration, 'strike': q.strike, 'call_put': q.call_put, 'bid': q.avg,
                         'ask': q.avg, 'vol': q.trades.astype(float), 'S': q.S_avg, 'r': q.r})


def us_rates(path: str) -> pd.Series:
    """FRED DTB3 (3-month Treasury bill), forward-filled, continuously compounded."""
    return np.log1p(pd.read_parquet(path).set_index('date')['dtb3'].ffill() / 100)


def parity_spot(g: pd.DataFrame) -> float:
    """Spot implied by put-call parity on the nearest expiry of at least 7 days: median over the 3 strikes with the smallest |C - P|
    of C - P + K exp(-rT)."""
    g7 = g[g['T'] >= 7 / 365]
    if g7.empty: return np.nan
    q_ = g7[g7['expiration'] == g7['expiration'].min()]
    pv = q_.pivot_table(index='strike', columns='cp', values='mid', aggfunc='mean')
    if not {'c', 'p'} <= set(pv.columns): return np.nan
    pv = pv.dropna()
    if len(pv) < 2: return np.nan
    d_ = pv['c'] - pv['p']; idx = d_.abs().nsmallest(3).index; T_, r_ = q_['T'].iloc[0], q_['r'].iloc[0]
    return float(np.median(d_.loc[idx].to_numpy() + idx.to_numpy() * np.exp(-r_ * T_)))


def quotes_us(path: str, ticker: str, year: int, rates: pd.Series) -> Optional[pd.DataFrame]:
    """US option-chain export (DoltHub): Mon / Wed / Fri only, bid > 0 and ask >= bid, volume set to 1 (neutral weights),
    spot from put-call parity, r from DTB3."""
    q = pd.read_parquet(path, filters=[('symbol', '==', ticker)], columns=['date', 'expiration', 'strike', 'call_put', 'bid', 'ask'])
    q['date'] = pd.to_datetime(q['date']); q['expiration'] = pd.to_datetime(q['expiration'])
    q = q[(q['date'].dt.year == year) & q['date'].dt.weekday.isin(list(GRID_WEEKDAYS)) & (q['bid'] > 0) & (q['ask'] >= q['bid'])].copy()
    if q.empty: return None
    q['r'] = rates.reindex(q['date'].drop_duplicates().sort_values(), method='ffill').reindex(q['date']).to_numpy()
    q['T'] = (q['expiration'] - q['date']).dt.days / 365; q['mid'] = (q['bid'] + q['ask']) / 2; q['cp'] = q['call_put'].astype(str).str.lower().str[0]
    S_ = q.groupby('date').apply(parity_spot, include_groups=False).rename('S')
    q = q.merge(S_, left_on='date', right_index=True, how='left'); q = q[np.isfinite(q['S']) & (q['S'] > 0)]
    return pd.DataFrame({'date': q['date'], 'expiration': q['expiration'], 'strike': q['strike'], 'call_put': q['call_put'],
                         'bid': q['bid'], 'ask': q['ask'], 'vol': 1.0, 'S': q['S'], 'r': q['r']}) if len(q) else None


def load_quotes(layout: Layout, m: str, ticker: str, year: int, added: bool = False, rates: Optional[pd.Series] = None):
    path = layout.chain_file(m, year, added)
    if SPECS[m].chain_format == 'chain_v2':
        return quotes_chain_v2(path, ticker)
    return quotes_us(path, ticker, year, rates if rates is not None else us_rates(layout.rates_file(m)))


class Quotes:
    """Standardised quotes of one unit, indexed by date (the original library's historical loader, inputs already holding S and r)."""

    def __init__(self, q: pd.DataFrame):
        df = q.reset_index(drop=True)
        assert 'S' in df.columns and df['S'].notna().all() and 'r' in df.columns, 'quotes need the spot S and the rate r on every row'
        df['_date'] = pd.to_datetime(pd.to_datetime(df['date'])).dt.normalize()
        df['underlying_price'] = df['S'].astype(float)
        rr = df.groupby('_date')['r'].first().sort_index()
        self.rates = pd.DataFrame({'_date': rr.index, 'rate': rr.values.astype(float)})
        df['asof'] = pd.to_datetime(df['date'])
        df['expiry'] = pd.to_datetime(df['expiration'])
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        tc = df['call_put'].astype(str).str.lower()
        df['type'] = np.where(tc.str.contains('c'), 'call', np.where(tc.str.contains('p'), 'put', tc))
        df['bid'] = pd.to_numeric(df['bid'], errors='coerce'); df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
        if 'mid' not in df.columns:
            df['mid'] = (df['bid'] + df['ask']) / 2
        df['S'] = pd.to_numeric(df['underlying_price'], errors='coerce')
        if 'vol' in df.columns:
            df['volume'] = pd.to_numeric(df['vol'], errors='coerce')
        df['T'] = (df['expiry'] - df['asof']).dt.days / 365.0
        df = df[df['T'] > 0].copy()
        df['k'] = np.log(df['strike'] / df['S'])
        self.df = df.drop(columns=['_date'])
        self.dates = sorted(self.df['asof'].dt.date.unique())
        self._idx = self.df.groupby(self.df['asof'].dt.date, sort=False).indices

    def snapshot(self, d: date) -> pd.DataFrame:
        idx = self._idx.get(d)
        return self.df.iloc[idx].copy() if idx is not None else self.df.iloc[0:0].copy()

    def rate(self, d: date) -> float:
        target = pd.Timestamp(d).normalize(); mask = self.rates['_date'] <= target
        return float(self.rates.loc[mask, 'rate'].iloc[-1]) if mask.any() else float(self.rates['rate'].iloc[0])


# ======================= one day =======================
def infer_forward(df_exp: pd.DataFrame, r: float, mode: str) -> float:
    """Forward of one expiry. 'carry': S exp(rT). 'parity+carry': median of K + (C - P) exp(rT) over strikes quoted as both call
    and put, else S exp(rT) (with out-of-the-money quotes only, a strike rarely has both)."""
    if mode == 'carry' and {'S', 'T'}.issubset(df_exp.columns):
        return float(df_exp['S'].iloc[0]) * np.exp(r * float(df_exp['T'].iloc[0]))
    t = df_exp['type'].astype(str).str.lower().str[0].to_numpy()
    K, mid = df_exp['strike'].to_numpy(float), df_exp['mid'].to_numpy(float)
    kc, kp = K[t == 'c'], K[t == 'p']
    if len(np.unique(kc)) == len(kc) and len(np.unique(kp)) == len(kp):
        common, ic, ip = np.intersect1d(kc, kp, return_indices=True)
        F = np.nan
        if len(common):
            Fv = common + (mid[t == 'c'][ic] - mid[t == 'p'][ip]) / np.exp(-r * float(df_exp['T'].iloc[0]))
            Fv = Fv[np.isfinite(Fv)]; F = np.median(Fv) if len(Fv) > 0 else np.nan
    else:   # duplicated strikes within a type: label-based matching, as the original library
        d = df_exp.assign(_t=df_exp['type'].astype(str).str.lower().str[0])
        calls, puts = d[d['_t'] == 'c'].set_index('strike'), d[d['_t'] == 'p'].set_index('strike')
        common_ = calls.index.intersection(puts.index); F = np.nan
        if len(common_):
            Fv = common_.to_numpy(dtype=float) + (calls.loc[common_, 'mid'].values - puts.loc[common_, 'mid'].values) / np.exp(-r * float(d['T'].iloc[0]))
            Fv = Fv[np.isfinite(Fv)]; F = np.median(Fv) if len(Fv) > 0 else np.nan
    if mode == 'parity+carry' and not np.isfinite(F) and {'S', 'T'}.issubset(df_exp.columns):
        return float(df_exp['S'].iloc[0]) * np.exp(r * float(df_exp['T'].iloc[0]))
    return F


def clean_iv_surface(source_df: pd.DataFrame, r: float, asof_date: date, forward_mode: str, otm_only: bool) -> pd.DataFrame:
    """Quote filters (spread <= 0.25 under a price of 1, else <= 80% of mid; mid > 0.05; T > 0.03 years; log-moneyness in [-0.7, 0.5];
    spread < 50% of mid; optionally out-of-the-money only), forwards per expiry, implied vols (kept within 0.0005-10)."""
    df = source_df.copy()
    df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
    df = df.dropna(subset=['expiry', 'strike'])
    df['T'] = (df['expiry'].dt.normalize() - pd.Timestamp(asof_date)).dt.days / 365.0
    df = df[df['T'] > 0].copy()
    if len(df) == 0: return pd.DataFrame()
    t = df['type'].astype(str).str.lower()
    df['type'] = np.where(t.str.startswith('c'), 'call', np.where(t.str.startswith('p'), 'put', t))
    for col in ('bid', 'ask', 'lastPrice', 'last_price'):
        if col not in df.columns: df[col] = np.nan
    if 'mid' not in df.columns:
        df['mid'] = (df['bid'] + df['ask']) / 2 if df[['bid', 'ask']].notna().all(1).any() else df.get('lastPrice', df.get('last_price', np.nan))
    df = df[df['mid'] > 0].copy()
    if len(df) == 0: return pd.DataFrame()
    df['spread'] = df['ask'] - df['bid']
    thr = np.where(df['mid'] < 1.0, 0.25, 0.80 * df['mid'])
    df = df[(df['spread'] >= 0) & (df['spread'] <= thr)].copy()
    if len(df) == 0: return pd.DataFrame()
    df = df[df['mid'] > 0.05].copy()
    df = df[df['T'] > 0.03].copy()
    if len(df) == 0: return pd.DataFrame()
    S = float(pd.to_numeric(df['S'], errors='coerce').dropna().iloc[0])
    df['S'] = S
    forwards = {}
    for exp, g in df.groupby('expiry', sort=True):
        if g.empty: continue
        if float(g['T'].iloc[0]) <= 0: continue
        F = infer_forward(g, r, forward_mode)
        if np.isfinite(F) and F > 0: forwards[exp] = F
    if not forwards: return pd.DataFrame()
    df['F'] = df['expiry'].map(forwards)
    df = df[df['F'].notna()].copy()
    if len(df) == 0: return pd.DataFrame()
    df['k'] = np.log(df['strike'] / df['F'])
    df = df[df['k'].between(-0.7, 0.5)].copy()
    if otm_only:
        df = df[((df['type'] == 'call') & (df['k'] >= 0)) | ((df['type'] == 'put') & (df['k'] < 0))].copy()
    if len(df) == 0: return pd.DataFrame()
    df = df[df['spread'] < 0.5 * df['mid']].copy()
    if len(df) == 0: return pd.DataFrame()
    out = []
    for exp, g in df.groupby('expiry', sort=True):
        if g.empty: continue
        T = float(g['T'].iloc[0])
        if T <= 0: continue
        q = r - (1.0 / T) * np.log(max(forwards[exp], 1e-12) / S)    # dividend yield implied by the forward
        gg = g.copy()
        gg['iv_clean'] = implied_vol_vec((gg['type'] == 'call').to_numpy(), S, gg['strike'].to_numpy(float), r, q, T, gg['mid'].to_numpy(float))
        gg = gg[(gg['iv_clean'] > 0.0005) & (gg['iv_clean'] < 10.0)]
        if len(gg) > 0: out.append(gg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _svi_rows(clean: pd.DataFrame, asof_date) -> list:
    rows = []
    for expiry, g in clean.groupby('expiry'):
        if len(g) < MIN_OPTIONS_PER_EXPIRY: continue
        p = smiles.fit_svi(g)
        if p is not None:
            T = float(g['T'].iloc[0]); iv, skew, curv = smiles.atm_metrics(p, T)
            rows.append({'asof': asof_date, 'expiry': expiry, 'T': T, 'ATM_IV': iv, 'ATM_skew': skew, 'ATM_curvature': curv,
                         'a': p.a, 'b': p.b, 'rho': p.rho, 'm': p.m, 'sigma': p.sigma, 'n_options': len(g)})
    return rows


def _ssvi_rows(clean: pd.DataFrame, asof_date, rate: float, forward_mode: str) -> list:
    rows = []
    try:
        clean = clean.copy(); forwards, thetas = {}, {}
        for exp, g in clean.groupby('expiry'):
            if len(g) < MIN_OPTIONS_PER_EXPIRY: continue
            F = infer_forward(g, rate, forward_mode)
            if np.isfinite(F):
                forwards[exp] = F; thetas[exp] = smiles.theta_atm(g, F)
        if len(thetas) < 2: return rows
        thetas = smiles.monotone_theta(thetas, pd.Timestamp(asof_date))
        clean['F'] = clean['expiry'].map(forwards); clean['k_fwd'] = np.log(clean['strike'] / clean['F'])
        if 'mid_iv' not in clean.columns: clean['mid_iv'] = clean['iv_clean']
        rho, eta, gamma, cost = smiles.fit_ssvi_penalised(clean, thetas, forwards)
        for exp in thetas:
            rows.append({'asof': asof_date, 'expiry': exp, 'T': float(clean[clean['expiry'] == exp]['T'].iloc[0]), 'theta': thetas[exp],
                         'ssvi_rho': rho, 'ssvi_eta': eta, 'ssvi_p': gamma, 'fit_cost': cost})
        if rows:   # constrained fit on the same inputs, stored as fx_ columns
            fx, arr = smiles.fit_ssvi_constrained(clean, thetas, forwards)
            extra = {'gj': eta * (1 + abs(rho)), 'rmse_vol': smiles.rmse_vol(arr, rho, eta, gamma)}
            if fx is not None:
                extra.update({'fx_rho': fx['rho'], 'fx_eta': fx['eta'], 'fx_p': fx['gamma'], 'fx_fit_cost': fx['fit_cost'],
                              'fx_rmse_vol': fx['rmse_vol'], 'fx_gj': fx['gj']})
            for row in rows: row.update(extra)
    except Exception as e:
        logger.warning(f'SSVI fitting failed for {asof_date}: {e}')
    return rows


DENSITY_COLS = ('rnd_mean', 'rnd_vol', 'rnd_skew', 'rnd_kurtosis', 'martingale_error')


def _rnd_rows(ssvi_rows: list, S: float, rate: float, asof_date) -> list:
    rows = []
    try:
        ss = pd.DataFrame(ssvi_rows)
        for _, row in density.extract(ss, S, rate).iterrows():   # densities of the penalised fit decide which expiries are kept
            rows.append({'asof': asof_date, 'expiry': row['expiry'], 'T': row['T'], 'F': row['F'], 'rnd_mean': row['rnd_mean'],
                         'rnd_vol': row['rnd_vol'], 'rnd_skew': row['rnd_skew'], 'rnd_kurtosis': row['rnd_kurtosis'],
                         'martingale_error': row['martingale_error']})
        if 'fx_rho' in ss.columns and rows:   # same extraction on the constrained fit -> fx_ columns
            fx = density.extract(ss.assign(ssvi_rho=ss['fx_rho'], ssvi_eta=ss['fx_eta'], ssvi_p=ss['fx_p']), S, rate)
            fx = {r_['expiry']: r_ for _, r_ in fx.iterrows()} if len(fx) else {}
            for row in rows:
                f_ = fx.get(row['expiry'])
                for c in DENSITY_COLS: row['fx_' + c] = f_[c] if f_ is not None else np.nan
    except Exception as e:
        logger.warning(f'RND extraction failed: {e}')
    return rows


def fit_day(quotes: Quotes, d: date, forward_mode: str, otm_only: bool) -> dict:
    """All fits of one date. Returns dict(svi, ssvi, rnd, error)."""
    res = {'svi': [], 'ssvi': [], 'rnd': [], 'error': None}
    try:
        snap = quotes.snapshot(d)
        if len(snap) < 10:
            res['error'] = f'Insufficient data: {len(snap)} options'; return res
        sp = quotes.snapshot(d)
        S = float(sp['S'].iloc[0]) if ('S' in sp.columns and sp['S'].notna().any()) else None
        if S is None or not np.isfinite(S):
            res['error'] = 'No spot price'; return res
        rate = quotes.rate(d)
        clean = clean_iv_surface(snap, rate, d, forward_mode, otm_only)
        if len(clean) < 10:
            res['error'] = f'Insufficient clean data: {len(clean)} options'; return res
        res['svi'] = _svi_rows(clean, d)
        if len(res['svi']) >= 2:
            res['ssvi'] = _ssvi_rows(clean, d, rate, forward_mode)
            if res['ssvi']:
                res['rnd'] = _rnd_rows(res['ssvi'], S, rate, d)
    except Exception as e:
        res['error'] = str(e)
    return res


# ======================= units =======================
def fit_unit(q: Optional[pd.DataFrame], m: str) -> Optional[dict]:
    """Fits of every date of one unit -> dict of DataFrames (svi, ssvi, rnd, errors), or None if there are no quotes."""
    if q is None or len(q) == 0:
        return None
    quotes = Quotes(q)
    if not quotes.dates:
        return None
    spec = SPECS[m]; out = {'svi': [], 'ssvi': [], 'rnd': [], 'errors': []}
    for d in quotes.dates:
        r_ = fit_day(quotes, d, spec.forward, spec.otm_only)
        out['svi'].extend(r_['svi']); out['ssvi'].extend(r_['ssvi']); out['rnd'].extend(r_['rnd'])
        if r_['error']: out['errors'].append({'asof': d, 'error': r_['error']})
    out = {k: pd.DataFrame(v) for k, v in out.items()}
    out['n_dates'] = len(quotes.dates)
    return out


def unit_dir(layout: Layout, m: str, ticker: str, year: int, added: bool) -> str:
    """Folder that holds (or will hold) a unit: the first folder where it is finished, else the write folder."""
    for d_ in layout.surface_dirs_to_read(m, added):
        if os.path.exists(f'{d_}/{ticker}/{year}.done'): return d_
    return layout.surface_dir(m)


def run_unit(layout: Layout, m: str, ticker: str, year: int, added: bool = False):
    done = f'{unit_dir(layout, m, ticker, year, added)}/{ticker}/{year}.done'
    if os.path.exists(done): return ticker, year, 'done before', 0, 0, 0.0
    t0 = time.time(); SURF = layout.surface_dir(m); os.makedirs(f'{SURF}/{ticker}', exist_ok=True)
    logging.getLogger().setLevel(logging.ERROR); warnings.filterwarnings('ignore')
    res = fit_unit(load_quotes(layout, m, ticker, year, added), m)
    if res is None:
        open(done, 'w').write('no quotes'); return ticker, year, 'no quotes', 0, 0, 0.0
    for kind in ('svi', 'ssvi', 'rnd', 'errors'):
        res[kind].to_parquet(f'{SURF}/{ticker}/{kind}_{year}.parquet', index=False)
    n, ok = res['n_dates'], res['n_dates'] - len(res['errors'])
    open(done, 'w').write(f'{n} dates, {ok} ok, {time.time() - t0:.0f}s')
    return ticker, year, 'ok', n, ok, time.time() - t0


def run_units(layout: Layout, m: str, units, added=frozenset(), n_jobs: Optional[int] = None):
    """Fit every unit (ticker, year) not finished yet, in parallel processes. ``added`` = units that are ex-ante test additions."""
    n_jobs = n_jobs or os.cpu_count() or 1
    todo = [u for u in units if not os.path.exists(f'{unit_dir(layout, m, *u, u in added)}/{u[0]}/{u[1]}.done')]
    print(f'{m}: {len(units)} units, {len(units) - len(todo)} done before, {len(todo)} to fit on {n_jobs} process(es)')
    t0, k = time.time(), 0
    def show(r):
        nonlocal k; k += 1; el = time.time() - t0
        print(f'  [{k}/{len(todo)}] {r[0]} {r[1]}: {r[2]}, dates {r[3]}, fitted {r[4]}, {r[5]:.0f}s | elapsed {el / 60:.0f} min, '
              f'ETA {el / k * (len(todo) - k) / 60:.0f} min', flush=True)
    if n_jobs > 1 and len(todo) > 1:
        from joblib import Parallel, delayed
        for r in Parallel(n_jobs=n_jobs, return_as='generator_unordered')(delayed(run_unit)(layout, m, t, y, (t, y) in added) for t, y in todo):
            show(r)
    else:
        for t, y in todo: show(run_unit(layout, m, t, y, (t, y) in added))
