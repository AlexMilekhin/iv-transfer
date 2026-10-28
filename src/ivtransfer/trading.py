"""Economic value of the US model's forecasts (paper Section 8).

1. Forecasts of the final US model (seed average x output scale, ex-ante test names) and of gradient boosting, on every validation
   and test sample, mapped to rows of the US state.
2. Every candidate trade priced once: an at-the-money straddle on the expiry closest to the sample's maturity, entered at the grid
   date after the forecast and held h = 6 grid dates, delta-hedged daily at the Black-76 delta of the day's SVI smile; exits
   without a two-sided quote are marked from the surface.
3. v1: long the straddle when the forecast is a rise, short when a fall, only when |forecast| >= c x a round-trip cost proxy, c chosen
   on the fold's validation year (Sharpe); P&L per unit of vega, under three cost cases. P&L split into the fixed-strike vol change
   (vega part) and the rest (gamma / theta). v2: gamma-neutral calendars (long 44-day, short n x 29-day straddles).
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd

from .config import Layout
from .evaluate import Evaluation, gb_fold, masks

H_ = 6; CS = [0, 0.5, 1, 1.5, 2, 3]; HEDGE_BP = 1e-4
CASES = {'taker (f=1)': 1.0, 'half-spread (f=0.5)': 0.5, 'gross (f=0)': 0.0}


def ndtr(x):   # plain wrapper: scipy's ufunc does not unpickle in some joblib workers
    from scipy.special import ndtr as _ndtr
    return _ndtr(x)


# ---------- 1. forecasts ----------
def us_forecasts(ev: Evaluation, layout: Layout) -> pd.DataFrame:
    """ev: an Evaluation of the final run on market 'US' only, built with with_gb=True and forecasts already run."""
    assert ev.gate_ok.get('US'), 'US gate failed'
    mt = pd.read_parquet(f'{layout.state_dir("US")}/meta.parquet', columns=['ticker', 'asof', 'target_T']).reset_index(drop=True)
    keys = lambda tk, a, t: pd.DataFrame({'tk': np.asarray(tk).astype(str), 'a': pd.to_datetime(np.asarray(a)).astype('datetime64[ns]'), 'tT': np.round(np.asarray(t, float), 6)})
    K0 = keys(mt['ticker'], mt['asof'], mt['target_T']); K0['row'] = np.arange(len(K0)); assert not K0.duplicated(['tk', 'a', 'tT']).any()
    def rows_of(D_, idx):
        r_ = keys(D_['tk'][idx], D_['asof'][idx], D_['tT'][idx]).merge(K0, on=['tk', 'a', 'tT'], how='left')['row']
        assert r_.notna().all(), 'a scored sample has no row in the US state'; return r_.to_numpy().astype(int)
    parts = []
    for per, folds in ev.periods.items():
        for Y in folds:
            if ('US', per, Y) not in ev.FC: continue
            Fv, Ft = ev.combine('US', per, Y); rec = ev.FC[('US', per, Y)]; DO, DX = ev.DO['US'], ev.DX['US']
            TR, VA, TE = masks(DO, Y); gb = gb_fold(DO, DX, TR, VA, rec['it'])
            iv = Fv['A'].index.to_numpy()
            parts.append(pd.DataFrame({'fold': Y, 'role': 'val', 'i': rows_of(DO, iv), 'moe': Fv['A'].to_numpy(), 'gb': gb['gb_va'].reindex(iv).to_numpy()}))
            parts.append(pd.DataFrame({'fold': Y, 'role': 'test', 'i': rows_of(DX, rec['it']), 'moe': Ft['A'], 'gb': gb['gb_te']}))
    return pd.concat(parts, ignore_index=True)


# ---------- 2. pricing ----------
def _read_chain(chain_files, sym, cols=('date', 'expiration', 'strike', 'call_put', 'bid', 'ask')):
    import pyarrow.parquet as pq
    return pd.concat([pq.read_table(c_, filters=[('symbol', '==', sym)], columns=list(cols)).to_pandas() for c_ in chain_files], ignore_index=True).drop_duplicates()


def b76_w(F, K, w):
    sw = np.sqrt(np.maximum(w, 1e-12)); d1 = (np.log(F / K) + w / 2) / sw; d2 = d1 - sw
    return F * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - F * ndtr(-d1), d1


def svi_w(a, b, rho, m, s, k): return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def price_symbol(sym, smp, chain_files, CAL, RATE, svi_dir):
    """Price the straddle of every sample of one underlying (smp: rows i, pos, target_T)."""
    q = _read_chain(chain_files, sym)
    q['date'] = pd.to_datetime(q['date']).astype('datetime64[ns]'); q['expiration'] = pd.to_datetime(q['expiration']).astype('datetime64[ns]')
    q = q[q['date'].isin(set(CAL[np.unique(np.concatenate([smp['pos'].to_numpy() + j for j in range(H_ + 1)]))]))]
    q = q[(q['bid'] > 0) & (q['ask'] >= q['bid'])].copy(); q['cp'] = q['call_put'].astype(str).str.upper().str[0]
    if q.empty: return pd.DataFrame()
    P = q.pivot_table(index=['date', 'expiration', 'strike'], columns='cp', values=['bid', 'ask'], aggfunc='mean')
    P.columns = [f'{a}{b}' for a, b in P.columns]; P = P.reset_index()
    for c in ('bidC', 'bidP', 'askC', 'askP'):
        if c not in P: P[c] = np.nan
    P['mC'], P['mP'] = (P['bidC'] + P['askC']) / 2, (P['bidP'] + P['askP']) / 2
    P['T'] = (P['expiration'] - P['date']).dt.days / 365; P['r'] = RATE.reindex(P['date']).to_numpy()
    B_ = P.dropna(subset=['mC', 'mP']).copy(); B_['ad'] = (B_['mC'] - B_['mP']).abs()
    B_['rk'] = B_.groupby(['date', 'expiration'])['ad'].rank(method='first')
    f3 = B_[B_['rk'] <= 3].assign(Fk=lambda d: d['strike'] + np.exp(d['r'] * d['T']) * (d['mC'] - d['mP']))
    FW = f3.groupby(['date', 'expiration'])['Fk'].median().rename('F').reset_index()
    ATM = B_[B_['rk'] == 1][['date', 'expiration', 'strike', 'T']]
    FW = FW.merge(P[['date', 'expiration', 'T', 'r']].drop_duplicates(['date', 'expiration']), on=['date', 'expiration'])
    nr = FW[FW['T'] >= 7 / 365].sort_values('T').drop_duplicates('date')
    SP = (nr['F'] * np.exp(-nr['r'] * nr['T'])).set_axis(nr['date']).rename('S')
    sp_ = f'{svi_dir}/historical_svi_{sym}_all.parquet'
    SV = pd.read_parquet(sp_, columns=['asof', 'expiry', 'T', 'a', 'b', 'rho', 'm', 'sigma']) if os.path.exists(sp_) else pd.DataFrame(columns=['asof', 'expiry', 'T', 'a', 'b', 'rho', 'm', 'sigma'])
    SV['asof'] = pd.to_datetime(SV['asof']).astype('datetime64[ns]'); SV['expiry'] = pd.to_datetime(SV['expiry']).astype('datetime64[ns]'); SV = SV.rename(columns={'asof': 'date', 'T': 'Ts'}).drop_duplicates(['date', 'expiry'])
    tr = smp.assign(e=CAL[smp['pos'].to_numpy()])
    cand = tr.merge(ATM.rename(columns={'date': 'e'}), on='e')
    cand = cand[cand['T'] >= 7 / 365].assign(dT=lambda d: (d['T'] - d['target_T']).abs())
    tr = cand.sort_values('dT').drop_duplicates('i').rename(columns={'expiration': 'E', 'strike': 'K', 'T': 'T0'})[['i', 'pos', 'target_T', 'E', 'K', 'T0']]
    if tr.empty: return pd.DataFrame()
    pth = tr.loc[tr.index.repeat(H_ + 1)].assign(step=lambda d: d.groupby('i').cumcount())
    pth['date'] = CAL[pth['pos'].to_numpy() + pth['step'].to_numpy()]; pth['T'] = (pth['E'] - pth['date']).dt.days / 365
    pth['r'] = RATE.reindex(pth['date']).to_numpy(); pth['S'] = SP.reindex(pth['date']).to_numpy()
    pth = pth.merge(P[['date', 'expiration', 'strike', 'bidC', 'askC', 'bidP', 'askP', 'mC', 'mP']].rename(columns={'expiration': 'E', 'strike': 'K'}), on=['date', 'E', 'K'], how='left')
    pth = pth.merge(FW[['date', 'expiration', 'F']].rename(columns={'expiration': 'E'}), on=['date', 'E'], how='left')
    pth['F'] = pth['F'].fillna(pth['S'] * np.exp(pth['r'] * pth['T'])); pth['k'] = np.log(pth['K'] / pth['F'])
    own = pth.merge(SV.rename(columns={'expiry': 'E'}), on=['date', 'E'], how='left')
    w_own = svi_w(own['a'], own['b'], own['rho'], own['m'], own['sigma'], own['k']) * own['T'] / own['Ts']
    q_ = pth[['date', 'T', 'k']].reset_index().sort_values('T'); sv_ = SV.sort_values('Ts')
    lo = pd.merge_asof(q_, sv_, left_on='T', right_on='Ts', by='date', direction='backward').set_index('index').sort_index()
    hi = pd.merge_asof(q_, sv_, left_on='T', right_on='Ts', by='date', direction='forward').set_index('index').sort_index()
    wl = svi_w(lo['a'], lo['b'], lo['rho'], lo['m'], lo['sigma'], lo['k']); wh = svi_w(hi['a'], hi['b'], hi['rho'], hi['m'], hi['sigma'], hi['k'])
    w_int = np.where(lo['Ts'].notna() & hi['Ts'].notna(), wl + (pth['T'] - lo['Ts']) / (hi['Ts'] - lo['Ts']) * (wh - wl),
                     np.where(lo['Ts'].notna(), wl / lo['Ts'] * pth['T'], wh / hi['Ts'] * pth['T']))
    pth['w'] = np.where(np.isfinite(w_own.to_numpy()), w_own.to_numpy(), w_int)
    pth['w_raw'] = pth['w']; pth['wATM'] = (svi_w(own['a'], own['b'], own['rho'], own['m'], own['sigma'], 0.0) * own['T'] / own['Ts']).to_numpy()
    pth['w'] = pth.groupby('i')['w'].ffill()
    c_, p_, d1 = b76_w(pth['F'].to_numpy(), pth['K'].to_numpy(), pth['w'].to_numpy()); dfc = np.exp(-pth['r'] * pth['T'])
    pth['markC'], pth['markP'] = c_ * dfc, p_ * dfc; pth['delta'] = 2 * ndtr(d1) - 1
    pth['vega'] = 2 * dfc * pth['F'] * np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi) * np.sqrt(np.maximum(pth['T'], 1e-9)) * 0.01
    pth['dS'] = pth.groupby('i')['S'].shift(-1) - pth['S']
    pth['dD'] = (pth['delta'] - pth.groupby('i')['delta'].shift(1).fillna(0)).abs()
    e0, eH = pth[pth['step'] == 0].set_index('i'), pth[pth['step'] == H_].set_index('i')
    hp = -pth[pth['step'] < H_].assign(t_=lambda d: d['delta'] * d['dS']).groupby('i')['t_'].sum(min_count=H_)
    hc = HEDGE_BP * (pth[pth['step'] < H_].assign(n=lambda d: d['dD'] * d['S']).groupby('i')['n'].sum() + (eH['delta'] * eH['S']).abs())
    out = pd.DataFrame({'i': e0.index, 'ticker': sym, 'entry': e0['date'], 'exit': eH['date'].reindex(e0.index), 'E': e0['E'], 'K': e0['K'], 'T0': e0['T0'],
                        'V0': e0['mC'] + e0['mP'], 'h0': ((e0['askC'] - e0['bidC']) + (e0['askP'] - e0['bidP'])) / 2, 'vega0': e0['vega'], 'S0': e0['S']}).set_index('i')
    xC = eH['mC'].where(eH['mC'].notna(), eH['markC']); xP = eH['mP'].where(eH['mP'].notna(), eH['markP'])
    out['VH'] = (xC + xP).reindex(out.index); out['n_marked'] = (eH['mC'].isna().astype(int) + eH['mP'].isna().astype(int)).reindex(out.index)
    hq = ((eH['askC'] - eH['bidC']) + (eH['askP'] - eH['bidP'])) / 2
    out['hH'] = hq.where(hq.notna(), out['h0'] / out['V0'] * out['VH']).reindex(out.index)
    out['hedge_pnl'] = hp.reindex(out.index); out['hedge_cost'] = hc.reindex(out.index)
    out['w0'], out['wH'], out['TH'] = e0['w_raw'], eH['w_raw'].reindex(out.index), eH['T'].reindex(out.index)
    out['wATM0'], out['wATMH'] = e0['wATM'], eH['wATM'].reindex(out.index)
    lr = np.log(pth.groupby('i')['S'].shift(-1) / pth['S']); out['rv'] = (lr ** 2).groupby(pth['i']).sum(min_count=H_).reindex(out.index)
    return out.reset_index()


def _price_and_save(sym, smp, chain_files, CAL, RATE, svi_dir, trades_dir):
    price_symbol(sym, smp, chain_files, CAL, RATE, svi_dir).to_parquet(f'{trades_dir}/{sym}.parquet', index=False); return sym


def candidates(layout: Layout, FC_T: pd.DataFrame, end_last: pd.Timestamp):
    """Samples to trade: rows of the US state with a forecast and a defined roll; entry at the next grid date, exit h dates later."""
    A_ = np.load(f'{layout.state_dir("US")}/arrays.npz', allow_pickle=True); meta_u = pd.read_parquet(f'{layout.state_dir("US")}/meta.parquet').reset_index(drop=True)
    CAL = pd.DatetimeIndex(pd.to_datetime(A_['CAL_DATES'])).astype('datetime64[ns]').sort_values(); asof_u = pd.to_datetime(meta_u['asof']).astype('datetime64[ns]')
    S_all = meta_u[['ticker', 'target_T', 'total_var_t']].assign(i=np.arange(len(meta_u)), asof=asof_u, mech=meta_u['mech'].to_numpy())
    S_all = S_all[(S_all['mech'] != 0) & S_all['i'].isin(set(FC_T['i']))]
    S_all['pos'] = CAL.searchsorted(S_all['asof'].to_numpy(), side='right')
    S_all = S_all[S_all['pos'] + H_ < len(CAL)]; S_all = S_all[CAL[S_all['pos'].to_numpy() + H_] <= end_last]
    return S_all, CAL


def price_all(layout: Layout, S_all, CAL, chain_files, trades_dir, n_jobs=None, log=print):
    RATES = pd.read_parquet(layout.rates_file('US')).set_index('date')['dtb3'].sort_index().ffill(); RATES = np.log1p(RATES / 100)
    RATES.index = pd.to_datetime(RATES.index).astype('datetime64[ns]'); RATE = RATES.reindex(CAL, method='ffill')
    os.makedirs(trades_dir, exist_ok=True); svi_dir = layout.collected_dir('US')
    todo = [t for t in sorted(S_all['ticker'].unique()) if not os.path.exists(f'{trades_dir}/{t}.parquet')]
    log(f'  straddles to price: {len(S_all):,} samples on {S_all.ticker.nunique()} names; tickers left {len(todo)}')
    from joblib import Parallel, delayed
    n_jobs = n_jobs or min(16, os.cpu_count() or 1); t0 = time.time()
    for k0 in range(0, len(todo), 40):
        jobs = [(t, S_all[S_all['ticker'] == t]) for t in todo[k0:k0 + 40]]
        Parallel(n_jobs=n_jobs)(delayed(_price_and_save)(t, smp, chain_files, CAL, RATE, svi_dir, trades_dir) for t, smp in jobs)
        log(f'  priced {min(k0 + 40, len(todo))}/{len(todo)} tickers | {time.time() - t0:.0f}s')
    return pd.concat([pd.read_parquet(f_) for f_ in glob.glob(f'{trades_dir}/*.parquet') if os.path.getsize(f_) > 0], ignore_index=True)


# ---------- 3. evaluation ----------
def evaluate(FC_T, T2RAW, S_all, CAL, periods, ends, out, note):
    GRID = pd.Series(np.arange(len(CAL)), index=CAL)
    def pnl(d, sgn, f): return (sgn * d['gross'] - f * (d['h0'] + d['hH']) - d['hedge_cost']) / d['vega0']
    def series(d, p):
        s = p.groupby(d['entry']).mean(); g = GRID[(GRID.index >= s.index.min()) & (GRID.index <= s.index.max())]
        return s.reindex(g.index).fillna(0.0)
    def sharpe(s):   # h overlapping portfolios, annualised with 26 two-week periods
        v = [s.iloc[o::H_] for o in range(H_)]; return float(np.mean([x.mean() / x.std() * np.sqrt(26) if x.std() > 0 else 0 for x in v]))
    def nw_t(x, lag):
        x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
        if n < 3 * lag: return np.nan
        e = x - x.mean(); v = e @ e / n
        for l in range(1, lag + 1): v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
        return x.mean() / np.sqrt(v / n)
    txt = []
    for per, folds in periods.items():
        TR = T2RAW.merge(S_all[['i', 'total_var_t']], on='i'); TR['gross'] = TR['VH'] - TR['V0'] + TR['hedge_pnl']
        TR = TR[pd.to_datetime(TR['exit']) <= ends[per]]
        ok_ = np.isfinite(TR[['V0', 'VH', 'h0', 'hH', 'vega0', 'hedge_pnl', 'hedge_cost']]).all(1) & (TR['V0'] > 0) & (TR['vega0'] > 0)
        txt.append(f'{per}: priced {len(TR):,} straddles, usable {ok_.mean():.1%}, exits marked from the surface {(TR.n_marked > 0).mean():.1%}'); TR = TR[ok_]
        D = FC_T[FC_T['fold'].isin(folds)].merge(TR, on='i'); D['proxy'] = 2 * D['total_var_t'] * 2 * D['h0'] / D['V0']
        D['year'] = pd.to_datetime(D['entry']).dt.year; D['entry'] = pd.to_datetime(D['entry'])
        def pick_c(d, col, f):
            best = (-np.inf, 0)
            for c in CS:
                s_ = d[np.abs(d[col]) >= c * d['proxy']]
                if len(s_) < 100: continue
                sh = sharpe(series(s_, pnl(s_, np.sign(s_[col]), f)))
                if sh > best[0] + 1e-9: best = (sh, c)
            return best[1]
        rep = {}
        for cl, f in CASES.items():
            for strat in ('MoE', 'GB', 'MoE c=0', 'always short', 'always long'):
                parts = []
                for Y in folds:
                    te = D[(D['fold'] == Y) & (D['role'] == 'test')]
                    if strat in ('MoE', 'GB'):
                        col = 'moe' if strat == 'MoE' else 'gb'; c = pick_c(D[(D['fold'] == Y) & (D['role'] == 'val')], col, f)
                        t_ = te[np.abs(te[col]) >= c * te['proxy']]; sg = np.sign(t_[col])
                    elif strat == 'MoE c=0': t_ = te; sg = np.sign(te['moe'])
                    else: t_ = te; sg = pd.Series(-1.0 if strat == 'always short' else 1.0, index=te.index)
                    parts.append(t_.assign(p=pnl(t_, sg, f), sg=sg))
                A = pd.concat(parts); s = series(A, A['p'])
                rep[(cl, strat)] = {'trades': len(A), 'P&L/trade (vol pts)': round(A['p'].mean(), 3), 'hit': round(float((A['p'] > 0).mean()), 3), 'Sharpe': round(sharpe(s), 2),
                                    'NW t': round(nw_t(s.values, H_), 1), 'long / short P&L': ' / '.join(f'{x:+.3f}' if x == x else 'none' for x in (A.loc[A.sg > 0, 'p'].mean(), A.loc[A.sg < 0, 'p'].mean()))}
                note('8', f'v1 straddles, {strat}, {cl}: P&L per trade (vol points of vega)', 'US', per, round(A['p'].mean(), 2), f'{len(A):,} trades, Sharpe {sharpe(s):.2f}, long / short {rep[(cl, strat)]["long / short P&L"]}')
        pd.DataFrame(rep).T.to_csv(f'{out}/trading_v1_{per}.csv')
        T_ = D[(D['role'] == 'test')].copy(); T_ = T_[np.isfinite(T_[['w0', 'wH', 'T0', 'TH']]).all(1) & (T_['w0'] > 0) & (T_['wH'] > 0) & (T_['TH'] > 0)]
        T_['pv'] = T_['gross'] / T_['vega0']; T_['dK'] = (np.sqrt(T_['wH'] / T_['TH']) - np.sqrt(T_['w0'] / T_['T0'])) * 100; T_['gt'] = T_['pv'] - T_['dK']
        up = T_[T_['moe'] > 0]
        note('8', 'vega part share of the variance of straddle P&L', 'US', per, f'{100 * T_["dK"].var() / T_["pv"].var():.0f}%', 'test trades, gross, per vega')
        note('8', 'trades with a forecast rise: vega part / gamma-theta part (vol points per vega)', 'US', per, f'{up["dK"].mean():+.2f} / {up["gt"].mean():+.2f}', f'{len(up):,} trades')
        T2 = T2RAW.copy(); T2 = T2[pd.to_datetime(T2['exit']) <= ends[per]]; T2['gross'] = T2['VH'] - T2['V0'] + T2['hedge_pnl']
        ok2 = np.isfinite(T2[['V0', 'VH', 'vega0', 'hedge_pnl', 'w0', 'wH', 'T0', 'TH']]).all(1) & (T2['V0'] > 0) & (T2['vega0'] > 0) & (T2['w0'] > 0) & (T2['wH'] > 0) & (T2['TH'] > 0)
        T2 = T2[ok2].copy(); T2['dK'] = (np.sqrt(T2['wH'] / T2['TH']) - np.sqrt(T2['w0'] / T2['T0'])) * 100
        A2 = T2.merge(S_all[['i', 'asof', 'target_T', 'total_var_t']], on='i')
        cols = ['i', 'ticker', 'asof', 'entry', 'exit', 'E', 'T0', 'V0', 'h0', 'hH', 'vega0', 'S0', 'w0', 'gross', 'hedge_cost', 'dK', 'target_T', 'total_var_t']
        P2 = A2.loc[np.isclose(A2['target_T'], 0.12), cols].merge(A2.loc[np.isclose(A2['target_T'], 0.08), cols], on=['ticker', 'asof'], suffixes=('_l', '_s'))
        P2 = P2[P2['E_l'] > P2['E_s']].copy()
        for s_ in ('l', 's'):
            P2[f'sig_{s_}'] = np.sqrt(P2[f'w0_{s_}'] / P2[f'T0_{s_}']); P2[f'gam_{s_}'] = P2[f'vega0_{s_}'] / (P2[f'S0_{s_}'] ** 2 * P2[f'sig_{s_}'] * P2[f'T0_{s_}'])
        P2['n'] = P2['gam_l'] / P2['gam_s']; P2['nv'] = P2['vega0_l'] - P2['n'] * P2['vega0_s']
        P2 = P2[np.isfinite(P2[['n', 'nv']]).all(1) & (P2['nv'] > 0)]
        P2['g'] = P2['gross_l'] - P2['n'] * P2['gross_s']; P2['spr'] = P2['h0_l'] + P2['hH_l'] + P2['n'] * (P2['h0_s'] + P2['hH_s'])
        P2['hc'] = P2['hedge_cost_l'] + P2['n'] * P2['hedge_cost_s']; P2['vpart'] = (P2['vega0_l'] * P2['dK_l'] - P2['n'] * P2['vega0_s'] * P2['dK_s']) / P2['nv']
        P2['proxy'] = 2 * (P2['h0_l'] + P2['n'] * P2['h0_s']); P2['entry'] = pd.to_datetime(P2['entry_l'])
        Fk = FC_T[FC_T['fold'].isin(folds)]
        Q = (P2.merge(Fk.rename(columns={'i': 'i_l', 'moe': 'moe_l', 'gb': 'gb_l'}), on='i_l').merge(Fk.rename(columns={'i': 'i_s', 'moe': 'moe_s', 'gb': 'gb_s'}), on=['i_s', 'fold', 'role']))
        for m_ in ('moe', 'gb'):
            dl = Q[f'{m_}_l'] / (2 * np.sqrt(Q['total_var_t_l'] / Q['target_T_l']) * Q['target_T_l']) * 100
            ds = Q[f'{m_}_s'] / (2 * np.sqrt(Q['total_var_t_s'] / Q['target_T_s']) * Q['target_T_s']) * 100
            Q[f'ep_{m_}'] = Q['vega0_l'] * dl - Q['n'] * Q['vega0_s'] * ds
        te_ = Q[Q['role'] == 'test']
        if len(te_) > 2:
            note('8', "calendars: correlation of the forecast's expected P&L with the calendar's vega part", 'US', per, round(float(te_['ep_moe'].corr(te_['vpart'])), 2), f'{len(te_):,} test pairs')
            for cl, f in CASES.items():
                p_ = (np.sign(te_['ep_moe']) * te_['g'] - f * te_['spr'] - te_['hc']) / te_['nv']
                note('8', f'v2 calendars, MoE c=0, {cl}: P&L per trade (vol points of net vega)', 'US', per, round(float(p_.mean()), 2), f'{len(te_):,} trades')
    return '\n'.join(txt)
