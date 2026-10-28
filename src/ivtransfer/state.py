"""From daily smile fits to the model-ready samples of one market (paper Sections 3.4-3.5 and 4.1).

1. ``collect``: per underlying, the fits of all its units in one file (constrained SSVI fit under the standard column names) and the
   daily regime inputs (enrich.py).
2. ``build``: sample weights from fit quality, constant-maturity interpolation at the market's maturities on the Mon / Wed / Fri grid,
   the input windows, the target (change in total variance over h = 6 grid dates net of the ex-ante roll) and the metadata.
   Saved as arrays.npz (X, X_gate, y, w, MECH_ALL, CAL_DATES), meta.parquet, CM.parquet and CM30.parquet.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import enrich as _enrich
from .config import (ALL17, COVID_WINDOW, GATING, GRID_WEEKDAYS, HORIZON, MAX_FILLED, SEQ_LEN, SPAN_H_MAX, SPAN_IN_MAX, SPECS, Layout)
from .surfaces import unit_dir

FX_COLS = {'ssvi': {'ssvi_rho': 'fx_rho', 'ssvi_eta': 'fx_eta', 'ssvi_p': 'fx_p', 'fit_cost': 'fx_fit_cost'},
           'rnd': {c: 'fx_' + c for c in ('rnd_mean', 'rnd_vol', 'rnd_skew', 'rnd_kurtosis', 'martingale_error')}}


# ======================= 1. collect =======================
def collect_ticker(layout: Layout, m: str, t: str, units, added=frozenset()):
    """Concatenate one underlying's unit files and enrich them. Returns a note string if the underlying is skipped."""
    years = [y for tt, y in sorted(units) if tt == t]
    dirs = {y: unit_dir(layout, m, t, y, (t, y) in added) for y in years}
    if not all(os.path.exists(f'{dirs[y]}/{t}/{y}.done') for y in years):
        return f'{t}: surfaces not finished - skipped'
    out = layout.collected_dir(m); os.makedirs(out, exist_ok=True)
    for kind in ('svi', 'ssvi', 'rnd'):
        fs = [f'{dirs[y]}/{t}/{kind}_{y}.parquet' for y in years]
        parts = [p for p in (pd.read_parquet(f) for f in fs if os.path.exists(f)) if len(p)]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if kind in FX_COLS and len(df) and any(c.startswith('fx_') for c in df.columns):
            for std_c, fx_c in FX_COLS[kind].items():   # constrained fit under the standard names; the penalised fit kept as repo_ columns
                df['repo_' + std_c] = df[std_c]; df[std_c] = df[fx_c]
        df.to_parquet(f'{out}/historical_{kind}_{t}_all.parquet', index=False)
    svi = pd.read_parquet(f'{out}/historical_svi_{t}_all.parquet')
    if len(svi) == 0:
        return f'{t}: no SVI fits - skipped'
    svi['asof'] = pd.to_datetime(svi['asof']); svi['expiry'] = pd.to_datetime(svi['expiry'])
    _enrich.enrich(svi).to_parquet(f'{out}/historical_svi_{t}_all_enriched.parquet', index=False)
    return None


def collect(layout: Layout, m: str, units, added=frozenset(), n_jobs: int = 8):
    tickers = sorted(set(t for t, _ in units))
    if n_jobs > 1 and len(tickers) > 1:
        from joblib import Parallel, delayed
        notes = Parallel(n_jobs=n_jobs)(delayed(collect_ticker)(layout, m, t, units, added) for t in tickers)
    else:
        notes = [collect_ticker(layout, m, t, units, added) for t in tickers]
    for n_ in notes:
        if n_: print(n_)


# ======================= 2. build =======================
def physics_weights(df: pd.DataFrame, calendar_penalty: float = 0.05) -> pd.DataFrame:
    """Surface-quality weight per (date, expiry) of one underlying, from past information only:
    fit cost scaled by the running min / max of the underlying's fit costs up to that day (missing: running median),
    martingale error (capped at 10%; missing: 1%), and a 5% penalty when the SVI level a falls from the previous date."""
    df = df.sort_values(['expiry', 'asof']).copy()
    df['theta_violation'] = (df.groupby('expiry')['a'].diff() < 0).fillna(False).astype(float)
    df['martingale_error'] = df['martingale_error'].fillna(0.01)
    d_ = pd.to_datetime(df['asof']); o_ = np.argsort(d_.to_numpy(), kind='stable'); fc_ = df['fit_cost'].iloc[o_]; dd_ = d_.iloc[o_]
    med_ = fc_.expanding().median().groupby(dd_.to_numpy()).last()
    lo_ = fc_.groupby(dd_.to_numpy()).min().cummin().ffill(); hi_ = fc_.groupby(dd_.to_numpy()).max().cummax().ffill()
    day_ = d_.to_numpy()
    fcf_ = df['fit_cost'].fillna(pd.Series(med_.reindex(day_).to_numpy(), index=df.index))
    lo_r, hi_r = lo_.reindex(day_).to_numpy(), hi_.reindex(day_).to_numpy()
    fit_cost_norm = (1 - (fcf_ - lo_r) / (hi_r - lo_r + 1e-8)).clip(0, 1).fillna(1.0)
    martingale_norm = 1 - df['martingale_error'].clip(0, 0.1) / 0.1
    df['physics_weight'] = fit_cost_norm * martingale_norm * (1.0 - calendar_penalty * df['theta_violation'])
    return df


def load_collected(layout: Layout, m: str, tickers) -> pd.DataFrame:
    """Enriched SVI rows of every underlying with all three collected files, with fit cost, martingale error and weights."""
    d = layout.collected_dir(m); parts = []
    for t in sorted(tickers):
        fs = [f'{d}/historical_svi_{t}_all_enriched.parquet', f'{d}/historical_ssvi_{t}_all.parquet', f'{d}/historical_rnd_{t}_all.parquet']
        if not all(os.path.exists(f) for f in fs): continue
        try:
            df, ssvi, rnd = (pd.read_parquet(f) for f in fs)
            for x in (df, ssvi, rnd):
                x['asof'] = pd.to_datetime(x['asof']); x['expiry'] = pd.to_datetime(x['expiry'])
            df = df.merge(ssvi[['asof', 'expiry', 'fit_cost']], on=['asof', 'expiry'], how='left')
            df = df.merge(rnd[['asof', 'expiry', 'martingale_error']], on=['asof', 'expiry'], how='left')
            df['ticker'] = t
            parts.append(physics_weights(df))
        except Exception as e:   # e.g. an underlying with no SSVI day at all
            print(f'  {t}: not loaded ({type(e).__name__}: {str(e)[:60]})')
    if not parts: raise ValueError(f'{m}: no underlying loaded')
    return pd.concat(parts, ignore_index=True)


def constant_maturity(df: pd.DataFrame, buckets) -> tuple:
    """Per underlying and grid date, values at each constant maturity tm (paper eq. 4): total variance linear in T between the
    expiries either side (7-150 days); when no expiry of 7+ days lies below tm, vol is held flat from the nearest expiry above if it is
    within 25 days of tm. Skew, curvature and weight are interpolated the same way. Returns (CM, CAL_DATES, CAL_IDX)."""
    src_ = df[pd.to_datetime(df['asof']).dt.weekday.isin(list(GRID_WEEKDAYS))]
    CAL_DATES = np.array(sorted(pd.to_datetime(src_['asof']).unique())); CAL_IDX = {d_: i for i, d_ in enumerate(CAL_DATES)}
    base = src_[['ticker', 'asof', 'T', 'ATM_IV', 'ATM_skew', 'ATM_curvature', 'physics_weight']].copy(); base['asof'] = pd.to_datetime(base['asof'])
    base = base[np.isfinite(base['T']) & np.isfinite(base['ATM_IV']) & (base['ATM_IV'] > 0) & (base['T'] >= 7 / 365) & (base['T'] <= 150 / 365)]
    base['w_'] = base['ATM_IV'] ** 2 * base['T']; base = base.sort_values('T')
    R_ = base[['ticker', 'asof', 'T', 'w_', 'ATM_skew', 'ATM_curvature', 'physics_weight']].assign(T_x=lambda d_: d_['T'])
    DAY = src_.assign(asof=pd.to_datetime(src_['asof'])).groupby(['ticker', 'asof'])[['IV_percentile', 'ATM_IV_zscore', 'skew_percentile', 'ATM_skew_zscore', 'term_slope']].first()
    parts = []
    for tm in buckets:
        tg = base[['ticker', 'asof']].drop_duplicates().assign(T=tm)
        lo = pd.merge_asof(tg, R_, on='T', by=['ticker', 'asof'], direction='backward')
        hi = pd.merge_asof(tg, R_, on='T', by=['ticker', 'asof'], direction='forward', allow_exact_matches=False)
        mm = lo.merge(hi, on=['ticker', 'asof', 'T'], suffixes=('_l', '_h'))
        flat = mm['T_x_l'].isna() & mm['T_x_h'].notna() & (mm['T_x_h'] - tm <= 25 / 365)          # short end: vol flat from the expiry above
        for c in ('w_', 'ATM_skew', 'ATM_curvature', 'physics_weight'):
            mm.loc[flat, f'{c}_l'] = mm.loc[flat, f'{c}_h']
        mm.loc[flat, 'w__l'] = mm.loc[flat, 'w__h'] / mm.loc[flat, 'T_x_h'] * tm; mm.loc[flat, 'w__h'] = mm.loc[flat, 'w__l']
        mm.loc[flat, 'T_x_l'] = tm - 1e-9
        mm = mm.dropna(subset=['T_x_l', 'T_x_h']).assign(flat=lambda d_: flat.reindex(d_.index).fillna(False).astype(bool))
        a = (tm - mm['T_x_l']) / (mm['T_x_h'] - mm['T_x_l']); wv = mm['w__l'] + a * (mm['w__h'] - mm['w__l'])
        parts.append(pd.DataFrame({'ticker': mm['ticker'], 'asof': mm['asof'], 'target_T': tm, 'T': tm, 'total_var': wv,
                                   'ATM_IV': np.sqrt(np.maximum(wv, 1e-12) / tm),
                                   'ATM_skew': mm['ATM_skew_l'] + a * (mm['ATM_skew_h'] - mm['ATM_skew_l']),
                                   'ATM_curvature': mm['ATM_curvature_l'] + a * (mm['ATM_curvature_h'] - mm['ATM_curvature_l']),
                                   'pw': mm['physics_weight_l'] + a * (mm['physics_weight_h'] - mm['physics_weight_l']), 'flat': mm['flat']})[wv > 0])
    CM = pd.concat(parts, ignore_index=True).join(DAY, on=['ticker', 'asof'])
    CM['di'] = CM['asof'].map(CAL_IDX).astype(int)
    CM = CM.sort_values(['ticker', 'target_T', 'di']).reset_index(drop=True)
    return CM, CAL_DATES, CAL_IDX


def samples(CM: pd.DataFrame, features=ALL17, gating=GATING, L=SEQ_LEN, H=HORIZON):
    """Grid-step inputs (lags, momentum), weights, input windows and full-change targets (paper eq. 5 and Table 3).
    Sample at grid date t: inputs t-9..t, target w(t+h) - w(t) at the same maturity; t and t+h must exist; an isolated missing input
    date is filled with the previous grid date (at most MAX_FILLED per window); the window spans at most SPAN_IN_MAX calendar days
    and the horizon at most SPAN_H_MAX."""
    KEY = pd.MultiIndex.from_arrays([CM['ticker'], CM['target_T'], CM['di']])
    def at_lag(vals, k):   # value k grid dates earlier (k > 0), same name and maturity; NaN if that date is missing
        return pd.Series(vals, index=KEY).reindex(pd.MultiIndex.from_arrays([CM['ticker'], CM['target_T'], CM['di'] - k])).to_numpy()
    tv = CM['total_var'].to_numpy()
    for k in (1, 2, 3, 5, 7): CM[f'total_var_change_lag{k}'] = tv - at_lag(tv, k)
    c5 = CM['ATM_IV'].to_numpy() - at_lag(CM['ATM_IV'].to_numpy(), 5)
    stack5 = np.column_stack([c5] + [at_lag(c5, j) for j in range(1, 5)])
    cnt5 = np.isfinite(stack5).sum(1)
    CM['IV_momentum'] = np.where(cnt5 >= 2, np.nanmean(np.where(np.isfinite(stack5), stack5, np.nan), axis=1), np.nan)
    CM['IV_acceleration'] = c5 - at_lag(c5, 1)
    # a missing weight: running median of the same name and maturity's earlier weights (none yet: 1)
    CM['pw'] = CM['pw'].fillna(CM.groupby(['ticker', 'target_T'])['pw'].transform(lambda s_: s_.expanding().median().shift(1))).fillna(1.0)
    CM['sample_weight'] = CM['pw'] / (CM['T'] + 0.1)
    FC = list(features)
    missing = set(FC) - set(CM.columns)
    if missing: raise ValueError(f'features not built: {missing}')
    FEAT = CM[FC].to_numpy(np.float32); FEAT[~np.isfinite(FEAT)] = 0.0
    IDX0 = np.column_stack([KEY.get_indexer(pd.MultiIndex.from_arrays([CM['ticker'], CM['target_T'], CM['di'] + j])) for j in range(-L, 1)])
    IDX = IDX0[:, 1:].copy(); prev = IDX0[:, :-1]
    fill = (IDX < 0) & (prev >= 0); fill[:, -1] = False
    IDX = np.where(fill, prev, IDX); n_filled = fill.sum(1)
    FUT = KEY.get_indexer(pd.MultiIndex.from_arrays([CM['ticker'], CM['target_T'], CM['di'] + H]))
    dates = CM['asof'].to_numpy(); sel = np.where((IDX >= 0).all(1) & (FUT >= 0) & (n_filled <= MAX_FILLED))[0]
    span_in = (dates[sel] - dates[IDX[sel, 0]]) / np.timedelta64(1, 'D'); span_h = (dates[FUT[sel]] - dates[sel]) / np.timedelta64(1, 'D')
    sel = sel[(span_in <= SPAN_IN_MAX) & (span_h <= SPAN_H_MAX)]
    assert len(sel), 'no samples left after the window filters'
    stats = {'windows with filled dates %': round(100 * float(np.mean(n_filled[sel] > 0)), 1), 'one date filled %': round(100 * float(np.mean(n_filled[sel] == 1)), 1),
             'two dates filled %': round(100 * float(np.mean(n_filled[sel] == 2)), 1)}
    X = FEAT[IDX[sel]]; X_gate = X[:, -1, [FC.index(c) for c in gating]].astype(np.float32)
    y = (tv[FUT[sel]] - tv[sel]).astype(np.float32); w = CM['sample_weight'].to_numpy(np.float32)[sel]
    meta = pd.DataFrame({'ticker': CM['ticker'].to_numpy()[sel], 'asof': dates[sel], 'asof_future': dates[FUT[sel]], 'expiry': pd.NaT, 'expiry_future': pd.NaT,
                         'T': CM['T'].to_numpy()[sel], 'T_future': CM['T'].to_numpy()[sel], 'total_var_t': tv[sel], 'total_var_future': tv[FUT[sel]],
                         'target_T': CM['target_T'].to_numpy()[sel]})
    return X, X_gate, y, w, meta, stats


def roll(layout: Layout, m: str, meta: pd.DataFrame):
    """Ex-ante roll (paper eq. 6): day t's expiries, each Delta closer to expiry at t+h, at their day-t vols, through the same
    interpolation rules. MECH = roll value - value at t (0 when no bracket exists or an expiry's vol is more than 2x away from the
    bucket's vol). Also flags whether the expiries assumed at t are the ones bracketing the bucket at t+h ('same')."""
    parts_ = []
    for t_ in sorted(meta['ticker'].unique()):
        p_ = f'{layout.collected_dir(m)}/historical_svi_{t_}_all_enriched.parquet'
        if os.path.exists(p_): parts_.append(pd.read_parquet(p_, columns=['asof', 'T', 'ATM_IV']).assign(ticker=t_))
    EX = pd.concat(parts_, ignore_index=True); EX['asof'] = pd.to_datetime(EX['asof']).astype('datetime64[ns]')
    EX = EX[np.isfinite(EX['T']) & np.isfinite(EX['ATM_IV']) & (EX['ATM_IV'] > 0) & (EX['T'] >= 7 / 365) & (EX['T'] <= 150 / 365)].copy()
    EX['exp'] = EX['asof'] + pd.to_timedelta(np.round(EX['T'] * 365), unit='D'); EX['w'] = EX['ATM_IV'] ** 2 * EX['T']
    EX = EX.drop_duplicates(['ticker', 'asof', 'exp']).sort_values('T')
    RX = EX[['ticker', 'asof', 'T', 'exp', 'w']].assign(Tx=EX['T'])
    def asof_pair(tk, day, Tq):
        q = pd.DataFrame({'ticker': tk, 'asof': day, 'T': Tq, 'i': np.arange(len(Tq))}).dropna(subset=['T']).sort_values('T')
        lo = pd.merge_asof(q, RX, on='T', by=['ticker', 'asof'], direction='backward')
        hi = pd.merge_asof(q, RX, on='T', by=['ticker', 'asof'], direction='forward', allow_exact_matches=False)
        return lo.merge(hi[['i', 'exp', 'w', 'Tx']], on='i', suffixes=('_l', '_h')).set_index('i').reindex(np.arange(len(Tq)))
    tk = meta['ticker'].to_numpy(); d0 = pd.to_datetime(meta['asof']).astype('datetime64[ns]').to_numpy()
    d1 = pd.to_datetime(meta['asof_future']).astype('datetime64[ns]').to_numpy(); tm = meta['target_T'].to_numpy(float)
    tv0 = meta['total_var_t'].to_numpy(float)
    Dl = ((d1 - d0) / np.timedelta64(1, 'D')) / 365
    b0 = asof_pair(tk, d0, tm + Dl)
    Tl1, Th1 = b0['Tx_l'].to_numpy(float) - Dl, b0['Tx_h'].to_numpy(float) - Dl
    Tl1 = np.where(Tl1 >= 7 / 365, Tl1, np.nan)
    flat1 = ~np.isfinite(Tl1) & np.isfinite(Th1) & (Th1 - tm <= 25 / 365)
    v_l, v_h = b0['w_l'].to_numpy(float) / b0['Tx_l'].to_numpy(float), b0['w_h'].to_numpy(float) / b0['Tx_h'].to_numpy(float)
    wl, wh = v_l * Tl1, v_h * Th1
    wcf = np.where(flat1, v_h * tm, wl + (tm - Tl1) / (Th1 - Tl1) * (wh - wl))
    iv_cm0 = np.sqrt(tv0 / tm); lg2 = np.log(2)
    sane_ = (np.abs(np.log(np.sqrt(v_h) / iv_cm0)) <= lg2) & (flat1 | (np.abs(np.log(np.sqrt(v_l) / iv_cm0)) <= lg2))
    ok_ = np.isfinite(wcf) & sane_; mech = np.where(ok_, wcf - tv0, 0.0)
    b1 = asof_pair(tk, d1, tm)   # diagnostic only: expiries actually bracketing the bucket at t+h
    same_ = (np.where(np.isfinite(Tl1), b0['exp_l'].to_numpy(), np.datetime64('NaT', 'ns')) == b1['exp_l'].to_numpy()) | (flat1 & b1['Tx_l'].isna().to_numpy())
    same_ &= b0['exp_h'].to_numpy() == b1['exp_h'].to_numpy()
    return mech, np.asarray(same_, bool)


def build(layout: Layout, m: str, units) -> dict:
    """Model-ready state of market m from its collected files; saved to layout.state_dir(m)."""
    spec = SPECS[m]
    df = load_collected(layout, m, set(t for t, _ in units))
    df = df.sort_values(['ticker', 'expiry', 'asof']).reset_index(drop=True)
    df = df[~df['ticker'].astype(str).isin(spec.cross_listed)].copy()
    df['asof'] = pd.to_datetime(df['asof'])
    c0, c1 = (pd.Timestamp(x) for x in COVID_WINDOW)
    df = df[~((df['asof'] >= c0) & (df['asof'] <= c1))].copy()
    CM, CAL_DATES, _ = constant_maturity(df, spec.buckets)
    X, X_gate, yfull, w, meta, stats = samples(CM)
    CM30 = CM.loc[CM['target_T'] == 0.08, ['ticker', 'asof', 'ATM_IV']].copy()
    mech, same = roll(layout, m, meta)
    meta['mech'] = mech; meta['same'] = same
    yfull = meta['total_var_future'].to_numpy(float) - meta['total_var_t'].to_numpy(float)
    y = (yfull - mech).astype(np.float32)
    dec = pd.to_datetime(meta['asof']).dt.year.to_numpy() <= 2025   # reported in Section 4.1 (the target is net of the roll in every market)
    stats |= {'samples': int(len(y)), 'names': int(meta['ticker'].nunique()), 'roll available %': round(100 * float(np.mean(mech != 0)), 1),
              'roll vs no-change skill, samples <= 2025 %': round(100 * float(1 - np.abs(yfull - mech)[dec].mean() / np.abs(yfull)[dec].mean()), 2),
              'sd net of roll / sd of full change, samples <= 2025': round(float((yfull - mech)[dec].std() / yfull[dec].std()), 3)}
    st = layout.state_dir(m); os.makedirs(st, exist_ok=True)
    json.dump(stats, open(f'{st}/state_stats.json', 'w'), indent=1)
    np.savez(f'{st}/arrays.npz', X=X, X_gate=X_gate, y=y, w=w, MECH_ALL=mech, CAL_DATES=pd.to_datetime(np.asarray(CAL_DATES)).values)
    meta.to_parquet(f'{st}/meta.parquet', index=False); CM.to_parquet(f'{st}/CM.parquet', index=False); CM30.to_parquet(f'{st}/CM30.parquet', index=False)
    print(f'{m}: state saved to {st}: {len(y):,} samples on {meta["ticker"].nunique()} names, buckets {sorted(meta["target_T"].unique())}')
    return dict(X=X, X_gate=X_gate, y=y, w=w, meta=meta, CM=CM, CM30=CM30, CAL_DATES=CAL_DATES)
