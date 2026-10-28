"""Daily regime inputs computed from each underlying's own history (paper Table 3): percentile rank and z-score of the ATM vol and
skew over the previous 60 dates (expanding window for the first 60), and the slope of ATM vol across maturities."""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK, MIN_PERIODS = 60, 20


def _rank(series: pd.Series) -> pd.Series:
    """Share of the window's values below today's value (window: all dates so far, then the last 60)."""
    out = pd.Series(index=series.index, dtype=float); v = series.values
    for i in range(len(series)):
        if i < MIN_PERIODS - 1:
            out.iloc[i] = np.nan
        else:
            w = v[:i + 1] if i < LOOKBACK else v[i - LOOKBACK + 1:i + 1]
            out.iloc[i] = (w < v[i]).sum() / len(w)
    return out


def _zscore(series: pd.Series) -> pd.Series:
    out = pd.Series(index=series.index, dtype=float); v = series.values
    for i in range(len(series)):
        if i < MIN_PERIODS - 1:
            out.iloc[i] = np.nan
        else:
            w = v[:i + 1] if i < LOOKBACK else v[i - LOOKBACK + 1:i + 1]
            out.iloc[i] = (v[i] - np.mean(w)) / (np.std(w) + 1e-8)
    return out


def _term_slope(group: pd.DataFrame) -> pd.Series:
    if len(group) < 2:
        return pd.Series(0.0, index=group.index)
    try:
        return pd.Series(np.polyfit(group['T'], group['ATM_IV'], 1)[0], index=group.index)
    except Exception:
        return pd.Series(0.0, index=group.index)


def enrich(df_svi: pd.DataFrame) -> pd.DataFrame:
    """Adds IV_percentile, ATM_IV_zscore, skew_percentile, ATM_skew_zscore (daily means over expiries) and term_slope to the SVI rows
    of one underlying; missing values are carried forward, then set to 0."""
    df = df_svi.copy()
    df['asof'] = pd.to_datetime(df['asof']); df['expiry'] = pd.to_datetime(df['expiry'])
    df = df.sort_values(['asof', 'expiry']).reset_index(drop=True)
    daily = df.groupby('asof').agg({'ATM_IV': 'mean', 'ATM_skew': 'mean'}).sort_index()
    daily['IV_percentile'] = _rank(daily['ATM_IV'])
    daily['ATM_IV_zscore'] = _zscore(daily['ATM_IV'])
    daily['skew_percentile'] = _rank(daily['ATM_skew'])
    daily['ATM_skew_zscore'] = _zscore(daily['ATM_skew'])
    df = df.merge(daily[['IV_percentile', 'ATM_IV_zscore', 'skew_percentile', 'ATM_skew_zscore']].reset_index(), on='asof', how='left')
    df['term_slope'] = df.groupby('asof', group_keys=False).apply(_term_slope)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].ffill().fillna(0)
    return df
