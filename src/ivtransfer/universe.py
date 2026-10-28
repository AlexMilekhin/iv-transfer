"""Universe lists (paper Section 3.2).

Each market script writes its in-year universe (the names meeting the market's liquidity rule on year Y's own data). The model uses
it as follows: training and validation years use their own year's list; test year Y uses year Y-1's list (ex ante), so a name that
was liquid in Y-1 but not in Y is an 'addition' (a test name only). Columns of the universe file: name, year, orig (in year Y's own
list), xa_test (a test name of year Y), added (test-only unit), unit (fitted).
"""
from __future__ import annotations

import pandas as pd

from .config import SPECS, Layout


def build(in_year: pd.DataFrame, test_years) -> pd.DataFrame:
    """in_year: columns name, year (one row per name in year's own universe)."""
    U_ = in_year[['name', 'year']].assign(name=lambda d: d['name'].astype(str), year=lambda d: d['year'].astype(int)).drop_duplicates()
    S_ = {y_: set(g['name']) for y_, g in U_.groupby('year')}
    for Y in test_years:
        assert Y in S_ and Y - 1 in S_, f'no universe rows for {Y} or {Y - 1}'
    rows = [{'name': n_, 'year': y_, 'orig': True, 'xa_test': (y_ in test_years) and (n_ in S_[y_ - 1])} for (n_, y_) in sorted(set(zip(U_['name'], U_['year'])))]
    for Y in test_years:
        rows += [{'name': n_, 'year': Y, 'orig': False, 'xa_test': True} for n_ in sorted(S_[Y - 1] - S_[Y])]
    X_ = pd.DataFrame(rows); X_['added'] = ~X_['orig']; X_['unit'] = True
    return X_


def counts(U: pd.DataFrame, test_years) -> pd.DataFrame:
    S_ = {y_: set(g['name']) for y_, g in U[U['orig']].groupby('year')}
    return pd.DataFrame([{'test year': Y, 'in-year universe': len(S_[Y]), 'ex-ante universe (Y-1 list)': len(S_[Y - 1]), 'in both': len(S_[Y] & S_[Y - 1]),
                          'entered on year-Y liquidity': len(S_[Y] - S_[Y - 1]), 'added (fell out in Y)': len(S_[Y - 1] - S_[Y])} for Y in test_years])


def write(layout: Layout, m: str) -> pd.DataFrame:
    L_ = pd.read_parquet(layout.in_year_file(m)); L_ = L_[L_['in_universe'].astype(bool)]
    U = build(L_.rename(columns={'underlying': 'name'}), SPECS[m].test_years)
    U.to_parquet(layout.universe_file(m), index=False)
    return counts(U, SPECS[m].test_years)


def units(layout: Layout, m: str):
    """-> (sorted units (name, year), set of ex-ante additions), cross-listed underlyings left out."""
    U = pd.read_parquet(layout.universe_file(m)); U = U[~U['name'].astype(str).isin(SPECS[m].cross_listed)]
    un = sorted(zip(U.loc[U['unit'], 'name'].astype(str), U.loc[U['unit'], 'year'].astype(int)))
    add = set(zip(U.loc[U['added'], 'name'].astype(str), U.loc[U['added'], 'year'].astype(int)))
    return un, add
