"""Canada: Montreal Exchange historical option data -> universe and option chains (paper Sections 3.1-3.2).

    python markets/canada.py --root DATA [--threads 4]

1. Every listed option class (adjusted classes skipped), every half-year window 2017-2026, as CSV from the exchange's historical-data
   page (resumable; a window that times out can be fetched month by month and joined).
2. Per class: two-sided quotes on Mon / Wed / Fri (dates up to 18 Oct 2026), mid = (bid + ask) / 2; forward per (date, expiry) by
   put-call parity (median over the 3 strikes with the smallest |C - P|); the nearest expiry of 7-120 days gives (Fa, Ta); carry per
   day from the first expiry at least 20 days later.
3. Universe: ATM (|ln K/F| <= 0.03, 14-60 days) call and put quoted on >= 80% of the year's grid dates and median ATM spread <= 15%
   (strict: <= 10%).
4. Chains for year Y's universe and the ex-ante test additions: price = mid, spot = Fa exp(-r Ta) with r = the day's median carry over
   classes (clipped to [-5%, 15%]); strikes within 50-200% of spot.
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivtransfer.config import SPECS, Layout   # noqa: E402

Y0, Y1 = 2017, 2026
END = pd.Timestamp('2026-10-18')
URL = 'https://www.m-x.ca/en/trading/data/historical'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36', 'Referer': URL}
WIN = [(f'{y}-01-01', f'{y}-06-30') for y in range(Y0, Y1 + 1)] + [(f'{y}-07-01', f'{y}-12-31') for y in range(Y0, Y1 + 1)]
num = lambda s: pd.to_numeric(s, errors='coerce')


# ---------------- 1. download ----------------
def download(out, threads=4):
    import requests
    from bs4 import BeautifulSoup
    os.makedirs(out, exist_ok=True)
    def fetch(sym, a, b, tries=3):
        fp = os.path.join(out, f'{sym.replace("/", "_")}_{a}_{b}.csv')
        if os.path.exists(fp): return 'ok'
        if os.path.exists(fp + '.none'): return 'none'
        for k in range(tries):
            try:
                r = requests.get(URL, params={'symbol': sym, 'from': a, 'to': b, 'dnld': 1}, headers=UA, timeout=240)
                t = r.text
                if r.status_code == 200 and t.count('\n') >= 2 and '<html' not in t[:500].lower():
                    open(fp, 'w', encoding='utf-8').write(t); return 'ok'
                if r.status_code == 200: open(fp + '.none', 'w').write(''); return 'none'
                if r.status_code in (403, 429, 503): time.sleep(60 * (k + 1)); continue   # throttled
            except Exception: pass
            time.sleep(3 * (k + 1))
        return 'fail'
    def history(sym):
        st = {}
        for a, b in WIN:
            r = fetch(sym, a, b); st[r] = st.get(r, 0) + 1; time.sleep(0.5)
        return sym, st
    soup = BeautifulSoup(requests.get(URL, headers=UA, timeout=60).text, 'html.parser')
    syms = sorted(x for x in {(o.get('value') or '').strip() for o in soup.find('select', {'name': 'o'}).find_all('option')} - {''} if not x[-1].isdigit())
    print(f'{len(syms)} option classes x {len(WIN)} half-years')
    with ThreadPoolExecutor(threads) as ex:
        for k, (sym, st) in enumerate(ex.map(history, syms)):
            if k % 50 == 0: print(f'  {k}/{len(syms)} {sym} {st}', flush=True)


# ---------------- 2. per class ----------------
USE = {'Date', 'Symbol', 'Root Symbol', 'Strike Price', 'Expiry Date', 'Call/Put', 'Ins. Type', 'Bid Price', 'Ask Price', 'Last Price', 'Volume'}
USE_B = {'Date', 'Symbol', 'Root Symbol', 'Strike Price', 'Expiry Date', 'Call/Put', 'Ins. Type', 'Bid Price', 'Ask Price'}


def csvs(dirs, nm):
    return sorted(f for d_ in dirs for f in glob.glob(f'{d_}/{nm}_*.csv') if os.path.basename(f).rsplit('_', 2)[0] == nm)


def read_class(dirs, nm, cols):
    D = []
    for f in csvs(dirs, nm):
        try: d = pd.read_csv(f, usecols=lambda c: c.strip() in cols, low_memory=False)
        except Exception: continue
        d.columns = [c.strip() for c in d.columns]; D.append(d)
    return pd.concat(D, ignore_index=True) if D else None


def _quotes(D, with_volume):
    root = D['Root Symbol'].astype(str).str.strip(); D = D[root == root.mode().iloc[0]]
    cp = D['Symbol'].astype(str).str.extract(r'\d{6}([CP])', expand=False).fillna(num(D['Call/Put']).map({0: 'C', 1: 'P'}))
    O = pd.DataFrame({'date': D['Date'], 'exp': pd.to_datetime(D['Expiry Date'], errors='coerce'), 'K': num(D['Strike Price']), 'cp': cp,
                      'bid': num(D['Bid Price']), 'ask': num(D['Ask Price'])} | ({'volume': num(D['Volume']).fillna(0)} if with_volume else {}))
    return O[O.cp.isin(['C', 'P']) & (O.exp > O.date) & (O.K > 0) & (O.bid > 0) & (O.ask >= O.bid)].drop_duplicates(['date', 'exp', 'K', 'cp'])


def _forwards(O):
    k_ = ['date', 'exp', 'K']
    W = O[O.cp == 'C'][k_ + ['mid']].rename(columns={'mid': 'C'}).merge(O[O.cp == 'P'][k_ + ['mid']].rename(columns={'mid': 'P'}), on=k_)
    W['gap'] = (W.C - W.P).abs(); W['Fk'] = W.K + W.C - W.P
    return W.sort_values('gap').groupby(['date', 'exp']).head(3).groupby(['date', 'exp'], as_index=False)['Fk'].median().rename(columns={'Fk': 'F'})


def proc(dirs, pt, nm):
    """Two-sided quotes on Mon / Wed / Fri, parity forwards, carry per day, statistics per year -> {pt}/{nm}.parquet, _r, _stats."""
    sp = f'{pt}/{nm}_stats.parquet'
    if os.path.exists(sp): return pd.read_parquet(sp)
    D = read_class(dirs, nm, USE)
    if D is None: return None
    D['Date'] = pd.to_datetime(D['Date'], errors='coerce')
    D = D[D['Date'].dt.year.between(Y0, Y1) & (D['Date'] <= END) & D['Date'].dt.weekday.isin([0, 2, 4])]
    it = num(D['Ins. Type']); U = D[it == 3].assign(S_mx=num(D['Last Price'])).groupby('Date')['S_mx'].last()
    if D[it == 0].empty: return None
    O = _quotes(D[it == 0], True)
    if O.empty: return None
    O['mid'] = (O.bid + O.ask) / 2; O['spr'] = (O.ask - O.bid) / O.mid
    Fw = _forwards(O)
    Fw = Fw[Fw.F > 0].copy(); Fw['T'] = (Fw.exp - Fw.date).dt.days / 365; Fw['er'] = Fw.groupby('date')['exp'].rank(method='dense')
    A_ = Fw[Fw['T'].between(7 / 365, 120 / 365)].sort_values(['date', 'T'])
    Fa = A_.groupby('date').head(1)[['date', 'exp', 'F', 'T']].rename(columns={'exp': 'ea', 'F': 'Fa', 'T': 'Ta'})
    B_ = A_.merge(Fa[['date', 'Ta']], on='date'); B_ = B_[B_['T'] >= B_['Ta'] + 20 / 365].groupby('date').head(1)[['date', 'F', 'T']].rename(columns={'F': 'Fb', 'T': 'Tb'})
    R_ = Fa.merge(B_, on='date', how='left'); R_['r'] = np.log(R_.Fb / R_.Fa) / (R_.Tb - R_.Ta)
    R_ = R_.merge(U.rename('S_mx'), left_on='date', right_index=True, how='left'); R_['sym'] = nm
    R_[['date', 'sym', 'Fa', 'Ta', 'r', 'S_mx']].to_parquet(f'{pt}/{nm}_r.parquet', index=False)
    O2 = O.merge(Fa[['date', 'ea', 'Fa']], left_on=['date', 'exp'], right_on=['date', 'ea']); O2['dk'] = (O2.K / O2.Fa - 1).abs()
    atm = O2[O2.dk == O2.groupby('date')['dk'].transform('min')].groupby('date').agg(ncp=('cp', 'nunique'), spr=('spr', 'mean'))
    atm = atm[atm.ncp == 2]
    O3 = O[O.volume > 0].merge(Fw[['date', 'exp', 'F', 'er']], on=['date', 'exp']); O3 = O3[np.abs(np.log(O3.K / O3.F)) <= 0.10]
    n_ = O3[O3.er <= 2].groupby(['date', 'er'])['K'].nunique().unstack('er').reindex(columns=[1.0, 2.0]).fillna(0)
    trd = (n_[1.0] >= 3) & (n_[2.0] >= 3)
    rows = []
    for Y in sorted(O.date.dt.year.unique()):
        a_ = atm[atm.index.year == Y]; oy = O[O.date.dt.year == Y]
        rows.append({'underlying': nm, 'year': int(Y), 'days': oy.date.nunique(), 'atm_days': len(a_), 'atm_spread_med': float(a_.spr.median()) if len(a_) else np.nan,
                     'traded3_days': int(trd[trd.index.year == Y].sum()), 'rows_per_day': len(oy) / max(1, oy.date.nunique())})
    P = O.merge(Fa[['date', 'Fa', 'Ta']], on='date').merge(Fw[['date', 'exp', 'F']], on=['date', 'exp'], how='left')
    P['days'] = (P.exp - P.date).dt.days; P = P[P.days.between(5, 400) & (P.K / P.Fa).between(0.5, 2.0)]
    P = P[['date', 'exp', 'K', 'cp', 'mid', 'volume', 'F', 'Fa', 'Ta']].astype({'K': 'float32', 'mid': 'float32', 'volume': 'float32', 'F': 'float32', 'Fa': 'float32', 'Ta': 'float32'})
    P.to_parquet(f'{pt}/{nm}.parquet', index=False)
    S = pd.DataFrame(rows); S.to_parquet(sp, index=False); return S


def atm_stats(dirs, pt, nm):
    """ATM (|ln K/F| <= 0.03, 14-60 days) quoted dates and median spread per year (the universe statistics)."""
    fp = f'{pt}/{nm}_atm.parquet'
    if os.path.exists(fp): return pd.read_parquet(fp)
    D = read_class(dirs, nm, USE_B)
    if D is None: return None
    D['Date'] = pd.to_datetime(D['Date'], errors='coerce')
    D = D[D['Date'].dt.year.between(Y0, Y1) & (D['Date'] <= END) & D['Date'].dt.weekday.isin([0, 2, 4]) & (num(D['Ins. Type']) == 0)]
    if D.empty: return None
    O = _quotes(D, False)
    O['mid'] = (O.bid + O.ask) / 2; O['spr'] = (O.ask - O.bid) / O.mid; O['days'] = (O.exp - O.date).dt.days
    Fw = _forwards(O)
    A = O[O.days.between(14, 60)].merge(Fw[Fw.F > 0], on=['date', 'exp']); A = A[np.abs(np.log(A.K / A.F)) <= 0.03]
    g = A.groupby('date').agg(ncp=('cp', 'nunique'))
    rows = []
    for Y in range(Y0, Y1 + 1):
        a_ = A[A.date.dt.year == Y]
        if not len(O[O.date.dt.year == Y]): continue
        rows.append({'underlying': nm, 'year': Y, 'atm30_days': int((g[g.index.year == Y].ncp == 2).sum()), 'atm30_quotes': len(a_),
                     'atm30_spread_med': float(a_.spr.median()) if len(a_) else np.nan})
    R = pd.DataFrame(rows); R.to_parquet(fp, index=False); return R


# ---------------- 3. universe and chains ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='./data'); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--skip-download', action='store_true')
    a = ap.parse_args(); L_ = Layout(a.root); raw = L_.raw('CA'); mx = f'{raw}/mx'; pt = f'{raw}/parts'; cv = os.path.dirname(L_.chain_file('CA', 2020))
    for d in (raw, pt, cv, os.path.dirname(L_.in_year_file('CA'))): os.makedirs(d, exist_ok=True)
    if not a.skip_download: download(mx, a.threads)
    dirs = [mx]
    classes = sorted({os.path.basename(f).rsplit('_', 2)[0] for f in glob.glob(f'{mx}/*.csv') if not os.path.basename(f).startswith('_')
                      and len(os.path.basename(f)[:-4].rsplit('_', 2)) == 3})
    from joblib import Parallel, delayed
    nj = min(8, os.cpu_count() or 1)
    Parallel(n_jobs=nj)(delayed(proc)(dirs, pt, c) for c in classes); Parallel(n_jobs=nj)(delayed(atm_stats)(dirs, pt, c) for c in classes)
    ST = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{pt}/*_stats.parquet')], ignore_index=True)
    AT = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{pt}/*_atm.parquet')], ignore_index=True)
    ST = ST.merge(AT, on=['underlying', 'year'], how='left')
    RT = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{pt}/*_r.parquet')], ignore_index=True)
    ST['ndays'] = ST.year.map(RT.groupby(RT.date.dt.year)['date'].nunique())
    ST['atm30_share'] = ST.atm30_days / ST.ndays; ST['traded3_share'] = ST.traded3_days / ST.ndays
    ST['in_universe'] = (ST.atm30_share >= 0.80) & (ST.atm30_spread_med <= 0.15); ST['strict'] = ST.in_universe & (ST.atm30_spread_med <= 0.10)
    ST.to_parquet(L_.in_year_file('CA'), index=False)
    S_ = {y_: set(g['underlying']) for y_, g in ST[ST.in_universe].groupby('year')}
    r_day = RT.dropna(subset=['r']).groupby('date')['r'].median().clip(-0.05, 0.15)
    cols = ['date', 'underlying', 'expiration', 'strike', 'call_put', 'avg', 'trades', 'S_avg', 'r', 'F']
    for Y in range(Y0, Y1 + 1):
        u_ = sorted(S_.get(Y, set()) | ((S_.get(Y - 1, set()) - S_.get(Y, set())) if Y in SPECS['CA'].test_years else set()))
        if not u_: continue
        lo, hi = pd.Timestamp(f'{Y}-01-01'), pd.Timestamp(f'{Y}-12-31')
        c = pd.concat([pd.read_parquet(f'{pt}/{n}.parquet', filters=[('date', '>=', lo), ('date', '<=', hi)]).assign(underlying=n) for n in u_ if os.path.exists(f'{pt}/{n}.parquet')], ignore_index=True)
        c = c.merge(r_day.rename('r').reset_index(), on='date', how='left'); c['r'] = c['r'].fillna(r_day.median()).astype('float32')
        c['S_avg'] = (c.Fa * np.exp(-c.r * c.Ta)).astype('float32'); c = c[(c.K / c.S_avg).between(0.5, 2.0)]
        c = c.rename(columns={'exp': 'expiration', 'K': 'strike', 'cp': 'call_put', 'mid': 'avg'}); c['trades'] = np.float32(1.0)
        c[cols].to_parquet(f'{cv}/chain_{Y}.parquet', index=False)
        print(f'{Y}: {len(u_)} names, {len(c):,} rows', flush=True)


if __name__ == '__main__':
    main()
