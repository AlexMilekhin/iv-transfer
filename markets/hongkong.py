"""Hong Kong: HKEX DTOP stock-option files (daily) -> universe and option chains (paper Sections 3.1-3.2).

    python markets/hongkong.py --root DATA [--end 2026-10-14]

1. DTOP_O_{YYYYMMDD}.zip for every business day 2018-01-01 .. END (the paper's 2026 download ran on 15 Oct 2026).
2. Series file of HKD stock options (SOM): two layouts (22 fields with the expiry day, 21 without); per series and day the settlement
   price, turnover (volume) and open interest. Expiry per contract month: the stated day; else (2018-25) the last date any series of the
   month appears, or the business day before the month's last business day.
3. Forward per (date, name, expiry) by put-call parity on settlement prices (median over the 3 strikes with the smallest |C - P|);
   carry per day = median over names of ln(F2 / F1) / (T2 - T1), clipped to [-5%, 15%].
4. Universe: >= 3 traded near-money strikes (|ln K/F| <= 10%) in both the first and the second expiry on >= 60% of days (strict: 5 on 80%).
5. Chains: traded options of the universe names, price = settlement, spot = nearest forward discounted at the day's carry.
"""
import argparse
import csv
import glob
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivtransfer.config import SPECS, Layout   # noqa: E402

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}
C7 = ['gross', 'net', 'netchg', 'to', 'deal', 'settle', 'chg']


def download(zd, end):
    import requests
    os.makedirs(zd, exist_ok=True)
    def dl(d):
        fp = f'{zd}/{d:%Y%m%d}.zip'
        if os.path.exists(fp) or os.path.exists(fp + '.none'): return 'cached'
        for k in range(3):
            try:
                r = requests.get(f'https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_O_{d:%Y%m%d}.zip', headers=UA, timeout=60)
                if r.status_code == 200 and r.content[:2] == b'PK' and len(r.content) > 5000: open(fp, 'wb').write(r.content); return 'ok'
                if r.status_code in (200, 404): open(fp + '.none', 'w').write(''); return 'none'   # holiday
            except Exception: pass
            time.sleep(2 + 3 * k)
        return 'fail'
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(dl, pd.bdate_range('2018-01-01', end)))
    print('download:', pd.Series(res).value_counts().to_dict())


def parse(fp):
    try: z = zipfile.ZipFile(fp)
    except Exception: return None
    n = next((x for x in z.namelist() if x.lower().endswith('_opt_dtl_all.raw')), None)
    if n is None: return None
    out = []
    for r in (r for r in csv.reader(z.read(n).decode('latin-1').splitlines()) if r and r[0].strip() == '01'):
        if r[1].strip() != 'SOM': continue   # HKD stock options (RSO = RMB-traded, dropped)
        if len(r) >= 22: d_, m_, y_, rest = r[4], r[5], r[6], r[7:]
        elif len(r) == 21: d_, m_, y_, rest = '', r[4], r[5], r[6:]
        else: continue
        out.append([r[3].strip(), d_.strip(), (m_.strip() + y_.strip().zfill(2)).upper()] + rest[:15])
    if not out: return None
    D = pd.DataFrame(out, columns=['und', 'e_d', 'ym', 'K'] + ['c_' + c for c in C7] + ['p_' + c for c in C7])
    for c in D.columns[3:]: D[c] = pd.to_numeric(D[c], errors='coerce')
    D['date'] = pd.Timestamp(os.path.basename(fp)[:8]); D['fmt'] = 22 if (D['e_d'] != '').all() else 21
    o = pd.concat([pd.DataFrame({'date': D['date'], 'und': D['und'], 'ym': D['ym'], 'e_d': D['e_d'], 'K': D['K'], 'cp': cp, 'settle': D[pre + 'settle'],
                                 'volume': D[pre + 'to'], 'oi': D[pre + 'gross'], 'fmt': D['fmt']}) for cp, pre in (('C', 'c_'), ('P', 'p_'))], ignore_index=True)
    o = o[(o['settle'] > 0) | (o['volume'] > 0) | (o['oi'] > 0)]
    for c in ('und', 'ym', 'e_d', 'cp'): o[c] = o[c].astype('category')
    for c in ('K', 'settle'): o[c] = o[c].astype('float32')
    for c in ('volume', 'oi'): o[c] = o[c].fillna(0).astype('int32')
    o['fmt'] = o['fmt'].astype('int8'); return o


def rd(fs, cols=None):
    import pyarrow.parquet as pq
    return pq.read_table(fs if isinstance(fs, str) else list(fs), columns=cols).to_pandas(strings_to_categorical=True)


def yearly(zd, ch, years):
    from joblib import Parallel, delayed
    md = f'{ch}/m'; os.makedirs(md, exist_ok=True)
    for Y in years:
        if os.path.exists(f'{ch}/opt_{Y}.parquet'): continue
        for M in range(1, 13):   # month by month (low memory)
            mp = f'{md}/opt_{Y}{M:02d}.parquet'; fs = sorted(glob.glob(f'{zd}/{Y}{M:02d}*.zip'))
            if os.path.exists(mp) or not fs: continue
            parts = [x for x in Parallel(n_jobs=2)(delayed(parse)(f) for f in fs) if x is not None]
            if parts: pd.concat(parts, ignore_index=True).astype({'und': str, 'ym': str, 'e_d': str, 'cp': str}).to_parquet(mp, index=False)
        rd(sorted(glob.glob(f'{md}/opt_{Y}*.parquet'))).to_parquet(f'{ch}/opt_{Y}.parquet', index=False)


def expiries(ch, years, observed=True):
    """Expiry date per contract month. 2018-25 (observed=True): stated day, else the last date observed if seen through its month,
    else the business day before the month's last business day. 2026: stated day, else that rule."""
    ALL = rd([f'{ch}/opt_{Y}.parquet' for Y in years], ['date', 'ym', 'e_d'])
    stated = ALL[ALL.e_d.astype(str) != ''].drop_duplicates('ym').set_index('ym')['e_d']; EXP = {}
    for ym, g in ALL.groupby('ym', observed=True):
        mth = pd.to_datetime('01' + ym, format='%d%b%y')
        if ym in stated.index: EXP[ym] = pd.to_datetime(str(stated[ym]).zfill(2) + ym, format='%d%b%y')
        elif observed and g['date'].max() >= mth: EXP[ym] = g['date'].max()
        else: EXP[ym] = (mth + pd.offsets.BMonthEnd(1)) - pd.offsets.BDay(1)
    for Y in years:
        A = rd(f'{ch}/opt_{Y}.parquet')
        if 'exp' not in A: A['exp'] = pd.to_datetime(A['ym'].astype(str).map(EXP)); A.to_parquet(f'{ch}/opt_{Y}.parquet', index=False)


def year_frames(ch, Y):
    A = rd(f'{ch}/opt_{Y}.parquet', ['date', 'und', 'exp', 'K', 'cp', 'settle', 'volume']); A['und'] = A['und'].astype(str)
    A = A[A['exp'] > A['date']]
    k_ = ['date', 'und', 'exp', 'K']
    W = A[A.cp == 'C'][k_ + ['settle']].rename(columns={'settle': 'C'}).merge(A[A.cp == 'P'][k_ + ['settle']].rename(columns={'settle': 'P'}), on=k_)
    W = W[(W.C > 0) & (W.P > 0)]; W['gap'] = (W.C - W.P).abs(); W['Fk'] = W.K + W.C - W.P
    Fw = W.sort_values('gap').groupby(['date', 'und', 'exp']).head(3).groupby(['date', 'und', 'exp'], as_index=False)['Fk'].median().rename(columns={'Fk': 'F'})
    Fw = Fw[Fw.F > 0]
    Fw['T'] = (Fw.exp - Fw.date).dt.days / 365; Fw['er'] = Fw.groupby(['date', 'und'])['exp'].rank(method='dense')
    P2 = Fw[Fw.er == 1].merge(Fw[Fw.er == 2], on=['date', 'und'], suffixes=('1', '2'))
    P2 = P2[(P2.T2 - P2.T1) > 10 / 365]; P2['r'] = np.log(P2.F2 / P2.F1) / (P2.T2 - P2.T1)
    r_day = P2.groupby('date')['r'].median().clip(-0.05, 0.15)
    F1 = Fw[Fw.er == 1][['date', 'und', 'F', 'T']].rename(columns={'F': 'F1', 'T': 'T1'})
    O = A.merge(Fw[['date', 'und', 'exp', 'F', 'er']], on=['date', 'und', 'exp'])
    return A, O, F1, r_day


def liquidity(A, O, Y):
    nm = O[(O.volume > 0) & (np.abs(np.log(O.K / O.F)) <= 0.10)]; nd = A['date'].nunique()
    n_ = nm[nm.er <= 2].groupby(['date', 'und', 'er'])['K'].nunique().unstack('er').fillna(0)
    sh = lambda s_: ((n_.get(1.0, 0) >= s_) & (n_.get(2.0, 0) >= s_)).groupby(level='und').sum() / nd
    L = pd.DataFrame({'underlying': sorted(A['und'].unique())}).assign(year=Y)
    L['in_universe'] = L['underlying'].map(sh(3) >= 0.6).fillna(False); L['strict'] = L['underlying'].map(sh(5) >= 0.8).fillna(False)
    return L


def chain(O, F1, r_day, names):
    c = O[O.und.isin(names) & (O.volume > 0) & (O.settle > 0)].merge(F1, on=['date', 'und']).merge(r_day.rename('r').reset_index(), on='date', how='left')
    c['r'] = c['r'].fillna(r_day.median()); c['S_avg'] = c.F1 * np.exp(-c.r * c.T1)
    c['days'] = (c.exp - c.date).dt.days; c = c[c.days.between(5, 400) & (c.K / c.S_avg).between(0.5, 2.0)]
    c = c.rename(columns={'und': 'underlying', 'exp': 'expiration', 'K': 'strike', 'cp': 'call_put', 'settle': 'avg', 'volume': 'trades'})
    c['call_put'] = c['call_put'].astype(str); c = c.drop_duplicates(['date', 'underlying', 'expiration', 'call_put', 'strike'])
    return c[['date', 'underlying', 'expiration', 'strike', 'call_put', 'avg', 'trades', 'S_avg', 'r', 'F']]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='./data'); ap.add_argument('--end', default='2026-10-14')
    a = ap.parse_args(); L_ = Layout(a.root); raw = L_.raw('HK'); ch = f'{raw}/yearly'; cv = os.path.dirname(L_.chain_file('HK', 2020))
    for d in (raw, ch, cv, os.path.dirname(L_.in_year_file('HK'))): os.makedirs(d, exist_ok=True)
    download(f'{raw}/dtop', a.end)
    yearly(f'{raw}/dtop', ch, range(2018, 2027))
    expiries(ch, range(2018, 2026), observed=True); expiries(ch, [2026], observed=False)
    frames = {Y: year_frames(ch, Y) for Y in range(2018, 2027)}
    L = pd.concat([liquidity(f[0], f[1], Y) for Y, f in frames.items()], ignore_index=True); L.to_parquet(L_.in_year_file('HK'), index=False)
    S_ = {y: set(g['underlying']) for y, g in L[L['in_universe']].groupby('year')}
    for Y, (A, O, F1, r_day) in frames.items():
        names = S_.get(Y, set()) | ((S_.get(Y - 1, set()) - S_.get(Y, set())) if Y in SPECS['HK'].test_years else set())
        c = chain(O, F1, r_day, names); c.to_parquet(f'{cv}/chain_{Y}.parquet', index=False)
        print(f'{Y}: universe {len(S_.get(Y, set()))}, chain names {len(names)}, {len(c):,} rows', flush=True)


if __name__ == '__main__':
    main()
