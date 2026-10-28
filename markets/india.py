"""India: NSE F&O bhavcopy (daily files) -> universe and option chains (paper Sections 3.1-3.2).

    python markets/india.py --root DATA

1. Daily F&O bhavcopies 2018-01-01 .. 2026-10-12 (old format fo{DD}{MON}{YYYY}bhav.csv.zip until 5 Jul 2024, UDiFF after).
2. Stock options and stock futures per year in one schema.
3. Universe: >= 3 traded near-money strikes (|ln K/F| <= 10%) in both the first and the second expiry on >= 60% of the year's days
   (strict: >= 5 strikes on >= 80%). F = the expiry's stock-future close, else the nearest future.
4. Chains: traded options of the universe names, price = close; spot = nearest future discounted at the 3-month rate (OECD, FRED
   INDIR3TIB01STM, continuous); one row per (date, name, expiry, type, strike).
"""
import argparse
import datetime as dt
import glob
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivtransfer.config import SPECS, Layout   # noqa: E402

END = dt.date(2026, 10, 12)


def url_for(d):
    if d < dt.date(2024, 7, 8):
        mon = d.strftime('%b').upper(); return f'https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{mon}/fo{d:%d}{mon}{d.year}bhav.csv.zip'
    return f'https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip'


def download(fo_dir):
    import requests
    os.makedirs(fo_dir, exist_ok=True)
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
                      'Accept': '*/*', 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.nseindia.com/'})
    try: S.get('https://www.nseindia.com', timeout=20)   # cookies
    except Exception: pass
    for d in pd.bdate_range('2018-01-01', END):
        d = d.date(); fp = f'{fo_dir}/{d:%Y%m%d}.csv.zip'
        if os.path.exists(fp) or os.path.exists(fp + '.none'): continue
        for k in range(3):
            try:
                r = S.get(url_for(d), timeout=30)
                if r.status_code == 200 and r.content[:2] == b'PK': open(fp, 'wb').write(r.content); break
                if r.status_code == 404: open(fp + '.none', 'w').write(''); break   # holiday
            except Exception: pass
            time.sleep(2 + 2 * k)
        time.sleep(0.3)


def norm(df):
    """One schema for both formats."""
    c = set(df.columns)
    if 'TckrSymb' in c:
        kind = df['FinInstrmTp'].map({'STO': 'OPTSTK', 'STF': 'FUTSTK'}).fillna('other')
        return pd.DataFrame({'date': pd.to_datetime(df['TradDt']), 'sym': df['TckrSymb'].astype(str).str.strip(), 'kind': kind, 'exp': pd.to_datetime(df['XpryDt']),
                             'K': df['StrkPric'].astype(float), 'cp': df['OptnTp'].astype(str).str.strip().str[0], 'close': df['ClsPric'].astype(float),
                             'settle': df['SttlmPric'].astype(float), 'traded': df['TtlTradgVol'].fillna(0) > 0, 'oi': df['OpnIntrst'].astype(float),
                             'und': df['UndrlygPric'].astype(float)})
    df = df.rename(columns=lambda x: x.strip())
    return pd.DataFrame({'date': pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y'), 'sym': df['SYMBOL'].astype(str).str.strip(), 'kind': df['INSTRUMENT'].str.strip(),
                         'exp': pd.to_datetime(df['EXPIRY_DT'], format='%d-%b-%Y'), 'K': df['STRIKE_PR'].astype(float), 'cp': df['OPTION_TYP'].astype(str).str.strip().str[0],
                         'close': df['CLOSE'].astype(float), 'settle': df['SETTLE_PR'].astype(float), 'traded': df['CONTRACTS'] > 0, 'oi': df['OPEN_INT'].astype(float), 'und': np.nan})


def _one(fp):
    z = zipfile.ZipFile(fp); n = norm(pd.read_csv(z.open(z.namelist()[0]), low_memory=False)); return n[n['kind'].isin(['OPTSTK', 'FUTSTK'])]


def parse(fo_dir, chn):
    from joblib import Parallel, delayed
    os.makedirs(chn, exist_ok=True); files = sorted(glob.glob(f'{fo_dir}/*.csv.zip'))
    for Y in range(2018, 2027):
        if os.path.exists(f'{chn}/opt_{Y}.parquet'): continue
        fy = [f for f in files if os.path.basename(f).startswith(str(Y))]
        if not fy: continue
        A = pd.concat(Parallel(n_jobs=os.cpu_count())(delayed(_one)(f) for f in fy), ignore_index=True)
        A[A.kind == 'FUTSTK'][['date', 'sym', 'exp', 'close', 'settle', 'oi', 'und']].to_parquet(f'{chn}/fut_{Y}.parquet', index=False)
        A[A.kind == 'OPTSTK'].drop(columns='kind').to_parquet(f'{chn}/opt_{Y}.parquet', index=False)
        print(f'{Y}: {len(fy)} days parsed', flush=True)


def rate(raw):
    import requests
    RF = f'{raw}/in_rate.parquet'
    if not os.path.exists(RF):   # India 3-month rate (OECD, monthly, %)
        t_ = pd.read_csv(io.StringIO(requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDIR3TIB01STM', timeout=60).text))
        t_.columns = ['date', 'r_pct']; t_['date'] = pd.to_datetime(t_['date']); t_['r_pct'] = pd.to_numeric(t_['r_pct'], errors='coerce'); t_.dropna().to_parquet(RF, index=False)
    return pd.read_parquet(RF).set_index('date')['r_pct'].sort_index()


def year_frame(chn, Y):
    O = pd.read_parquet(f'{chn}/opt_{Y}.parquet'); Fu = pd.read_parquet(f'{chn}/fut_{Y}.parquet', columns=['date', 'sym', 'exp', 'close']).rename(columns={'close': 'F'})
    O = O.merge(Fu, on=['date', 'sym', 'exp'], how='left')
    F1 = Fu.sort_values('exp').drop_duplicates(['date', 'sym']).rename(columns={'F': 'F1', 'exp': 'exp1'})
    O = O.merge(F1, on=['date', 'sym'], how='left'); O['F'] = O['F'].fillna(O['F1'])
    return O


def liquidity(O, Y):
    nm = O[O['traded'] & (np.abs(np.log(O['K'] / O['F'])) <= 0.10)].copy(); nm['er'] = nm.groupby(['date', 'sym'])['exp'].rank(method='dense')
    n_ = nm[nm.er <= 2].groupby(['date', 'sym', 'er'])['K'].nunique().unstack('er').fillna(0); nd = O['date'].nunique()
    sh = lambda s_: ((n_.get(1.0, 0) >= s_) & (n_.get(2.0, 0) >= s_)).groupby(level='sym').sum() / nd
    L = pd.DataFrame({'underlying': sorted(O['sym'].unique())}).assign(year=Y)
    L['in_universe'] = L['underlying'].map(sh(3) >= 0.6).fillna(False); L['strict'] = L['underlying'].map(sh(5) >= 0.8).fillna(False)
    return L


def chain(O, names, RT):
    c = O[O['sym'].isin(names) & O['traded'] & (O['close'] > 0) & O['F1'].notna()].copy()
    c['r'] = np.log1p(RT.reindex(c['date'].dt.to_period('M').dt.to_timestamp(), method='ffill').to_numpy() / 100)
    c['S_avg'] = c['F1'] * np.exp(-c['r'] * (c['exp1'] - c['date']).dt.days / 365)
    c['days'] = (c['exp'] - c['date']).dt.days; c = c[c['days'].between(5, 400) & (c['K'] / c['S_avg']).between(0.5, 2.0)]
    c = c.rename(columns={'sym': 'underlying', 'exp': 'expiration', 'K': 'strike', 'cp': 'call_put', 'close': 'avg'}).assign(trades=1)
    c = c.drop_duplicates(['date', 'underlying', 'expiration', 'call_put', 'strike'])
    return c[['date', 'underlying', 'expiration', 'strike', 'call_put', 'avg', 'trades', 'S_avg', 'r', 'F']]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument('--root', default='./data')
    a = ap.parse_args(); L_ = Layout(a.root); raw = L_.raw('IN'); chn = f'{raw}/yearly'; cv = os.path.dirname(L_.chain_file('IN', 2020))
    for d in (raw, cv, os.path.dirname(L_.in_year_file('IN'))): os.makedirs(d, exist_ok=True)
    download(f'{raw}/fo'); parse(f'{raw}/fo', chn); RT = rate(raw)
    years = [Y for Y in range(2018, 2027) if os.path.exists(f'{chn}/opt_{Y}.parquet')]
    L = pd.concat([liquidity(year_frame(chn, Y), Y) for Y in years], ignore_index=True); L.to_parquet(L_.in_year_file('IN'), index=False)
    S_ = {y: set(g['underlying']) for y, g in L[L['in_universe']].groupby('year')}
    for Y in years:
        names = S_.get(Y, set()) | ((S_.get(Y - 1, set()) - S_.get(Y, set())) if Y in SPECS['IN'].test_years else set())
        c = chain(year_frame(chn, Y), names, RT); c.to_parquet(f'{cv}/chain_{Y}.parquet', index=False)
        print(f'{Y}: universe {len(S_.get(Y, set()))}, chain names {len(names)}, {len(c):,} rows', flush=True)


if __name__ == '__main__':
    main()
