"""Brazil: B3 COTAHIST yearly files -> universe and option chains from average trade prices (paper Sections 3.1-3.2).

    python markets/br.py --root DATA

1. COTAHIST_A{Y}.ZIP (2015-2026) from B3's historical-series page, parsed from the fixed-width layout: spot rows (TPMERC 010) and
   option rows (070 calls, 080 puts); prices in cents with the quotation factor applied.
2. CDI (BCB series 4389), r = ln(1 + CDI).
3. Per year: options mapped to their stock by root and share class (most traded spot code; spot types 02 shares, 14 ETFs / funds,
   34 / 35 BDRs); strike-adjusted series (INDOPC != 0) and untraded series dropped. Universe: an underlying-year counts if on >= 50% of
   the year's trading days it has >= 3 TRADED options with strike 80-120% of spot and 15-60 days to expiry.
4. Chain per year: one row per (date, underlying, expiry, type, strike), the most traded series if duplicated; price = average trade
   price, S = the stock's average trade price, forward by carry F = S / exp(-rT). Chains are built for year Y's universe and, in test
   years, separately for the names of year Y-1's universe that are not in year Y's (as in the original run).
"""
import argparse
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivtransfer.config import SPECS, Layout   # noqa: E402

YEARS = range(2015, 2027)
URLS = ['https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{y}.ZIP', 'https://www.b3.com.br/pesquisapregao/download?filelist=COTAHIST_A{y}.ZIP']
FIELDS = [('date', 3, 10), ('codbdi', 11, 12), ('codneg', 13, 24), ('tpmerc', 25, 27), ('nomres', 28, 39), ('especi', 40, 49),
          ('open', 57, 69), ('high', 70, 82), ('low', 83, 95), ('avg', 96, 108), ('last', 109, 121), ('bid', 122, 134), ('ask', 135, 147),
          ('trades', 148, 152), ('qty', 153, 170), ('volume', 171, 188), ('strike', 189, 201), ('indopc', 202, 202), ('expiry', 203, 210),
          ('fatcot', 211, 217), ('isin', 231, 242), ('dismes', 243, 245)]
PRICE = ['open', 'high', 'low', 'avg', 'last', 'bid', 'ask', 'strike']
SPOT_BDI = ('02', '14', '34', '35')
COLS = ['date', 'codbdi', 'codneg', 'tpmerc', 'especi', 'bid', 'ask', 'avg', 'last', 'high', 'low', 'trades', 'qty', 'volume', 'strike', 'indopc', 'expiry', 'root']
KEYS = ['date', 'underlying', 'expiration']


# ---------------- 1. COTAHIST ----------------
def valid_zip(blob):
    try: return len(zipfile.ZipFile(io.BytesIO(blob)).namelist()) > 0 and len(blob) > 1_000_000
    except Exception: return False


def fetch(raw, y):
    import requests
    p = f'{raw}/COTAHIST_A{y}.ZIP'
    if os.path.exists(p):
        blob = open(p, 'rb').read()
        if valid_zip(blob): return blob
        os.remove(p)
    for u in URLS:
        for attempt in range(3):   # streamed, retried: B3 drops connections on big files
            try:
                with requests.get(u.format(y=y), timeout=300, stream=True, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                    if r.status_code != 200: break
                    blob = b''.join(r.iter_content(chunk_size=1 << 20))
                if valid_zip(blob): open(p, 'wb').write(blob); return blob
            except Exception as e:
                print(f'  {y} attempt {attempt + 1}: {type(e).__name__}')
            time.sleep(5)
    return None


def parse_bytes(data):
    """Vectorised fixed-width parse (every record 245 characters + line end); type-01 records of spot and option markets."""
    h = data.index(b'\n') + 1; L = data.index(b'\n', h) + 1 - h; data = data[h:]
    if len(data) % L: data = data[:len(data) - len(data) % L]
    M = np.frombuffer(data, dtype=np.uint8).reshape(-1, L)
    M = M[(M[:, 0] == ord('0')) & (M[:, 1] == ord('1'))]
    tp = M[:, 24:27]
    keep = (tp[:, 0] == ord('0')) & ((tp[:, 1:] == np.frombuffer(b'10', np.uint8)).all(1) | (tp[:, 1:] == np.frombuffer(b'70', np.uint8)).all(1)
                                     | (tp[:, 1:] == np.frombuffer(b'80', np.uint8)).all(1))
    M = M[keep]
    def txt(a, b): return np.char.strip(np.ascontiguousarray(M[:, a - 1:b]).view(f'S{b - a + 1}').ravel().astype(str))
    def num(a, b):
        d = M[:, a - 1:b].astype(np.int64) - 48; d = np.where((d < 0) | (d > 9), 0, d)
        return d @ (10 ** np.arange(b - a, -1, -1, dtype=np.int64))
    df = pd.DataFrame({nm: (txt(a, b) if nm in ('codbdi', 'codneg', 'tpmerc', 'nomres', 'especi', 'isin') else num(a, b)) for nm, a, b in FIELDS})
    df['date'] = pd.to_datetime(df.date.astype(str), format='%Y%m%d')
    df['expiry'] = pd.to_datetime(df.expiry.where(df.expiry != 99991231).astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    df['volume'] = df.volume / 100
    fat = df.fatcot.replace(0, 1)
    for c in PRICE: df[c] = df[c] / 100 / (fat if c != 'strike' else 1)
    df['root'] = df.codneg.str[:4]
    return df


def cotahist(raw, y):
    out = f'{raw}/cotahist_{y}.parquet'
    if os.path.exists(out): return out
    z = fetch(raw, y)
    if z is None: raise SystemExit(f'{y}: download failed - put COTAHIST_A{y}.ZIP into {raw} by hand and rerun')
    zf = zipfile.ZipFile(io.BytesIO(z)); parse_bytes(zf.read(zf.namelist()[0])).to_parquet(out)
    return out


def cdi(raw, years):
    p = f'{raw}/cdi.parquet'
    if os.path.exists(p): return pd.read_parquet(p).cdi
    import requests
    parts = []
    for y in range(min(years) - 1, max(years) + 1):   # BCB caps a daily query at 10 years
        u = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4389/dados?formato=json&dataInicial=01/01/{y}&dataFinal=31/12/{y}'
        parts.append(pd.DataFrame(requests.get(u, timeout=60, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}).json()))
    d = pd.concat(parts)
    s = pd.Series(pd.to_numeric(d.valor, errors='coerce').values / 100, index=pd.to_datetime(d.data, format='%d/%m/%Y')).dropna().sort_index()
    s = s[~s.index.duplicated()]; assert s.between(0.01, 0.30).all(), 'CDI outside 1-30%'
    s.rename('cdi').to_frame().to_parquet(p); return s


# ---------------- 2. universe and chains ----------------
def options(raw, y):
    df = pd.read_parquet(f'{raw}/cotahist_{y}.parquet', columns=COLS)
    spot = df[(df.tpmerc == '010') & df.codbdi.isin(SPOT_BDI)].copy(); spot['cls'] = spot.especi.str.split().str[0]
    opt = df[df.tpmerc.isin(['070', '080'])].copy(); del df
    opt['cls'] = opt.especi.str.split().str[0]
    sp = spot.groupby(['root', 'cls', 'codneg']).volume.sum().reset_index().sort_values('volume').drop_duplicates(['root', 'cls'], keep='last')
    opt = opt.merge(sp[['root', 'cls', 'codneg']].rename(columns={'codneg': 'underlying'}), on=['root', 'cls'], how='left')
    opt = opt[opt.underlying.notna() & (opt.indopc == 0) & (opt.avg > 0) & (opt.trades > 0)]
    px = spot.rename(columns={'codneg': 'underlying', 'avg': 'S_avg', 'last': 'S_close', 'codbdi': 'spot_bdi'})[['date', 'underlying', 'S_avg', 'S_close', 'spot_bdi']]
    opt = opt.merge(px.drop_duplicates(['date', 'underlying']), on=['date', 'underlying'], how='inner')
    opt['days'] = (opt.expiry - opt.date).dt.days
    return opt


def liquidity(opt, y):
    mon = (opt.strike / opt.S_close).between(0.8, 1.2) & opt.days.between(15, 60); ndays = opt.date.nunique()
    v2 = (opt[mon].groupby(['underlying', 'date']).size() >= 3).groupby(level=0).sum() / ndays
    tab = pd.DataFrame({'share_days_traded': v2}).fillna(0)
    tab['in_universe'] = tab.share_days_traded >= 0.5
    tab['kind'] = opt.groupby('underlying').spot_bdi.first().reindex(tab.index).map({'02': 'share', '14': 'ETF/fund', '34': 'BDR', '35': 'BDR'})
    tab['year'] = y
    return tab.reset_index().rename(columns={'index': 'underlying'})


def chain(opt, names, rates):
    o = opt[opt.underlying.isin(names) & opt.days.between(5, 400) & (opt.strike / opt.S_avg).between(0.5, 2.0)]
    c = pd.DataFrame({'date': o.date.values, 'underlying': o.underlying.values, 'expiration': o.expiry.values, 'strike': o.strike.values,
                      'call_put': np.where(o.tpmerc.values == '070', 'C', 'P'), 'avg': o.avg.values, 'last': o['last'].values,
                      'trades': o.trades.values, 'qty': o.qty.values, 'volume': o.volume.values,
                      'S_avg': o.S_avg.values, 'S_close': o.S_close.values, 'days': o.days.values})
    c = c.sort_values('trades', ascending=False).drop_duplicates(KEYS + ['call_put', 'strike'], keep='first')
    rr = np.log1p(rates.reindex(pd.DatetimeIndex(sorted(c.date.unique())), method='ffill'))
    c['r'] = c.date.map(rr).values; c['T'] = c.days / 365.0; c['D'] = np.exp(-c.r * c['T']); c['F'] = c.S_avg / c.D
    return c.sort_values(KEYS + ['call_put', 'strike']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument('--root', default='./data')
    a = ap.parse_args(); L_ = Layout(a.root); raw = L_.raw('BR'); cv = os.path.dirname(L_.chain_file('BR', 2020))
    for d in (raw, cv, os.path.dirname(L_.in_year_file('BR'))): os.makedirs(d, exist_ok=True)
    for y in YEARS: cotahist(raw, y)
    rates = cdi(raw, YEARS)
    LIQ = [liquidity(options(raw, y), y) for y in YEARS]
    L = pd.concat(LIQ, ignore_index=True); L.to_parquet(L_.in_year_file('BR'), index=False)
    S_ = {y: set(g['underlying']) for y, g in L[L['in_universe']].groupby('year')}
    for y in YEARS:
        opt = options(raw, y); parts = [chain(opt, S_.get(y, set()), rates)]
        add = (S_.get(y - 1, set()) - S_.get(y, set())) if y in SPECS['BR'].test_years else set()
        if add: parts.append(chain(opt, add, rates))
        c = pd.concat(parts, ignore_index=True); c.to_parquet(f'{cv}/chain_{y}.parquet', index=False)
        print(f'{y}: universe {len(S_.get(y, set()))}, ex-ante additions {len(add)}, chain {len(c):,} rows', flush=True)


if __name__ == '__main__':
    main()
