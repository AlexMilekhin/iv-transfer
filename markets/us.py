"""United States: DoltHub option-chain database (post-no-preference/options) -> universe and option chains (paper Sections 3.1-3.2).

    python markets/us.py --root DATA [--snapshot path/to/dolt_snapshot.tar]

Needs the dolt CLI (https://docs.dolthub.com/introduction/installation) and duckdb. Steps:
  1. clone the database (shallow) or unpack a saved snapshot; the paper used the snapshot of 13 Oct 2026 (dates up to 12 Oct 2026);
  2. export every quote of 2019 to 12 Oct 2026 to local parquet (date, symbol, expiry, strike, type, bid, ask, vendor vol, delta);
  3. universe rule per year 2020-25: ATM quotes (|delta| 0.4-0.6, bid > 0, ask > bid) on >= 80% of the year's dates (as many as the busiest
     name) and median ATM bid-ask spread <= 10% of mid. 2026: the 2025 list (fixed on 1 Jan 2026: ATM spreads widened from October 2025,
     so the 10% cut would keep 32 names);
  4. chains of every unit (name in year Y's list, plus in test years the names of year Y-1's list) -> chains/us/chain_{Y}.parquet,
     sorted by symbol, date, expiry, strike and type;
  5. FRED DTB3 (3-month Treasury bill) -> raw/us/dtb3.parquet.
"""
import argparse
import io
import os
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ivtransfer.config import SPECS, Layout   # noqa: E402

D_END = '2026-10-12'
COLS = 'date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta'


def sh(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def restore(work, snapshot=None):
    db = f'{work}/options'
    if os.path.exists(f'{db}/.dolt'): return db
    os.makedirs(work, exist_ok=True)
    if shutil.which('dolt') is None: sh('curl -sL https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash')
    sh('dolt config --global --add user.name ivt; dolt config --global --add user.email ivt@example.invalid')
    if snapshot and os.path.exists(snapshot):
        sh(f'tar -xf {snapshot} -C {work}')
    else:
        sh('dolt clone --depth 1 post-no-preference/options', cwd=work)
    assert os.path.exists(f'{db}/.dolt'), 'dolt database not restored'
    return db


def export_years(db, pq_dir, years):
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    os.makedirs(pq_dir, exist_ok=True)
    for Y in years:
        fp = f'{pq_dir}/opt_{Y}.parquet'
        if os.path.exists(fp): continue
        hi = D_END if Y == 2026 else f'{Y}-12-31'
        q = f"SELECT {COLS} FROM option_chain WHERE date >= '{Y}-01-01' AND date <= '{hi}'"
        with open(fp, 'wb') as f_: subprocess.run(['dolt', 'sql', '-q', q, '-r', 'parquet'], cwd=db, stdout=f_, stderr=subprocess.PIPE)
        try: pq.ParquetFile(fp).metadata.num_rows
        except Exception:   # this dolt cannot write parquet: CSV, then convert
            with open(f'{pq_dir}/tmp.csv', 'wb') as f_: subprocess.run(['dolt', 'sql', '-q', q, '-r', 'csv'], cwd=db, stdout=f_, check=True)
            pq.write_table(pcsv.read_csv(f'{pq_dir}/tmp.csv'), fp); os.remove(f'{pq_dir}/tmp.csv')
        print(f'{Y}: exported', flush=True)


def view(con, files, last):
    con.execute(f"""CREATE OR REPLACE VIEW o AS SELECT CAST(date AS DATE) d, act_symbol s, CAST(expiration AS DATE) e, CAST(strike AS DOUBLE) k, call_put cp,
                    CAST(bid AS DOUBLE) bid, CAST(ask AS DOUBLE) ask, CAST(vol AS DOUBLE) iv, CAST(delta AS DOUBLE) delta
                    FROM read_parquet({files}) WHERE CAST(date AS DATE) <= DATE '{last}'""")


def universe(con, pq_dir) -> pd.DataFrame:
    view(con, [f'{pq_dir}/opt_{Y}.parquet' for Y in range(2019, 2026)], '2025-12-31')
    SP = con.execute("""WITH a AS (SELECT year(d) y, s, d, median((ask - bid) / ((ask + bid) / 2)) rs FROM o
                        WHERE abs(delta) BETWEEN 0.4 AND 0.6 AND bid > 0 AND ask > bid GROUP BY 1, 2, 3)
                        SELECT y, s, COUNT(*) n_days, median(rs) med_rs FROM a GROUP BY 1, 2""").df()
    SP['full'] = SP['n_days'] >= 0.8 * SP.groupby('y')['n_days'].transform('max')
    SP['in_universe'] = (SP['y'] >= 2020) & SP['full'] & (SP['med_rs'] <= 0.10)
    L = SP[SP['y'] >= 2020].rename(columns={'s': 'underlying', 'y': 'year', 'med_rs': 'med_spread'})[['underlying', 'year', 'n_days', 'med_spread', 'in_universe']]
    L26 = L[(L['year'] == 2025) & L['in_universe']].assign(year=2026, n_days=pd.NA, med_spread=pd.NA)   # 2026: the 2025 list
    return pd.concat([L, L26], ignore_index=True)


def export_chains(con, pq_dir, L, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    S_ = {y: set(g['underlying']) for y, g in L[L['in_universe']].groupby('year')}
    for Y in SPECS['US'].years:
        names = S_.get(Y, set()) | (S_.get(Y - 1, set()) if Y in SPECS['US'].test_years else set())
        view(con, [f'{pq_dir}/opt_{Y}.parquet'], D_END)
        con.register('U', pd.DataFrame({'symbol': sorted(names)}))
        con.execute(f"""COPY (SELECT o.d AS date, o.s AS symbol, o.e AS expiration, o.k AS strike, o.cp AS call_put, o.bid, o.ask, o.iv AS iv_vendor, o.delta
                        FROM o JOIN U ON o.s = U.symbol WHERE year(o.d) = {Y} ORDER BY o.s, o.d, o.e, o.k, o.cp)
                        TO '{out_dir}/chain_{Y}.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE 200000)""")
        print(f'{Y}: chains of {len(names)} names', flush=True)


def rates(path):
    import requests
    t_ = pd.read_csv(io.StringIO(requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3', timeout=60).text))
    t_.columns = ['date', 'dtb3']; t_['date'] = pd.to_datetime(t_['date']); t_['dtb3'] = pd.to_numeric(t_['dtb3'], errors='coerce')
    os.makedirs(os.path.dirname(path), exist_ok=True); t_.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='./data'); ap.add_argument('--snapshot', default=None); ap.add_argument('--work', default='/tmp/ivt_dolt')
    a = ap.parse_args(); L_ = Layout(a.root); raw = L_.raw('US'); os.makedirs(raw, exist_ok=True); os.makedirs(os.path.dirname(L_.in_year_file('US')), exist_ok=True)
    import duckdb
    db = restore(a.work, a.snapshot); pq_dir = f'{a.work}/parquet'
    export_years(db, pq_dir, range(2019, 2027))
    con = duckdb.connect()
    L = universe(con, pq_dir); L.to_parquet(L_.in_year_file('US'), index=False)
    print('universe by year:', L[L['in_universe']].groupby('year').size().to_dict())
    export_chains(con, pq_dir, L, os.path.dirname(L_.chain_file('US', 2020)))
    rates(L_.rates_file('US'))


if __name__ == '__main__':
    main()
