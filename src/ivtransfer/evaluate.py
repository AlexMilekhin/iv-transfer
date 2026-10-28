"""Scoring of the saved models (paper Sections 4.6 and 5, Appendix C).

For every market, test period and fold: the saved models are reloaded and their forecasts recomputed on the CPU in float32. A
reproduction gate checks that each model reproduces its saved test forecasts (99% of samples within 0.1% of the mean forecast and
correlation >= 0.9999); a market that fails is left out. Then:

* forecasts are averaged over seeds and multiplied by an output scale k in {0, 0.1, ..., 1} chosen on the validation year (MAE);
* E = w A + (1 - w) D with w in {0, 0.1, ..., 1} chosen on the validation year;
* the 'chosen' model is the variant (A-E) with the highest primary skill on the validation year;
* skill = 1 - sum |error| / sum |target| (the roll forecasts 0, since the target is net of the roll), in %.

Two views of a market's state: the training view (each year's own universe list, name-years above 100% median ATM vol left out),
which supplies the training / validation samples, and the test view (test year Y: names in year Y-1's list; the vol rule uses the
previous year's median). Scores: 'primary' = forecast at t against the target starting at the next grid date, on well-measured
name-years; 'same-day' = target starting at t, all names.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as _norm

from .config import ALL17, MAX_NAME_YEAR_IV, TrainConfig, Layout
from .model import MoE, Norm, predict
from .train import masks as _masks, model_inputs

CPU = torch.device('cpu')
SCORES = {'primary': (1, True), 'same-day': (0, False)}     # (delay in grid dates, well-measured names only)
VARS = ['A', 'B', 'C', 'D']
PERIODS = {'2023-25': [2023, 2024, 2025], '2026': [2026]}


def masks(D, Y):
    return _masks(D, Y)[:3]


# ======================= data views =======================
def quality_years(D) -> set:
    """Well-measured name-years: (name, Y) if in Y-1 the name's daily ATM vol changes (shortest maturity) had a noise sd below the
    true-move sd; noise variance = minus the first-order autocovariance of the changes (paper Section 4.6)."""
    iv = D['X'][:, -1, D['feat'].index('ATM_IV')]
    Q = pd.DataFrame({'tk': D['tk'], 'd': np.asarray(D['asof']), 'tT': np.round(D['tT'], 3), 'iv': iv})
    Q = Q[Q.tT == Q.tT.min()].sort_values(['tk', 'd']); Q['yr'] = pd.DatetimeIndex(Q['d']).year; Q['dv'] = Q.groupby('tk').iv.diff()
    def est(g):
        x = g.dv.dropna().to_numpy()
        if len(x) < 61: return pd.Series({'noise': np.nan, 'true': np.nan})
        c1 = np.mean((x[1:] - x.mean()) * (x[:-1] - x.mean())); nv = max(-c1, 0); tv = max(x.var() - 2 * nv, 0)
        return pd.Series({'noise': np.sqrt(nv), 'true': np.sqrt(tv)})
    E = Q.groupby(['tk', 'yr']).apply(est, include_groups=False).reset_index(); E['ok'] = E.noise < E.true; E['yr'] += 1
    return set(map(tuple, E.loc[E.ok, ['tk', 'yr']].to_numpy()))


def load_view(layout: Layout, m: str, cfg: TrainConfig, view: str, with_gb=False) -> dict:
    """view 'train': own-year vol rule, ex-ante test additions left out (= what the models were trained and validated on).
    view 'test': previous-year vol rule, every unit kept (test names are selected per fold from the universe file)."""
    st = layout.state_dir(m); A_ = np.load(f'{st}/arrays.npz', allow_pickle=True); assert A_['X'].shape[2] == 17
    mt = pd.read_parquet(f'{st}/meta.parquet').reset_index(drop=True)
    cap = cfg.max_name_year_iv if cfg.max_name_year_iv is not None else MAX_NAME_YEAR_IV
    ny = pd.DataFrame({'tk': mt['ticker'].astype(str), 'yr': pd.to_datetime(mt['asof']).dt.year, 'iv': A_['X'][:, -1, ALL17.index('ATM_IV')]})
    if view == 'train':
        keep = (ny.groupby(['tk', 'yr'])['iv'].transform('median') <= cap).to_numpy()
    else:   # a name-year is dropped if the name's median ATM vol in the PREVIOUS year was above the cap (no previous year: kept)
        med = ny.groupby(['tk', 'yr'])['iv'].median(); prev = med.reindex(pd.MultiIndex.from_arrays([ny['tk'], ny['yr'] - 1])).to_numpy()
        keep = ~(prev > cap)
    dropped = ny[~keep].groupby(['tk', 'yr']).size()
    if view == 'train':
        U_ = pd.read_parquet(layout.universe_file(m)); add_ = set(zip(U_.loc[U_['added'], 'name'].astype(str), U_.loc[U_['added'], 'year'].astype(int)))
        keep = keep & ~np.array([u in add_ for u in zip(ny['tk'], ny['yr'])], bool)
    mt = mt[keep].reset_index(drop=True); X17 = A_['X'][keep].astype(np.float32); yk = A_['y'][keep].astype(np.float64)
    D = dict(X=model_inputs(X17, mt, yk, cfg.features), feat=list(cfg.features), y=yk, tk=mt['ticker'].astype(str).to_numpy(),
             asof=pd.to_datetime(mt['asof']).astype('datetime64[ns]'), afut=pd.to_datetime(mt['asof_future']).astype('datetime64[ns]'),
             tT=mt['target_T'].to_numpy(float), mech=mt['mech'].to_numpy(float), dropped={f'{t} {y}': int(n) for (t, y), n in dropped.items()})
    D['same'] = mt['same'].to_numpy(bool) if 'same' in mt else np.zeros(len(mt), bool)
    if with_gb: D['X17'] = X17; D['w'] = A_['w'][keep].astype(np.float64)
    a_ = np.asarray(D['asof']); K = pd.DataFrame({'i': np.arange(len(yk)), 'tk': D['tk'], 'd': a_, 'tT': np.round(D['tT'], 3)}).sort_values(['tk', 'tT', 'd'])
    D['nx'] = K.groupby(['tk', 'tT'])['i'].shift(-1).reindex(range(len(K))).to_numpy(); D['a'] = a_; D['qy'] = quality_years(D)
    return D


def picked(D, idx, Yv, L, qual):
    """Samples scored and the index of their scored target. L = 1: the target starting at the next grid date (at most 4 calendar
    days later); L = 0: the same date. The scored target must have a defined roll and lie in year Yv; qual: well-measured names only."""
    if L: j = D['nx'][idx]; ok = np.isfinite(j); j = np.nan_to_num(j, nan=0).astype(int)
    else: j = idx.copy(); ok = np.ones(len(idx), bool)
    ok &= ((D['a'][j] - D['a'][idx]) / np.timedelta64(1, 'D') <= 3 * L + 1) & (D['mech'][j] != 0) & (pd.DatetimeIndex(D['a'][j]).year == Yv)
    if qual: ok &= np.array([(t, Yv) in D['qy'] for t in D['tk'][idx]], bool)
    return ok, j


def nwt(x, L=6):
    """t statistic of the mean with a Newey-West (Bartlett, L lags) variance."""
    x = np.asarray(x, float); n = len(x); e = x - x.mean(); s = e @ e / n
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n)


def gb_fold(DO, DX, TR, VA, it_x, log=print):
    """Gradient boosting (paper Section 4.4) on the fold's training block: the 17 inputs of the 10-date window, squared error,
    number of trees and output scale chosen on the validation year (MAE)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    TREES = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 600]; KS1 = np.round(np.arange(0, 1.01, 0.1), 2)
    fl = lambda D_, ix: D_['X17'][ix].reshape(len(ix), -1)
    itr, iva = np.where(TR)[0], np.where(VA)[0]; Xtr, Xva = fl(DO, itr), fl(DO, iva); ytr, yva = DO['y'][itr], DO['y'][iva]
    gb = HistGradientBoostingRegressor(max_iter=10, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
                                       early_stopping=False, warm_start=True, random_state=0)
    best, worse = (np.inf, None, None, 0), 0
    for n_it in TREES:
        gb.set_params(max_iter=n_it); gb.fit(Xtr, ytr, sample_weight=DO['w'][itr]); pv = gb.predict(Xva); m_ = np.mean(np.abs(pv - yva))
        if m_ < best[0]: best, worse = (m_, pv, gb.predict(fl(DX, it_x)), n_it), 0
        else:
            worse += 1
            if worse >= 3: break
    _, pv, px, nb = best; k = float(KS1[np.argmin([np.mean(np.abs(k_ * pv - yva)) for k_ in KS1])])
    return dict(gb_va=pd.Series(k * pv, index=iva), gb_te=k * px, trees=nb, k=k)


# ======================= evaluation =======================
class Evaluation:
    """Forecasts of one run folder for every market, period and fold, and the scores built from them."""

    def __init__(self, layout: Layout, run_name: str, cfg: TrainConfig, markets, seeds, periods=None, test_years=None, with_gb=False, log=print):
        self.layout, self.run, self.cfg, self.seeds, self.log = layout, layout.run_dir(run_name), cfg, list(seeds), log
        self.markets, self.periods, self.with_gb = list(markets), dict(periods or PERIODS), with_gb
        self.test_years = test_years or {m: [Y for f in self.periods.values() for Y in f] for m in self.markets}
        self.KS = np.round(np.arange(0, cfg.k_cap + 1e-9, 0.1), 2); self.WS = np.round(np.arange(0, 1 + 1e-9, 0.1), 2)
        self.FC, self.DO, self.DX, self.gate_ok, self.gate_rows, self.gb_rows = {}, {}, {}, {}, [], []
        torch.set_num_threads(os.cpu_count() or 1)

    # ---- forecasts ----
    def _load_model(self, name, us=False):
        m_ = MoE(self.cfg); s_ = torch.load(f'{self.run}/models/{name}.pt', map_location='cpu', weights_only=False)
        m_.load_state_dict(s_['state'] if us else s_); return m_.to(CPU)

    def forecasts_market(self, m):
        ok_all = True; cfg = self.cfg
        DO = self.DO[m] = load_view(self.layout, m, cfg, 'train', self.with_gb)
        DX = self.DX[m] = load_view(self.layout, m, cfg, 'test', self.with_gb)
        U_ = pd.read_parquet(self.layout.universe_file(m))
        test_names = {int(y_): set(g['name'].astype(str)) for y_, g in U_[U_['xa_test']].groupby('year')}
        for per, folds in self.periods.items():
            for Y in folds:
                if Y not in self.test_years[m]: continue
                names = [f'US_A_{Y}_s{s}' for s in self.seeds] if m == 'US' else [f'{m}_{v}_{Y}_s{s}' for v in VARS for s in self.seeds]
                if not all(os.path.exists(f'{self.run}/pred/{n}.parquet') for n in names): self.log(f'  {m} {Y}: saved forecasts incomplete - fold skipped'); continue
                TR, VA, TE = masks(DO, Y); nB = Norm(cfg).fit(DO['X'][TR]); sy = float(np.std(DO['y'][TR]))
                Xte_o, Gte_o = nB.tr(DO['X'][TE]); it_o = np.where(TE)[0]; iv_ = np.where(VA)[0]; Xva_, Gva_ = nB.tr(DO['X'][VA])
                TX = (pd.DatetimeIndex(DX['asof']).year == Y) & np.isin(DX['tk'], list(test_names[Y])); it_x = np.where(TX)[0]; Xte_x, Gte_x = nB.tr(DX['X'][TX])
                rec = dict(va={}, te={}, it=it_x)
                for v in (['A'] if m == 'US' else VARS):
                    for s in self.seeds:
                        nm = f'{m}_{v}_{Y}_s{s}' if m != 'US' else f'US_A_{Y}_s{s}'
                        P = pd.read_parquet(f'{self.run}/pred/{nm}.parquet').groupby(['part', 'i'])['pred'].mean(); te_saved = P.loc['test']
                        if m == 'US':
                            mdl = self._load_model(f'US_{Y}_s{s}', us=True); sy_ = float(torch.load(f'{self.run}/models/US_{Y}_s{s}.pt', map_location='cpu', weights_only=False)['sy'])
                            assert abs(sy_ - sy) < 1e-9 * max(sy, 1e-12) + 1e-12, 'US target scale differs from the saved one'
                        elif v == 'B': mdl = self._load_model(f'US_{Y}_s{s}', us=True); sy_ = sy
                        else: mdl = self._load_model(nm); sy_ = sy
                        po = predict(mdl, Xte_o, Gte_o) * sy_      # reproduction gate on the samples the saved forecasts were made for
                        so = te_saved.reindex(it_o).to_numpy(); fin_ = np.isfinite(so); r_ = np.abs(po - so)[fin_] / max(np.nanmean(np.abs(so)), 1e-12)
                        cc_ = float(np.corrcoef(po[fin_], so[fin_])[0, 1]) if fin_.sum() > 2 else np.nan
                        self.gate_rows.append({'market': m, 'period': per, 'fold': Y, 'variant': v, 'seed': s, 'test rows': len(it_o), 'saved rows matched': int(fin_.sum()),
                                               'median |diff| / mean |forecast|': float(np.median(r_)) if len(r_) else np.nan,
                                               'p99 |diff| / mean |forecast|': float(np.percentile(r_, 99)) if len(r_) else np.nan,
                                               'max |diff| / mean |forecast|': float(r_.max()) if len(r_) else np.nan, 'corr': cc_})
                        if not (fin_.all() and np.percentile(r_, 99) <= 1e-3 and cc_ >= 0.9999): ok_all = False
                        rec['va'][(v, s)] = pd.Series(predict(mdl, Xva_, Gva_) * sy_, index=iv_); rec['te'][(v, s)] = predict(mdl, Xte_x, Gte_x) * sy_
                if self.with_gb and m != 'US':
                    g_ = gb_fold(DO, DX, TR, VA, it_x); rec |= {'gb_va': g_['gb_va'], 'gb_te': g_['gb_te']}
                    self.gb_rows.append({'fold': f'{m} {Y}', 'trees': g_['trees'], 'k': g_['k']})
                self.FC[(m, per, Y)] = rec
        return ok_all

    def run_forecasts(self):
        for m in self.markets:
            t0 = time.time(); self.gate_ok[m] = self.forecasts_market(m)
            self.log(f'[forecasts {m}] gate {"PASSED" if self.gate_ok[m] else "FAILED (left out of the tables)"} ({(time.time() - t0) / 60:.1f} min)')
        return pd.DataFrame(self.gate_rows)

    # ---- combinations and scores ----
    def kfit(self, f, y):
        return self.KS[np.argmin([np.mean(np.abs(k * f - y)) for k in self.KS])]

    def combine(self, m, per, Y, seeds=None):
        """Seed average, output scale on validation, E = w A + (1 - w) D. -> (validation forecasts, test forecasts) per model."""
        seeds = self.seeds if seeds is None else seeds
        rec = self.FC[(m, per, Y)]; y = self.DO[m]['y']; Fv, Ft = {}, {}
        for v in (['A'] if m == 'US' else VARS):
            va = pd.concat([rec['va'][(v, s)] for s in seeds], axis=1).mean(axis=1)
            te = np.mean([rec['te'][(v, s)] for s in seeds], axis=0)
            k = self.kfit(va.to_numpy(), y[va.index.to_numpy()]); Fv[v], Ft[v] = k * va, k * te
        if m != 'US':
            iv = Fv['A'].index; Fv = {v: Fv[v].reindex(iv) for v in Fv}; yv = y[iv.to_numpy()]
            w = self.WS[int(np.argmin([np.mean(np.abs(w * Fv['A'].to_numpy() + (1 - w) * Fv['D'].to_numpy() - yv)) for w in self.WS]))]
            Fv['E'] = w * Fv['A'] + (1 - w) * Fv['D']; Ft['E'] = w * Ft['A'] + (1 - w) * Ft['D']
            if 'gb_va' in rec: Fv['GB'] = rec['gb_va'].reindex(iv); Ft['GB'] = rec['gb_te']
        return Fv, Ft

    def choose(self, m, Y, Fv, score='primary', models=None):
        """Model with the highest validation skill (validation year Y - 1) on the given score."""
        DO = self.DO[m]; models = models or [v for v in ('A', 'B', 'C', 'D', 'E')]
        iv = Fv['A'].index.to_numpy(); ok, j = picked(DO, iv, Y - 1, *SCORES[score]); b = np.abs(DO['y'][j[ok]]).sum()
        vsk = {v: 1 - np.abs(Fv[v].reindex(Fv['A'].index).to_numpy()[ok] - DO['y'][j[ok]]).sum() / b for v in models}
        return max(vsk, key=vsk.get)

    def score_all(self, seeds=None, scores=SCORES):
        """-> skill (%) and sample count per (market, period, score, model), the validation choices, and the daily loss differentials vs A."""
        seeds = self.seeds if seeds is None else seeds
        acc, pick, dm = {}, {}, {}
        for (m, per, Y), rec in self.FC.items():
            if not self.gate_ok.get(m, False): continue
            vs_ = ['A'] if m == 'US' else ['A', 'B', 'C', 'D', 'E']
            Fv, Ft = self.combine(m, per, Y, seeds); D = self.DX[m]; it = rec['it']; ch = {}
            if m != 'US':
                for sc in SCORES: ch[f'chosen on val {sc}'] = self.choose(m, Y, Fv, sc, vs_)
                pick[(m, per, Y)] = ch
            vs_s = vs_ + (['GB'] if 'GB' in Ft else [])
            for sc in scores:
                ok, j = picked(D, it, Y, *scores[sc]); yt = D['y'][j[ok]]; d_ = D['a'][j[ok]]
                for name, v in [(v, v) for v in vs_s] + list(ch.items()):
                    e = np.abs(Ft[v][ok] - yt); k_ = (m, per, sc, name); s0 = acc.get(k_, (0.0, 0.0, 0)); acc[k_] = (s0[0] + e.sum(), s0[1] + np.abs(yt).sum(), s0[2] + len(yt))
                    if name in 'BCDE' and len(name) == 1:
                        dm.setdefault((m, per, sc, name), []).append(pd.Series(e - np.abs(Ft['A'][ok] - yt)).groupby(d_).mean())
        sk = {k: 100 * (1 - e / b) for k, (e, b, n) in acc.items()}; ns = {k: n for k, (e, b, n) in acc.items()}
        return sk, ns, pick, dm

    def tables(self):
        """Table 6 (gains), Table 7 (validation choice), the best variant across leave-one-seed-out ensembles, Diebold-Mariano
        statistics (C1), the 2026 family with Benjamini-Hochberg at 10% (C2) and the US model at home."""
        SK, NS, PICK, DM = self.score_all(); self.SK, self.NS, self.PICK, self.DM = SK, NS, PICK, DM
        self.LOO = [self.score_all([s for s in self.seeds if s != x])[0] for x in self.seeds] if len(self.seeds) > 1 else []
        gain = lambda R, m, per, sc, v: R.get((m, per, sc, v), np.nan) - R.get((m, per, sc, 'A'), np.nan)
        T6, T7, T9, C1 = [], [], [], []
        for m in [m for m in self.markets if m != 'US' and self.gate_ok.get(m)]:
            for per, folds in self.periods.items():
                if (m, per, 'primary', 'A') not in SK: continue
                for sc in SCORES:
                    r = {'market': m, 'period': per, 'score': sc, 'samples': NS[(m, per, sc, 'A')], 'local A %': round(SK[(m, per, sc, 'A')], 2)}
                    r |= {f'{v} gain': round(gain(SK, m, per, sc, v), 2) for v in 'BCDE'}
                    if (m, per, sc, 'GB') in SK: r['GB gain'] = round(gain(SK, m, per, sc, 'GB'), 2)
                    T6.append(r)
                    for c_ in ('chosen on val primary', 'chosen on val same-day'):
                        T7.append({'market': m, 'period': per, 'test score': sc, 'selection': c_,
                                   'picks by fold': ' '.join(PICK[(m, per, Y)][c_] for Y in folds if (m, per, Y) in PICK), 'gain over A': round(gain(SK, m, per, sc, c_), 2)})
                    for v in 'BCDE':
                        s_ = pd.concat(DM[(m, per, sc, v)]).groupby(level=0).mean().sort_index()
                        C1.append({'market': m, 'period': per, 'score': sc, 'variant': v, 'days': len(s_), 't vs A': round(nwt(s_.values), 2)})
                best = max('BCDE', key=lambda v: gain(SK, m, per, 'primary', v)); g_ = [gain(R, m, per, 'primary', best) for R in self.LOO] or [gain(SK, m, per, 'primary', best)]
                T9.append({'market': m, 'period': per, 'best variant (primary)': best, 'all-seed gain': round(gain(SK, m, per, 'primary', best), 2),
                           'leave-one-seed-out': f'{min(g_):+.2f} .. {max(g_):+.2f}'})
        T6, T7, T9, C1 = (pd.DataFrame(t) for t in (T6, T7, T9, C1))
        C2 = C1[(C1.period == '2026') & (C1.score == 'primary')].copy() if len(C1) else pd.DataFrame(columns=['t vs A'])
        if len(C2):   # Benjamini-Hochberg at 10% on one-sided p values (negative t = lower error than A)
            c_ = C2.copy(); c_['p (one-sided)'] = _norm.cdf(c_['t vs A'].to_numpy(float)); c_ = c_.sort_values('p (one-sided)'); n_ = len(c_)
            thr = 0.10 * (np.arange(n_) + 1) / max(n_, 1); kmax = max([i + 1 for i in range(n_) if c_['p (one-sided)'].iloc[i] <= thr[i]], default=0)
            C2.loc[c_.index, 'p (one-sided)'] = c_['p (one-sided)']; C2.loc[c_.index, 'BH threshold'] = thr; C2.loc[c_.index, 'passes BH 10%'] = np.arange(n_) < kmax
        USH = []
        for per, folds in self.periods.items():
            for Y in folds:
                if ('US', per, Y) not in self.FC or not self.gate_ok.get('US'): continue
                Fv, Ft = self.combine('US', per, Y); D = self.DX['US']; it = self.FC[('US', per, Y)]['it']; ok, j = picked(D, it, Y, *SCORES['same-day']); yt = D['y'][j[ok]]
                USH.append({'year': Y, 'samples': int(ok.sum()), 'names': len(set(D['tk'][it])), 'US skill at home, same-day %': round(100 * (1 - np.abs(Ft['A'][ok] - yt).sum() / np.abs(yt).sum()), 2)})
        self.T6, self.T7, self.T9, self.C1, self.C2, self.USH = T6, T7, T9, C1, C2, pd.DataFrame(USH)
        return dict(table6_gains=T6, table7_chosen=T7, table9_seeds=T9, c1_dm=C1, c2_2026_bh=C2, us_home=self.USH)

    def best_variant(self, m, per):
        t_ = self.T9[(self.T9.market == m) & (self.T9.period == per)]
        return t_['best variant (primary)'].iloc[0] if len(t_) else None
