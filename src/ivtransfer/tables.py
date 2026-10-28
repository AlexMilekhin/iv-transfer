"""Every table and reported number of the paper, from an ``Evaluation`` of the final models (and one of the initial network).

Each function writes its CSV to ``out`` and records the numbers quoted in the text in a ``Numbers`` registry (section, item, market,
period, value, how), saved as paper_numbers.csv. Bootstrap draws use fixed seeds, so every interval is reproducible.
"""
from __future__ import annotations

import copy
import glob
import json
import os

import numpy as np
import pandas as pd

from .config import Layout, TrainConfig
from .evaluate import SCORES, Evaluation, masks, picked
from .model import MoE, Norm, predict
from .train import load_market

N_BOOT = 2000
sk = lambda e, y: 100 * (1 - np.sum(e) / np.sum(y))
skill = lambda e, y: 100 * (1 - np.abs(e).sum() / np.abs(y).sum())


class Numbers:
    def __init__(self): self.rows = []
    def __call__(self, section, item, market, period, value, how=''):
        self.rows.append({'section': section, 'item': item, 'market': market, 'period': period, 'value': value, 'how': how})
    def save(self, path): pd.DataFrame(self.rows).to_csv(path, index=False)


def _tm(ev): return [m for m in ev.markets if m != 'US']


def _view(ev, m, per, Y):
    rec = ev.FC[(m, per, Y)]; return rec, ev.DX[m], rec['it']


# ======================= Section 3 / Table 1 =======================
def table1(ev: Evaluation, out, note):
    """Names and samples scored per market and test year (same-day set: every name with a defined roll)."""
    rows = []
    for m in ev.markets:
        for per, folds in ev.periods.items():
            for Y in folds:
                if (m, per, Y) not in ev.FC: continue
                rec, D, it = _view(ev, m, per, Y); ok, j = picked(D, it, Y, *SCORES['same-day']); okp, jp = picked(D, it, Y, *SCORES['primary'])
                rows.append({'market': m, 'year': Y, 'names scored': len(set(D['tk'][it[ok]])), 'same-day samples': int(ok.sum()),
                             'primary samples': int(okp.sum()), 'names in the test list': len(set(D['tk'][it]))})
    T1 = pd.DataFrame(rows); T1.to_csv(f'{out}/table1_counts.csv', index=False)
    for m, g in T1.groupby('market'):
        d_ = g[g.year <= 2025]
        note('Table 1', 'underlyings per test year 2023-25 (min-max)', m, '2023-25', f'{d_["names scored"].min()}-{d_["names scored"].max()}', 'names with >= 1 scored same-day sample')
        note('Table 1', 'test samples 2023-25', m, '2023-25', int(d_['same-day samples'].sum()), 'same-day set (roll defined)')
        g26 = g[g.year == 2026]
        if len(g26): note('Table 1', '2026 underlyings / samples', m, '2026', f'{int(g26["names scored"].iloc[0])} / {int(g26["same-day samples"].iloc[0]):,}', '')
    DO = ev.DO['US']; k_ = pd.DatetimeIndex(DO['asof']).year <= 2025
    note('3.1', 'US samples 2020-25 / underlyings', 'US', '2020-25', f'{int(k_.sum()):,} / {len(set(DO["tk"][k_]))}', 'in-year units, after the volatility rule')
    for m in ev.markets:
        TR, VA, TE = masks(ev.DO[m], 2025)
        note('Intro', 'training samples of the 2025 model', m, '2025 fold', f'{int(TR.sum()):,}', 'targets <= 2023, names with >= 50 samples')
    return T1


def top1_share(ev, out, note):
    """Share of the total absolute target carried by the largest 1% of samples (Section 4.2)."""
    rows = []
    for m in ev.markets:
        for per, folds in ev.periods.items():
            for Y in folds:
                if (m, per, Y) not in ev.FC: continue
                rec, D, it = _view(ev, m, per, Y); ok, j = picked(D, it, Y, *SCORES['same-day']); a_ = np.sort(np.abs(D['y'][j[ok]]))[::-1]
                if len(a_): rows.append({'market': m, 'year': Y, 'top 1% share %': round(100 * a_[:max(1, len(a_) // 100)].sum() / a_.sum(), 1)})
    T_ = pd.DataFrame(rows); T_.to_csv(f'{out}/top1_share.csv', index=False)
    note('4.2', 'share of total |target| from the largest 1% of samples (range over markets and years)', 'all', '2023-26', f'{T_["top 1% share %"].min()}-{T_["top 1% share %"].max()}%', 'same-day test sets')
    return T_


def data_quality(ev, layout: Layout, out, note):
    """Well-measured share of test name-years (Section 4.6); filled input windows and the roll check from the state builder."""
    rows = []
    for m in ev.markets:
        D = ev.DX[m]; ny = {u for u in set(zip(D['tk'], pd.DatetimeIndex(D['asof']).year)) if u[1] in ev.test_years[m]}
        rows.append({'market': m, 'test name-years': len(ny), 'well-measured %': round(100 * len(ny & D['qy']) / max(len(ny), 1), 1)})
    Q = pd.DataFrame(rows); Q.to_csv(f'{out}/well_measured.csv', index=False)
    for r in rows: note('4.6 / 6.3', 'well-measured share of test name-years', r['market'], 'test years', f'{r["well-measured %"]}%', 'previous-year noise estimator')
    for m in ev.markets:
        p = f'{layout.state_dir(m)}/state_stats.json'
        if os.path.exists(p):
            st = json.load(open(p))
            note('3.4 / 4.1', 'windows with filled dates', m, 'all years', f"{st['windows with filled dates %']}% (1 date {st['one date filled %']}%, 2 dates {st['two dates filled %']}%)", 'state builder')
            note('3.4 / 4.1', 'ex-ante roll vs no-change, samples <= 2025', m, 'all years', f"skill {st['roll vs no-change skill, samples <= 2025 %']:+.2f}%, sd ratio {st['sd net of roll / sd of full change, samples <= 2025']}", 'state builder')
    return Q


# ======================= Sections 6-7 =======================
def retention(ev, out, note, per='2023-25'):
    """Validation skill at t+1 relative to t+0, all names, after the output scale (Section 6.3)."""
    rows = []
    for m in ev.markets:
        acc = {}
        for Y in ev.periods[per]:
            if (m, per, Y) not in ev.FC: continue
            Fv, Ft = ev.combine(m, per, Y); DO = ev.DO[m]
            for v, s_ in Fv.items():
                iv = s_.index.to_numpy(); f_ = s_.to_numpy()
                for L in (0, 1):
                    ok, j = picked(DO, iv, Y - 1, L, False); e_ = f_[ok] - DO['y'][j[ok]]; a0 = acc.setdefault((v, L), [0.0, 0.0])
                    a0[0] += np.abs(e_).sum(); a0[1] += np.abs(DO['y'][j[ok]]).sum()
        for v in sorted({k[0] for k in acc}):
            s0 = 100 * (1 - acc[(v, 0)][0] / acc[(v, 0)][1]); s1 = 100 * (1 - acc[(v, 1)][0] / acc[(v, 1)][1])
            rows.append({'market': m, 'variant': v, 'validation skill t+0 %': round(s0, 2), 't+1 %': round(s1, 2), 'retention %': round(100 * s1 / s0, 0) if s0 > 0 else np.nan})
    R = pd.DataFrame(rows); R.to_csv(f'{out}/retention.csv', index=False)
    for r in rows: note('6.3', 'retention ratio (validation years 2022-24)', r['market'], r['variant'], r['retention %'], f't+0 {r["validation skill t+0 %"]}, t+1 {r["t+1 %"]}')
    return R


def output_scales(ev, out, note):
    """Output scales k chosen on validation, per market, fold and variant (Section 6.4)."""
    rows = []
    for (m, per, Y), rec in ev.FC.items():
        y = ev.DO[m]['y']
        for v in (['A'] if m == 'US' else ['A', 'B', 'C', 'D']):
            va = pd.concat([rec['va'][(v, s)] for s in ev.seeds], axis=1).mean(axis=1)
            rows.append({'market': m, 'fold': Y, 'variant': v, 'k1': float(ev.kfit(va.to_numpy(), y[va.index.to_numpy()]))})
    K_ = pd.DataFrame(rows); K_.to_csv(f'{out}/output_scales.csv', index=False)
    for m, g in K_[(K_.variant == 'B') & (K_.fold <= 2025)].groupby('market'): note('6.4', 'output scale k* of the zero-shot model, folds 2023-25 (min-max)', m, '2023-25', f'{g.k1.min():.1f}-{g.k1.max():.1f}', '')
    for m, g in K_[(K_.variant == 'B') & (K_.fold == 2026)].groupby('market'): note('6.4', 'output scale k* of the zero-shot model, 2026', m, '2026', f'{g.k1.iloc[0]:.1f}', '')
    return K_


def diagnostics_2026(ev, out, note):
    """2026 skill by month, target drift 2025 -> 2026, and the names behind Canada's local shortfall (Sections 6.4 and 7.6)."""
    rows = []
    for m in ev.markets:
        if (m, '2026', 2026) not in ev.FC: continue
        Fv, Ft = ev.combine(m, '2026', 2026); rec, D, it = _view(ev, m, '2026', 2026)
        for sc in SCORES:
            ok, j = picked(D, it, 2026, *SCORES[sc]); yt = D['y'][j[ok]]; mon = pd.DatetimeIndex(D['a'][it[ok]]).month
            for v, f_ in Ft.items():
                e_ = f_[ok] - yt
                for mo in sorted(set(mon)):
                    k_ = mon == mo; rows.append({'market': m, 'score': sc, 'variant': v, 'month': int(mo), 'samples': int(k_.sum()), 'skill %': round(skill(e_[k_], yt[k_]), 1)})
    M_ = pd.DataFrame(rows); M_.to_csv(f'{out}/skill_2026_by_month.csv', index=False)
    for m in ev.markets:
        g = M_[(M_.market == m) & (M_.score == 'primary')] if len(M_) else M_
        if not len(g): continue
        mar = g[g.month == 3]['skill %']; aa = g[g.month.between(4, 8)].groupby('variant')['skill %'].mean()
        if len(mar): note('7.6', 'March 2026 skill, all models (min to max)', m, '2026', f'{mar.min()} to {mar.max()}', 'primary score')
        if len(aa): note('7.6', 'April-August 2026 mean monthly skill, all models (min to max)', m, '2026', f'{aa.min():.1f} to {aa.max():.1f}', 'primary score')
    if len(M_) and 'BR' in set(M_.market):
        g = M_[(M_.market == 'BR') & (M_.score == 'primary') & (M_.month.isin([8, 9]))]
        for v in ('A', 'B'):
            gv = g[g.variant == v]
            if len(gv): note('6.4', 'Brazil August-September 2026 skill', 'BR', '2026', ' / '.join(f'{int(r.month)}: {r["skill %"]}' for _, r in gv.iterrows()), f'variant {v}, primary')
    drift = []
    for m in ev.markets:
        for Y, per in ((2025, '2023-25'), (2026, '2026')):
            if (m, per, Y) not in ev.FC: continue
            rec, D, it = _view(ev, m, per, Y); ok, j = picked(D, it, Y, *SCORES['same-day']); yt = D['y'][j[ok]]
            drift.append({'market': m, 'year': Y, 'mean target': float(np.mean(yt)), 'mean |target|': float(np.mean(np.abs(yt)))})
    Dr = pd.DataFrame(drift); Dr.to_csv(f'{out}/target_drift.csv', index=False)
    for m, g in Dr.groupby('market'):
        if set(g.year) == {2025, 2026}:
            a5, a6 = g[g.year == 2025].iloc[0], g[g.year == 2026].iloc[0]
            note('7.6', 'mean |target| 2026 vs 2025', m, '2026', f'{100 * (a6["mean |target|"] / a5["mean |target|"] - 1):+.0f}%', f'mean target 2026 {a6["mean target"]:+.2e}')
    if ('CA', '2026', 2026) in ev.FC:
        Fv, Ft = ev.combine('CA', '2026', 2026); rec, D, it = _view(ev, 'CA', '2026', 2026); ok, j = picked(D, it, 2026, *SCORES['primary'])
        yt = D['y'][j[ok]]; c_ = pd.Series(np.abs(Ft['A'][ok] - yt) - np.abs(yt)).groupby(D['tk'][it[ok]]).sum().sort_values(ascending=False)
        tot = c_.sum(); top = c_.head(5)
        note('7.6', "Canada 2026: share of the local model's shortfall vs the roll from its top 5 names", 'CA', '2026', f'{100 * top.sum() / tot:.0f}%', ', '.join(top.index))
    return M_, Dr


def robustness_items(ev, out, note):
    """Leave-one-seed-out ensembles (7.3), same-day model choice (7.4), gradient boosting (7.2), the volatility rule (3.2), listing check (7.7)."""
    rows = []
    for m in _tm(ev):
        for per in ev.periods:
            g_ = {v: [R.get((m, per, 'primary', v), np.nan) - R.get((m, per, 'primary', 'A'), np.nan) for R in ev.LOO] for v in 'BCDE'}
            rows.append({'market': m, 'period': per} | {f'{v} min over LOO': round(min(x), 2) for v, x in g_.items()} | {f'{v} max over LOO': round(max(x), 2) for v, x in g_.items()})
            note('7.3', 'US-informed variants above A in every leave-one-seed-out ensemble', m, per, ' '.join(v for v, x in g_.items() if min(x) > 0) or 'none',
                 f'{len(ev.seeds)} ensembles of {len(ev.seeds) - 1} seeds')
    L_ = pd.DataFrame(rows); L_.to_csv(f'{out}/loo_by_variant.csv', index=False)
    for _, r in ev.T7[(ev.T7['test score'] == 'primary') & (ev.T7.selection == 'chosen on val same-day')].iterrows():
        note('7.4', 'gain of the model chosen on same-day validation, primary test score', r['market'], r['period'], f'{r["gain over A"]:+.2f}', f'picks {r["picks by fold"]}')
    for _, r in ev.T6[ev.T6.score == 'same-day'].iterrows():
        if 'GB gain' not in r: continue
        best = max('BCDE', key=lambda v: r[f'{v} gain'])
        note('7.2', 'GB vs local A, same-day (gain, points)', r['market'], r['period'], f'{r["GB gain"]:+.2f}',
             f'A skill {r["local A %"]}; best transfer {best} {r[f"{best} gain"]:+.2f} (skill {r["local A %"] + r[f"{best} gain"]:.2f})')
    vr = []
    for m in ev.markets:
        DO, DX = ev.DO[m], ev.DX[m]
        vr.append({'market': m, 'training / validation years (own-year median)': ', '.join(k_ for k_ in DO['dropped'] if int(k_.split()[-1]) <= 2025) or 'none',
                   'test years (previous-year median)': ', '.join(k_ for k_ in DX['dropped'] if int(k_.split()[-1]) in ev.test_years[m]) or 'none'})
    V_ = pd.DataFrame(vr); V_.to_csv(f'{out}/volatility_rule.csv', index=False)
    for r in vr: note('3.2', 'name-years removed by the 100% volatility rule', r['market'], 'all', f'training: {r["training / validation years (own-year median)"]}; test: {r["test years (previous-year median)"]}', '')
    listing_check(ev, note, ev.seeds, 'final network')
    return L_, V_


def listing_check(ev, note, seeds, label):
    """US model at home: skill on samples whose bracketing expiries at t+h are those assumed by the roll, against all samples (7.7)."""
    for per, folds in ev.periods.items():
        acc = {'all': [0, 0, 0], 'same': [0, 0, 0]}
        for Y in folds:
            if ('US', per, Y) not in ev.FC: continue
            Fv, Ft = ev.combine('US', per, Y, seeds); rec, D, it = _view(ev, 'US', per, Y); ok, j = picked(D, it, Y, *SCORES['same-day']); yt = D['y'][j[ok]]; e_ = Ft['A'][ok] - yt
            for k_, mk_ in (('all', np.ones(len(yt), bool)), ('same', D['same'][it[ok]])):
                acc[k_][0] += np.abs(e_[mk_]).sum(); acc[k_][1] += np.abs(yt[mk_]).sum(); acc[k_][2] += int(mk_.sum())
        if acc['all'][1] > 0:
            s_all, s_same = 100 * (1 - acc['all'][0] / acc['all'][1]), 100 * (1 - acc['same'][0] / max(acc['same'][1], 1e-12))
            note('7.7', f'US model at home: skill on samples with the same bracketing expiries vs all with a roll ({label})', 'US', per, f'{s_same:+.1f} vs {s_all:+.1f}',
                 f'{100 - 100 * acc["same"][2] / acc["all"][2]:.0f}% of samples change expiries')


# ======================= bootstrap (Section 5, Appendix C) =======================
def _blocks(index, blk):
    mon = index.to_period('M'); ms = sorted(set(mon)); grp = {mm: i // blk for i, mm in enumerate(ms)}
    return np.array([grp[mm] for mm in mon])


def day_sums(ev, m, per, sc='primary'):
    parts = []
    for Y in ev.periods[per]:
        if (m, per, Y) not in ev.FC: continue
        Fv, Ft = ev.combine(m, per, Y); rec, D, it = _view(ev, m, per, Y); ok, j = picked(D, it, Y, *SCORES[sc]); yt = D['y'][j[ok]]
        cols = {'abs_y': np.abs(yt)} | {v: np.abs(f_[ok] - yt) for v, f_ in Ft.items()}
        if m != 'US': cols['chosen'] = cols[ev.PICK[(m, per, Y)][f'chosen on val {sc}']]
        parts.append(pd.DataFrame(cols, index=pd.DatetimeIndex(D['a'][it[ok]])).groupby(level=0).sum())
    return pd.concat(parts).sort_index() if parts else None


def bootstrap(ev, out, note):
    """Block-bootstrap 95% intervals of the gain over A (best variant and validation choice; 1- and 3-month blocks) and of the
    differences between markets' best gains (the same calendar blocks drawn for every market)."""
    rng = np.random.default_rng(12345); rows, DS = [], {}
    for m in _tm(ev):
        for per in ev.periods:
            S_ = day_sums(ev, m, per)
            if S_ is None: continue
            DS[(m, per)] = S_; best = ev.best_variant(m, per)
            for blk, lab in ((1, '1-month blocks'), (3, '3-month blocks')):
                G_ = S_.groupby(_blocks(S_.index, blk)).sum(); nb = len(G_)
                for v in (best, 'chosen'):
                    e_v, e_a, ay = G_[v].to_numpy(), G_['A'].to_numpy(), G_['abs_y'].to_numpy(); bs = []
                    for _ in range(N_BOOT):
                        ix = rng.integers(0, nb, nb); bs.append(100 * (e_a[ix].sum() - e_v[ix].sum()) / ay[ix].sum())
                    pt = 100 * (e_a.sum() - e_v.sum()) / ay.sum(); lo, hi = np.percentile(bs, [2.5, 97.5])
                    rows.append({'market': m, 'period': per, 'model': v if v != 'chosen' else 'chosen on validation', 'blocks': lab, 'n blocks': nb,
                                 'gain': round(pt, 2), '95% low': round(lo, 2), '95% high': round(hi, 2), 'share of draws > 0': round(float(np.mean(np.array(bs) > 0)), 3)})
    B_ = pd.DataFrame(rows); B_.to_csv(f'{out}/bootstrap_gains.csv', index=False)
    for _, r in B_[B_.blocks == '3-month blocks'].iterrows():
        note('5.2 / C', f'95% interval of the gain, {r["model"]} (3-month block bootstrap)', r['market'], r['period'], f'{r["gain"]:+.2f} [{r["95% low"]:+.2f}, {r["95% high"]:+.2f}]', '1-month blocks in bootstrap_gains.csv')
    dr = []
    for per in ev.periods:
        mk = [m for m in _tm(ev) if (m, per) in DS]
        if len(mk) < 2: continue
        bestv = {m: ev.best_variant(m, per) for m in mk}
        for blk, lab in ((1, '1-month blocks'), (3, '3-month blocks')):
            allm = sorted(set().union(*[set(DS[(m, per)].index.to_period('M')) for m in mk])); grp = {mm: i // blk for i, mm in enumerate(allm)}; nb = max(grp.values()) + 1
            GM = {m: DS[(m, per)].groupby(np.array([grp[mm] for mm in DS[(m, per)].index.to_period('M')])).sum().reindex(range(nb), fill_value=0.0) for m in mk}
            draws = {m: [] for m in mk}
            for _ in range(N_BOOT):
                ix = rng.integers(0, nb, nb)
                for m in mk:
                    g_ = GM[m]; ay = g_['abs_y'].to_numpy()[ix].sum()
                    draws[m].append(100 * (g_['A'].to_numpy()[ix].sum() - g_[bestv[m]].to_numpy()[ix].sum()) / ay if ay > 0 else np.nan)
            for a_i, a in enumerate(mk):
                for b in mk[a_i + 1:]:
                    d_ = np.array(draws[a]) - np.array(draws[b]); d_ = d_[np.isfinite(d_)]
                    pt = (100 * (GM[a]['A'].sum() - GM[a][bestv[a]].sum()) / GM[a]['abs_y'].sum()) - (100 * (GM[b]['A'].sum() - GM[b][bestv[b]].sum()) / GM[b]['abs_y'].sum())
                    lo, hi = np.percentile(d_, [2.5, 97.5])
                    dr.append({'period': per, 'blocks': lab, 'pair': f'{a} - {b}', 'difference of best gains': round(pt, 2), '95% low': round(lo, 2), '95% high': round(hi, 2),
                               'excludes 0': bool(lo > 0 or hi < 0)})
    Dd = pd.DataFrame(dr); Dd.to_csv(f'{out}/bootstrap_market_differences.csv', index=False)
    for _, r in Dd[Dd.blocks == '3-month blocks'].iterrows():
        note('5.2', 'difference of best gains between markets, 95% interval (3-month blocks)', r['pair'], r['period'], f'{r["difference of best gains"]:+.2f} [{r["95% low"]:+.2f}, {r["95% high"]:+.2f}]',
             'excludes 0' if r['excludes 0'] else 'includes 0')
    return B_, Dd


# ======================= Appendix A: validation skill =======================
def validation_skills(ev, seeds, label, per='2023-25'):
    rows = []
    for (m, p_, Y), rec in ev.FC.items():
        if p_ != per: continue
        DO = ev.DO[m]
        for v in (['A'] if m == 'US' else ['A', 'B', 'C', 'D']):
            fs = [rec['va'][(v, s_)] for s_ in seeds if (v, s_) in rec['va']]
            if len(fs) < len(seeds): continue
            f_ = pd.concat(fs, axis=1).mean(axis=1); iv = f_.index.to_numpy(); fv = f_.to_numpy(); yv = DO['y'][iv]
            k = ev.kfit(fv, yv)
            rows.append({'network': label, 'market': m, 'variant': v, 'fold': Y, 'k': float(k), 'validation skill %': 100 * (1 - np.mean(np.abs(k * fv - yv)) / np.mean(np.abs(yv)))})
    return pd.DataFrame(rows)


# ======================= robustness tables and simple benchmarks (Sections 5.4 and 7, Tables 8, C1-C3) =======================
class Robustness:
    """Strict liquidity subsets, equal weight per name, leave one name out, surface-fit diagnostics, simple benchmarks with the
    enlarged model choice, their bootstrap intervals, and skill by test year."""
    ALPHAS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]; TREES = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 600]
    FAM = {'Ridge window': 'win', 'Ridge last date': 'last', 'Term slope only': 'slope', 'GB abs-error': 'win'}
    NETS = ['A', 'B', 'C', 'D', 'E']

    def __init__(self, ev: Evaluation, layout: Layout, out, note, log=print):
        self.ev, self.layout, self.out, self.note, self.log = ev, layout, out, note, log
        self.EXTRA, self.POOL_PICK, self.US_FITS = {}, {}, {}

    MODELS_OF = staticmethod(lambda T: [c for c in T.columns if c not in ('tk', 'd', 'fold', 'abs_y')])

    def test_frame(self, m, per, sc, extra=True):
        """One row per scored test sample: name, date, fold, |y|, |error| of every model, the validation choice(s)."""
        ev = self.ev; parts = []
        for Y in ev.periods[per]:
            if (m, per, Y) not in ev.FC: continue
            Fv, Ft = ev.combine(m, per, Y); rec, D, it = _view(ev, m, per, Y); ok, j = picked(D, it, Y, *SCORES[sc]); yt = D['y'][j[ok]]
            if extra and (m, per, Y) in self.EXTRA: Ft = Ft | self.EXTRA[(m, per, Y)]['te']
            cols = {'tk': D['tk'][it[ok]], 'd': D['a'][it[ok]], 'fold': Y, 'abs_y': np.abs(yt)} | {v: np.abs(np.asarray(f_)[ok] - yt) for v, f_ in Ft.items()}
            if m != 'US': cols['chosen'] = cols[ev.PICK[(m, per, Y)]['chosen on val primary']]
            if (m, per, Y) in self.POOL_PICK: cols['chosen (pool)'] = cols[self.POOL_PICK[(m, per, Y)]]
            parts.append(pd.DataFrame(cols))
        return pd.concat(parts, ignore_index=True) if parts else None

    def strict(self):
        """Table 2's strict liquidity rule applied ex ante: test name-year (n, Y) kept if n is in the strict list of year Y-1."""
        rows = []
        for m in [m for m in _tm(self.ev) if m in ('IN', 'HK', 'CA')]:
            p = self.layout.in_year_file(m)
            if not os.path.exists(p): self.log(f'  {m}: {p} not found - skipped'); continue
            L_ = pd.read_parquet(p); L_ = L_[L_['strict'].astype(bool)]
            S_ = {int(y_): set(g['underlying'].astype(str)) for y_, g in L_.groupby('year')}
            for per in self.ev.periods:
                for sc in SCORES:
                    T = self.test_frame(m, per, sc, extra=False)
                    if T is None: continue
                    keep = np.array([tk in S_.get(int(fy) - 1, set()) for tk, fy in zip(T.tk, T.fold)])
                    for lab, T_ in (('all names', T), ('strict subset', T[keep])):
                        if not len(T_): rows.append({'market': m, 'period': per, 'score': sc, 'set': lab, 'names': 0, 'samples': 0}); continue
                        r_ = {'market': m, 'period': per, 'score': sc, 'set': lab, 'names': T_.tk.nunique(), 'samples': len(T_), 'local A skill %': round(sk(T_['A'], T_.abs_y), 2)}
                        for v in [c for c in self.MODELS_OF(T_) if c != 'A']: r_[f'{v} gain'] = round(sk(T_[v], T_.abs_y) - sk(T_['A'], T_.abs_y), 2)
                        rows.append(r_)
        Q1 = pd.DataFrame(rows); Q1.to_csv(f'{self.out}/q1_strict_subsets.csv', index=False); return Q1

    def equal_weight(self):
        rows = []
        for m in _tm(self.ev):
            for per in self.ev.periods:
                for sc in SCORES:
                    T = self.test_frame(m, per, sc, extra=False)
                    if T is None: continue
                    G = T.groupby('tk'); n_ = G.size(); M_ = G[[c for c in self.MODELS_OF(T)] + ['abs_y']].sum().div(n_, axis=0)
                    for lab, (num, den) in (('pooled (as Table 6)', (T, T)), ('equal weight per name', (M_, M_))):
                        a_ = sk(num['A'], den.abs_y)
                        r_ = {'market': m, 'period': per, 'score': sc, 'weighting': lab, 'names': int(len(n_)), 'local A skill %': round(a_, 2)}
                        for v in [c for c in self.MODELS_OF(T) if c != 'A']: r_[f'{v} gain'] = round(sk(num[v], den.abs_y) - a_, 2)
                        rows.append(r_)
        Q2 = pd.DataFrame(rows); Q2.to_csv(f'{self.out}/q2_equal_weight.csv', index=False); return Q2

    def leave_one_name_out(self):
        rows, top = [], []
        for m in _tm(self.ev):
            for per in self.ev.periods:
                bv = self.ev.best_variant(m, per)
                for sc in SCORES:
                    T = self.test_frame(m, per, sc, extra=False)
                    if T is None or bv is None: continue
                    G = T.groupby('tk')[[c for c in self.MODELS_OF(T)] + ['abs_y']].sum(); tot = G.sum()
                    for v in dict.fromkeys([bv, 'chosen']):
                        full = 100 * (tot['A'] - tot[v]) / tot['abs_y']
                        wo = 100 * ((tot['A'] - G['A']) - (tot[v] - G[v])) / (tot['abs_y'] - G['abs_y'])
                        rows.append({'market': m, 'period': per, 'score': sc, 'model': v if v != 'chosen' else 'chosen on validation', 'names': len(G),
                                     'gain': round(full, 2), 'min without one name': round(wo.min(), 2), 'name': wo.idxmin(), 'max without one name': round(wo.max(), 2),
                                     'names whose removal leaves gain <= 0': int((wo <= 0).sum())})
                        if sc == 'primary':
                            c_ = (full - wo).sort_values(ascending=False).head(3)
                            top.append({'market': m, 'period': per, 'model': v if v != 'chosen' else 'chosen on validation',
                                        'largest contributors (points of gain they add)': ', '.join(f'{k} {x:+.2f}' for k, x in c_.items())})
        Q3 = pd.DataFrame(rows); Q3.to_csv(f'{self.out}/q3_leave_one_name_out.csv', index=False); pd.DataFrame(top).to_csv(f'{self.out}/q3_top_contributors.csv', index=False)
        return Q3

    def surface_fits(self):
        """Surface-fit diagnostics per market and year from the collected SVI / SSVI files."""
        rows = []
        for m in self.ev.markets:
            d = self.layout.collected_dir(m); fs = sorted(f for f in glob.glob(f'{d}/historical_svi_*_all.parquet') if not f.endswith('_enriched.parquet'))
            if not fs: continue
            S_ = []
            for f_ in fs:
                d_ = pd.read_parquet(f_, columns=['asof', 'expiry', 'n_options', 'rho', 'b', 'sigma']); d_['tk'] = os.path.basename(f_)[len('historical_svi_'):-len('_all.parquet')]; S_.append(d_)
            S_ = pd.concat(S_, ignore_index=True); S_['year'] = pd.to_datetime(S_['asof']).dt.year
            import pyarrow.parquet as pq
            ss_ = []
            for f_ in glob.glob(f'{d}/historical_ssvi_*_all.parquet'):
                c2 = pq.read_schema(f_).names; ss_.append(pd.read_parquet(f_, columns=[c for c in ('asof', 'fx_rmse_vol', 'fx_fit_cost', 'fx_gj', 'rmse_vol', 'fit_cost') if c in c2]))
            SS = pd.concat(ss_, ignore_index=True) if ss_ else pd.DataFrame(columns=['asof']); SS['year'] = pd.to_datetime(SS['asof']).dt.year if len(SS) else []
            for y_, g in S_.groupby('year'):
                nd = g.groupby(['tk', 'asof']).size()
                r_ = {'market': m, 'year': int(y_), 'name-days fitted': int(len(nd)), 'expiries per name-day (median)': float(nd.median()),
                      'options per expiry (median)': float(g['n_options'].median()), 'options per expiry (p10)': float(g['n_options'].quantile(0.1)),
                      'share of expiries with < 5 options %': round(100 * float((g['n_options'] < 5).mean()), 1),
                      'share |rho| > 0.99 %': round(100 * float((g['rho'].abs() > 0.99).mean()), 1), 'share b ~ 0 (flat smile) %': round(100 * float((g['b'].abs() < 1e-6).mean()), 1),
                      'share sigma < 0.001 (kinked smile) %': round(100 * float((g['sigma'] < 1e-3).mean()), 1)}
                h = SS[SS['year'] == y_] if len(SS) else SS
                if len(h):
                    rv = 'fx_rmse_vol' if 'fx_rmse_vol' in h else ('rmse_vol' if 'rmse_vol' in h else None); fc = 'fx_fit_cost' if 'fx_fit_cost' in h else ('fit_cost' if 'fit_cost' in h else None)
                    if rv: r_ |= {'SSVI fit error, vol points (median)': round(float(h[rv].median()), 2), 'SSVI fit error, vol points (p90)': round(float(h[rv].quantile(0.9)), 2)}
                    if fc: r_ |= {'SSVI fit failed (no fit cost) %': round(100 * float(h[fc].isna().mean()), 1)}
                    if 'fx_gj' in h: r_ |= {'SSVI eta(1+|rho|) max': round(float(h['fx_gj'].max()), 3)}
                rows.append(r_)
        Q4 = pd.DataFrame(rows); Q4.to_csv(f'{self.out}/q4_surface_fits.csv', index=False); return Q4

    # ---- simple benchmarks (Section 4.4): inputs standardised as for the network, target / training sd, sample weights as in
    #      training; penalty or number of trees on the validation year's MAE; output scale on validation. 'B' = fitted on the US fold. ----
    def _feats(self, Xs, mode):
        if mode == 'win': return Xs.reshape(len(Xs), -1)
        if mode == 'last': return Xs[:, -1, :]
        return Xs[:, -1, [self.ev.cfg.features.index('term_slope')]]

    def _fit_family(self, fam, Xtr, ytr, wtr, Xva, yva):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        if fam == 'GB abs-error':
            gb = HistGradientBoostingRegressor(loss='absolute_error', max_iter=10, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
                                               early_stopping=False, warm_start=True, random_state=0)
            best, worse = (np.inf, None), 0
            for n_it in self.TREES:
                gb.set_params(max_iter=n_it); gb.fit(Xtr, ytr, sample_weight=wtr); m_ = np.mean(np.abs(gb.predict(Xva) - yva))
                if m_ < best[0]: best, worse = (m_, copy.deepcopy(gb)), 0
                else:
                    worse += 1
                    if worse >= 3: break
            return best[1]
        best = (np.inf, None)
        for a in (self.ALPHAS if fam != 'Term slope only' else [0.0]):
            r_ = Ridge(alpha=a) if a > 0 else Ridge(alpha=1e-8); r_.fit(Xtr, ytr, sample_weight=wtr); m_ = np.mean(np.abs(r_.predict(Xva) - yva))
            if m_ < best[0]: best = (m_, r_)
        return best[1]

    def _us_models(self, per, Y):
        if (per, Y) in self.US_FITS: return self.US_FITS[(per, Y)]
        DU = self.ev.DO['US']; TR, VA, _ = masks(DU, Y); nU = Norm(self.ev.cfg).fit(DU['X'][TR]); syU = float(np.std(DU['y'][TR]))
        Xtr, _ = nU.tr(DU['X'][TR]); Xva, _ = nU.tr(DU['X'][VA]); out = {}
        for fam, md in self.FAM.items():
            out[fam] = self._fit_family(fam, self._feats(Xtr, md), DU['y'][TR] / syU, DU['w'][TR], self._feats(Xva, md), DU['y'][VA] / syU)
        self.US_FITS[(per, Y)] = (out, nU, syU); return self.US_FITS[(per, Y)]

    def benchmarks(self):
        ev = self.ev
        for (m, per, Y), rec in ev.FC.items():
            DO = ev.DO[m]; DX = ev.DX[m]; TR, VA, _ = masks(DO, Y); nL = Norm(ev.cfg).fit(DO['X'][TR]); sy = float(np.std(DO['y'][TR]))
            iv = np.where(VA)[0]; it_x = rec['it']; Xtr, _ = nL.tr(DO['X'][TR]); Xva, _ = nL.tr(DO['X'][VA]); Xte, _ = nL.tr(DX['X'][it_x]); yv = DO['y'][iv]
            va_, te_ = {}, {}
            for fam, md in self.FAM.items():
                loc = self._fit_family(fam, self._feats(Xtr, md), DO['y'][TR] / sy, DO['w'][TR], self._feats(Xva, md), yv / sy)
                cands = [(f'{fam} A', loc)] + ([(f'{fam} B', self._us_models(per, Y)[0][fam])] if m != 'US' else [])
                for nm, mdl in cands:
                    pv, pt = mdl.predict(self._feats(Xva, md)) * sy, mdl.predict(self._feats(Xte, md)) * sy; k = float(ev.kfit(pv, yv))
                    va_[nm] = pd.Series(k * pv, index=iv); te_[nm] = k * pt
            self.EXTRA[(m, per, Y)] = {'va': va_, 'te': te_}
            self.log(f'  benchmarks {m} {per} {Y}: {len(te_)} models')
        for (m, per, Y), rec in ev.FC.items():   # model choice on the validation year's primary score over the enlarged pool
            if m == 'US': continue
            Fv, _ = ev.combine(m, per, Y); Fv = Fv | self.EXTRA[(m, per, Y)]['va']; DO = ev.DO[m]
            ivA = Fv['A'].index.to_numpy(); ok, j = picked(DO, ivA, Y - 1, *SCORES['primary']); b_ = np.abs(DO['y'][j[ok]]).sum()
            vsk = {v: 1 - np.abs(Fv[v].reindex(ivA).to_numpy()[ok] - DO['y'][j[ok]]).sum() / b_ for v in Fv}
            self.POOL_PICK[(m, per, Y)] = max(vsk, key=vsk.get)
        rows = []
        for m in ev.markets:
            for per in ev.periods:
                for sc in SCORES:
                    T = self.test_frame(m, per, sc)
                    if T is None: continue
                    a_ = sk(T['A'], T.abs_y)
                    for v in self.MODELS_OF(T):
                        rows.append({'market': m, 'period': per, 'score': sc, 'model': v, 'skill %': round(sk(T[v], T.abs_y), 2), 'gain over A': round(sk(T[v], T.abs_y) - a_, 2)})
        Q5 = pd.DataFrame(rows); Q5.to_csv(f'{self.out}/q5_baselines_skill.csv', index=False)
        Q5.pivot_table(index=['score', 'market', 'period'], columns='model', values='skill %').round(2).to_csv(f'{self.out}/q5_skill_wide.csv')
        pd.DataFrame([{'market': m, 'period': per, 'fold': Y, 'chosen (pool)': v, 'chosen (A-E)': ev.PICK[(m, per, Y)]['chosen on val primary']}
                      for (m, per, Y), v in self.POOL_PICK.items()]).to_csv(f'{self.out}/q5_pool_choice.csv', index=False)
        return Q5

    def benchmark_bootstrap(self):
        """3-month blocks in 2023-25, 1-month in 2026; primary score: every benchmark vs A, the best network variant vs each benchmark."""
        ev = self.ev; rng = np.random.default_rng(54321); rows = []
        for m in _tm(ev):
            for per in ev.periods:
                T = self.test_frame(m, per, 'primary'); bv = ev.best_variant(m, per)
                if T is None: continue
                S_ = T.groupby('d')[[c for c in self.MODELS_OF(T)] + ['abs_y']].sum(); S_.index = pd.DatetimeIndex(S_.index)
                G_ = S_.groupby(_blocks(S_.index, 3 if per == '2023-25' else 1)).sum(); nb = len(G_); ay = G_['abs_y'].to_numpy()
                base_ = [v for v in self.MODELS_OF(T) if v not in self.NETS + ['GB', 'chosen', 'chosen (pool)']]
                for a, b in [(v, 'A') for v in base_] + [(bv, v) for v in base_] + [('chosen (pool)', 'A'), ('chosen', 'A')]:
                    if a not in G_ or b not in G_: continue
                    ea, eb = G_[a].to_numpy(), G_[b].to_numpy(); bs = []
                    for _ in range(N_BOOT):
                        ix = rng.integers(0, nb, nb); bs.append(100 * (eb[ix].sum() - ea[ix].sum()) / ay[ix].sum())
                    lo, hi = np.percentile(bs, [2.5, 97.5]); pt = 100 * (eb.sum() - ea.sum()) / ay.sum()
                    rows.append({'market': m, 'period': per, 'comparison': f'{a} minus {b}', 'skill difference': round(pt, 2), '95% low': round(lo, 2),
                                 '95% high': round(hi, 2), 'blocks': nb, 'share of draws > 0': round(float(np.mean(np.array(bs) > 0)), 3)})
        B_ = pd.DataFrame(rows); B_.to_csv(f'{self.out}/q5_bootstrap.csv', index=False); return B_

    def by_year(self):
        rows, agg = [], []
        for m in self.ev.markets:
            for per in self.ev.periods:
                for sc in SCORES:
                    T = self.test_frame(m, per, sc)
                    if T is None: continue
                    for Y, g in T.groupby('fold'):
                        for v in self.MODELS_OF(g): rows.append({'score': sc, 'market': m, 'year': int(Y), 'model': v, 'skill %': round(sk(g[v], g.abs_y), 2), 'samples': len(g)})
                    if per == '2023-25':
                        for v in self.MODELS_OF(T): agg.append({'score': sc, 'market': m, 'model': v, 'skill 2023-25 %': round(sk(T[v], T.abs_y), 2)})
        Y6 = pd.DataFrame(rows); Y6.to_csv(f'{self.out}/q6_skill_by_year.csv', index=False)
        pd.DataFrame(agg).to_csv(f'{self.out}/q6_skill_2023_25.csv', index=False)
        return Y6


# ======================= Appendix C: score decomposition, single seeds, roll check =======================
CELLS = [('primary', 'delayed (t+1)', 'well-measured', (1, True)), ('delayed, all names', 'delayed (t+1)', 'all names', (1, False)),
         ('same-day, well-measured', 'same-day (t)', 'well-measured', (0, True)), ('same-day', 'same-day (t)', 'all names', (0, False))]


def _frame(ev, m, per, LQ, seeds):
    parts, picks = [], []
    vs_ = ['A'] if m == 'US' else ['A', 'B', 'C', 'D', 'E']
    for Y in ev.periods[per]:
        if (m, per, Y) not in ev.FC: continue
        Fv, Ft = ev.combine(m, per, Y, seeds); D = ev.DX[m]; it = ev.FC[(m, per, Y)]['it']
        ok, j = picked(D, it, Y, *LQ); yt = D['y'][j[ok]]
        cols = {'d': D['a'][it[ok]], 'tk': D['tk'][it[ok]], 'abs_y': np.abs(yt)} | {v: np.abs(np.asarray(Ft[v])[ok] - yt) for v in vs_}
        if m != 'US':
            ch = ev.choose(m, Y, Fv, 'primary', vs_); cols['chosen'] = cols[ch]; picks.append(ch)
        parts.append(pd.DataFrame(cols))
    T = pd.concat(parts, ignore_index=True) if parts else None
    if T is not None: T.attrs['picks'] = ' '.join(picks)
    return T


def _boot(T, v, blk, rng):
    S_ = T.groupby('d')[[v, 'A', 'abs_y']].sum(); S_.index = pd.DatetimeIndex(S_.index)
    G_ = S_.groupby(_blocks(S_.index, blk)).sum(); nb = len(G_); e_v, e_a, ay = G_[v].to_numpy(), G_['A'].to_numpy(), G_['abs_y'].to_numpy()
    bs = np.array([100 * (e_a[ix].sum() - e_v[ix].sum()) / ay[ix].sum() for ix in (rng.integers(0, nb, nb) for _ in range(N_BOOT))])
    lo, hi = np.percentile(bs, [2.5, 97.5]); return round(lo, 2), round(hi, 2), round(float((bs > 0).mean()), 3), nb


def score_decomposition(ev, out, note):
    """Table C3: the primary score's two filters taken apart (delayed / same-day target x well-measured / all names)."""
    rng = np.random.default_rng(12345); rows = []
    for m in ev.markets:
        for per in ev.periods:
            bv = None if m == 'US' else ev.best_variant(m, per); blk = 1 if per == '2026' else 3
            for sc, lag, sub, LQ in CELLS:
                T = _frame(ev, m, per, LQ, ev.seeds)
                if T is None or not len(T): continue
                a_ = sk(T['A'], T.abs_y)
                r_ = {'market': m, 'period': per, 'target': lag, 'names': sub, 'samples': len(T), 'name count': T.tk.nunique(), 'local A skill %': round(a_, 2)}
                if m != 'US':
                    r_ |= {f'{v} gain': round(sk(T[v], T.abs_y) - a_, 2) for v in 'BCDE'}
                    r_ |= {'chosen picks': T.attrs['picks'], 'chosen gain': round(sk(T['chosen'], T.abs_y) - a_, 2), 'best variant': bv, 'best gain': round(sk(T[bv], T.abs_y) - a_, 2)}
                    lo, hi, sh, nb = _boot(T, bv, blk, rng); r_ |= {'best 95% low': lo, 'best 95% high': hi}
                    lo, hi, sh, nb = _boot(T, 'chosen', blk, rng); r_ |= {'chosen 95% low': lo, 'chosen 95% high': hi, 'chosen share > 0': sh, 'blocks': nb}
                    note('C3', f'gain over A, target {lag}, {sub}: best ({bv}) / chosen on validation', m, per,
                         f"{r_['best gain']:+.2f} [{r_['best 95% low']:+.2f}, {r_['best 95% high']:+.2f}] / {r_['chosen gain']:+.2f} [{r_['chosen 95% low']:+.2f}, {r_['chosen 95% high']:+.2f}]",
                         f"{r_['samples']:,} samples, {r_['name count']} names, local A {a_:+.2f}%")
                rows.append(r_)
    P1 = pd.DataFrame(rows); P1.to_csv(f'{out}/c3_score_decomposition.csv', index=False); return P1


def single_seeds(ev, out, note):
    """Table 10: every seed scored on its own (own output scale, combination weight and validation choice) against A of the same seed."""
    rows = []
    for m in ev.markets:
        for per in ev.periods:
            bv = None if m == 'US' else ev.best_variant(m, per)
            for sc in ('primary', 'same-day'):
                for s in ev.seeds:
                    T = _frame(ev, m, per, SCORES[sc], [s])
                    if T is None or not len(T): continue
                    a_ = sk(T['A'], T.abs_y); r_ = {'market': m, 'period': per, 'score': sc, 'seed': s, 'local A skill %': round(a_, 2)}
                    if m != 'US':
                        r_ |= {f'{v} gain': round(sk(T[v], T.abs_y) - a_, 2) for v in 'BCDE'}
                        r_ |= {'best variant (ten seeds)': bv, 'best gain': round(sk(T[bv], T.abs_y) - a_, 2), 'chosen picks': T.attrs['picks'], 'chosen gain': round(sk(T['chosen'], T.abs_y) - a_, 2)}
                    rows.append(r_)
    P2 = pd.DataFrame(rows); P2.to_csv(f'{out}/table10_single_seeds.csv', index=False)
    S_ = []
    for (m, per, sc), g in P2.groupby(['market', 'period', 'score'], sort=False):
        if m == 'US':
            S_.append({'market': m, 'period': per, 'score': sc, 'single-seed A skill % (min / median / max)': f"{g['local A skill %'].min():+.2f} / {g['local A skill %'].median():+.2f} / {g['local A skill %'].max():+.2f}"}); continue
        r_ = {'market': m, 'period': per, 'score': sc, 'best variant': g['best variant (ten seeds)'].iloc[0]}
        for c, lab in (('best gain', 'best'), ('chosen gain', 'chosen')):
            r_[f'{lab}: min / median / max'] = f"{g[c].min():+.2f} / {g[c].median():+.2f} / {g[c].max():+.2f}"; r_[f'{lab}: seeds > 0'] = f"{int((g[c] > 0).sum())} of {len(g)}"
        r_['every US-informed variant > 0 in every seed'] = ' '.join(v for v in 'BCDE' if (g[f'{v} gain'] > 0).all()) or 'none'
        S_.append(r_)
        if sc == 'primary':
            note('7.3 / Table 10', f'single-seed gain over A of the same seed, primary: best ({r_["best variant"]}) min / median / max; chosen on validation', m, per,
                 f"{r_['best: min / median / max']} ({r_['best: seeds > 0']} seeds > 0); chosen {r_['chosen: min / median / max']} ({r_['chosen: seeds > 0']})", f'{len(ev.seeds)} seeds, each scored alone')
    pd.DataFrame(S_).to_csv(f'{out}/table10_summary.csv', index=False)
    return P2


def roll_check(ev, out, note):
    """Ex-ante roll against no change on samples whose target ends by 31 Dec 2022 (before every test year) and up to 2025 (Section 4.1)."""
    rows = []
    for m in ev.markets:
        D = ev.DO[m]; ya = pd.DatetimeIndex(D['asof']).year.to_numpy(); ye = pd.DatetimeIndex(D['afut']).year.to_numpy()
        full = D['y'] + D['mech']
        for lab, sel in (('targets ending by 31 Dec 2022 (before every test year)', ye <= 2022), ('samples up to 2025 (as in Section 4.1)', ya <= 2025)):
            for sub, s2 in (('all samples', sel), ('samples with a roll', sel & (D['mech'] != 0))):
                y_, f_ = D['y'][s2], full[s2]
                rows.append({'market': m, 'sample': lab, 'subset': sub, 'samples': int(s2.sum()), 'years': f'{ya[s2].min()}-{ye[s2].max()}' if s2.any() else '',
                             'roll vs no-change skill %': round(100 * (1 - np.abs(y_).mean() / np.abs(f_).mean()), 2) if s2.any() else np.nan,
                             'sd net of roll / sd of full change': round(y_.std() / f_.std(), 3) if s2.any() else np.nan})
        r_ = [r for r in rows if r['market'] == m and r['subset'] == 'all samples']
        note('4.1', 'ex-ante roll vs no-change, skill % (sd ratio): targets ending by 2022 | samples up to 2025', m, 'pre-test',
             f"{r_[0]['roll vs no-change skill %']:+.2f} ({r_[0]['sd net of roll / sd of full change']}) | {r_[1]['roll vs no-change skill %']:+.2f} ({r_[1]['sd net of roll / sd of full change']})",
             f"{r_[0]['samples']:,} samples, {r_[0]['years']}")
    P3 = pd.DataFrame(rows); P3.to_csv(f'{out}/roll_check.csv', index=False); return P3


# ======================= initial network (Sections 6.1, 6.2, 6.4, 7.1, 7.5, 7.7; Appendix A) =======================
def training_curves(run_dir):
    """Validation skill per epoch of every seed-0 model -> peak epoch, peak and last skill (Section 6.2)."""
    rows = []
    for f_ in sorted(glob.glob(f'{run_dir}/pred/*_s0.json')):
        p_ = os.path.basename(f_)[:-5].split('_'); h_ = np.array(json.load(open(f_)).get('hist', []))
        if len(p_) != 4 or not len(h_): continue
        b_ = int(np.argmax(h_[:, 2]))
        rows.append({'market': p_[0], 'variant': p_[1], 'fold': int(p_[2]), 'epochs': len(h_), 'best epoch': int(h_[b_, 0]), 'first val skill %': round(100 * h_[0, 2], 2),
                     'peak val skill %': round(100 * h_[b_, 2], 2), 'last val skill %': round(100 * h_[-1, 2], 2), 'decline after peak': round(100 * (h_[b_, 2] - h_[-1, 2]), 2)})
    return pd.DataFrame(rows)


def permutation_importance(layout: Layout, run_dir, cfg: TrainConfig, markets, folds, variants=('A', 'B'), device=None):
    """Validation skill lost when one input (its whole window and its gating copy) is shuffled across samples, seed-0 models,
    after the output scale; mean over folds, in points (Section 6.1, Table 9)."""
    import torch
    device = device or torch.device('cpu')
    DATA = {m: load_market(layout, m, cfg, verbose=False) for m in ['US'] + list(markets)}
    def model_file(m, v, Y): return f'{run_dir}/models/US_{Y}_s0.pt' if v == 'B' or m == 'US' else f'{run_dir}/models/{m}_{v}_{Y}_s0.pt'
    def load_net(p):
        ck = torch.load(p, map_location='cpu', weights_only=False); m_ = MoE(cfg); m_.load_state_dict(ck['state'] if 'state' in ck else ck); return m_.to(device)
    from .train import masks as tmasks
    rng = np.random.default_rng(0); IMP = {}; KSi = np.round(np.arange(0, cfg.k_cap + 1e-9, 0.1), 2)
    for m in ['US'] + list(markets):
        D_ = DATA[m]
        for v in (['A'] if m == 'US' else list(variants)):
            per_fold = []
            for Y in folds:
                p = model_file(m, v, Y)
                if not os.path.exists(p): continue
                TR, VA, TE, ok = tmasks(D_, Y)
                if not ok: continue
                nB = Norm(cfg).fit(D_['X'][TR]); sy = float(np.std(D_['y'][TR])); Xv_raw, yv = D_['X'][VA], D_['y'][VA]; net = load_net(p)
                base = predict(net, *nB.tr(Xv_raw), device) * sy
                k1 = float(KSi[np.argmin([np.mean(np.abs(k * base - yv)) for k in KSi])]); s0 = 1 - np.mean(np.abs(k1 * base - yv)) / np.mean(np.abs(yv))
                row = {}
                for j, c in enumerate(cfg.features):
                    Xp = Xv_raw.copy(); Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
                    pp = predict(net, *nB.tr(Xp), device) * sy; row[c] = s0 - (1 - np.mean(np.abs(k1 * pp - yv)) / np.mean(np.abs(yv)))
                per_fold.append(row)
            if per_fold: IMP[f'{m} {v}'] = pd.DataFrame(per_fold).mean() * 100
    T_imp = pd.DataFrame(IMP)
    return T_imp.loc[T_imp.mean(axis=1).sort_values(ascending=False).index]


def initial_network_items(ev_i: Evaluation, run_dir, out, note, val_final=None):
    """Transfer before tuning (7.1), the zero-shot model without an output scale (6.4), intercept + slope calibration (7.5),
    listing check (7.7), Appendix A (validation skill, initial vs final), Table 9 and training curves (6.1-6.2)."""
    V = pd.concat([val_final if val_final is not None else pd.DataFrame(), validation_skills(ev_i, [0], 'initial network, seed 0')], ignore_index=True)
    V.to_csv(f'{out}/appendix_a_validation.csv', index=False)
    P = V.groupby(['market', 'variant', 'network'])['validation skill %'].mean().unstack('network')
    if 'final network, seed 0' in P:
        P['change (final - initial, seed 0)'] = P['final network, seed 0'] - P['initial network, seed 0']; P = P.round(2)
        tot = P.mean(numeric_only=True).round(2); tr = P[P.index.get_level_values('variant').isin(['B', 'C', 'D'])]['change (final - initial, seed 0)']
        nf = [c for c in P.columns if c.startswith('final network,') and 'seed 0' not in c]
        note('Appendix A', 'validation skill, mean over markets, variants and folds 2023-25: initial -> final network (seed 0)', 'all', '2022-24 validation years',
             f"{tot['initial network, seed 0']:+.2f} -> {tot['final network, seed 0']:+.2f} ({tot['change (final - initial, seed 0)']:+.2f} points)",
             (f"{nf[0][len('final network, '):]}: {tot[nf[0]]:+.2f}; " if nf else '') + f"transfer variants B-D change {tr.mean():+.2f}, better in {int((tr > 0).sum())} of {len(tr)}")
        for (m, v), r in P.iterrows():
            note('Appendix A', 'validation skill, initial -> final network (seed 0), mean over folds 2023-25', m, v,
                 f"{r['initial network, seed 0']:+.2f} -> {r['final network, seed 0']:+.2f}", f"change {r['change (final - initial, seed 0)']:+.2f}")
    for _, r in ev_i.T6.iterrows():
        note('7.1', f'initial network, {r["score"]}: B skill vs A skill', r['market'], r['period'], f'{r["local A %"] + r["B gain"]:+.1f} vs {r["local A %"]:+.1f}',
             'gains C/D/E ' + ' '.join(f'{r[f"{v} gain"]:+.2f}' for v in 'CDE'))
    KS_bak = ev_i.KS; ev_i.KS = np.array([1.0]); SKr = ev_i.score_all([0])[0]; ev_i.KS = KS_bak
    for m in _tm(ev_i):
        for per in ev_i.periods:
            for sc in SCORES:
                k_ = (m, per, sc, 'B')
                if k_ in SKr: note('6.4', f'initial network: zero-shot skill without the output scale, {sc}', m, per, f'{SKr[k_]:+.1f}', f'scaled: {ev_i.SK[k_]:+.1f}')
    BG = np.round(np.arange(0, 1.501, 0.05), 2); rows = []
    for m in _tm(ev_i):
        for per, folds in ev_i.periods.items():
            acc = {}
            for Y in folds:
                if (m, per, Y) not in ev_i.FC: continue
                rec, D, it = _view(ev_i, m, per, Y); DO = ev_i.DO[m]
                for v in ('A', 'B'):
                    va = rec['va'][(v, 0)]; iv = va.index.to_numpy(); fv = va.to_numpy(); yv = DO['y'][iv]; ft = rec['te'][(v, 0)]
                    k = ev_i.kfit(fv, yv)
                    _, b, a = min((np.mean(np.abs(np.median(yv - b * fv) + b * fv - yv)), b, float(np.median(yv - b * fv))) for b in BG)
                    for sc in SCORES:
                        ok, j = picked(D, it, Y, *SCORES[sc]); yt = D['y'][j[ok]]
                        for lab, f_ in (('scale only', k * ft[ok]), ('intercept + slope', a + b * ft[ok])):
                            s0 = acc.setdefault((v, sc, lab), [0.0, 0.0]); s0[0] += np.abs(f_ - yt).sum(); s0[1] += np.abs(yt).sum()
            for (v, sc, lab), (e_, y_) in acc.items():
                rows.append({'market': m, 'period': per, 'variant': v, 'score': sc, 'calibration': lab, 'skill %': round(100 * (1 - e_ / y_), 2)})
    C_ = pd.DataFrame(rows); C_.to_csv(f'{out}/initial_calibration.csv', index=False)
    for (m, per, v, sc), g in C_.groupby(['market', 'period', 'variant', 'score']):
        g = g.set_index('calibration')['skill %']
        note('7.5', f'initial network, variant {v}, {sc}: intercept + slope vs scale only', m, per, f'{g.get("intercept + slope", np.nan):+.1f} vs {g.get("scale only", np.nan):+.1f}', '')
    listing_check(ev_i, note, [0], 'initial network')
    if os.path.exists(f'{run_dir}/importance.csv'):
        I_ = pd.read_csv(f'{run_dir}/importance.csv', index_col=0); I_.to_csv(f'{out}/table9_importance.csv')
        for c in I_.columns:
            if 'term_slope' in I_.index: note('6.1 / Table 9', 'term slope: validation skill lost when permuted (points)', c.split()[0], c.split()[1], round(float(I_.loc['term_slope', c]), 1), 'initial network, mean over folds')
        if 'US A' in I_:
            for c in I_.columns:
                if c != 'US A': note('6.1', 'Spearman correlation of input importances with the US model at home', c.split()[0], c.split()[1], round(float(I_[c].corr(I_['US A'], method='spearman')), 2), '')
            oth = I_['US A'].drop('term_slope', errors='ignore').sort_values(ascending=False)
            note('6.1', 'US at home: next most important input after the term slope', 'US', 'A', f'{oth.index[0]} {oth.iloc[0]:.1f}', '')
    if os.path.exists(f'{run_dir}/curves.csv'):
        Cu = pd.read_csv(f'{run_dir}/curves.csv'); Cu = Cu[Cu.fold >= 2023].reset_index(drop=True); Cu.to_csv(f'{out}/training_curves.csv', index=False)
        for _, r in Cu[(Cu.fold == 2025) & (Cu.variant == 'A')].iterrows():
            note('6.2', 'initial network, local model, 2025 fold: validation skill at its peak -> at the last epoch', r['market'], f'peak epoch {int(r["best epoch"])}', f'{r["peak val skill %"]:+.1f} -> {r["last val skill %"]:+.1f}', f'{int(r["epochs"])} epochs')
        pk = Cu['best epoch']; note('6.2', 'initial network: epoch of peak validation skill (range over every model)', 'all', 'all folds', f'{int(pk.min())}-{int(pk.max())}', f'median {pk.median():.0f}')
    return C_
