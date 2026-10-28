"""The stages of the paper, in order. Each stage is resumable and writes only to its own folders.

    universe   in-year lists (from the market scripts) -> universe files with the ex-ante test names
    surfaces   daily smile fits of every unit                         (CPU, many cores; hours for the US)
    states     collected fits -> model-ready samples per market        (CPU; minutes)
    train      final network, 10 seeds, folds 2022-26                  (GPU; hours)
    initial    initial network, 1 seed, + training curves and permutation importance (GPU; ~30 min)
    paper      every table and number of Sections 3-7 and Appendices A and C (CPU; ~2 h)
    trading    Section 8                                               (CPU; ~1 h the first time)
    figures    Figure 2
"""
from __future__ import annotations

import os
import time

import pandas as pd

from . import state as _state, surfaces as _surfaces, tables as T, trading as _trading, universe as _universe
from .config import MARKETS, SPECS, TARGET_MARKETS, DriveLayout, Layout, TrainConfig
from .evaluate import Evaluation
from .train import train_all

FOLDS = [2022, 2023, 2024, 2025, 2026]
PAPER_PERIODS = {'2023-25': [2023, 2024, 2025], '2026': [2026]}
N_SEEDS = 10


def run_names(layout: Layout):
    return ('final_v5', 'final_v5_initial') if isinstance(layout, DriveLayout) else ('final', 'initial')


def stage_universe(layout: Layout, markets=MARKETS):
    for m in markets:
        print(f'{m}: test universes\n' + _universe.write(layout, m).to_string(index=False))


def stage_surfaces(layout: Layout, markets=MARKETS, n_jobs=None):
    for m in markets:
        units, added = _universe.units(layout, m)
        _surfaces.run_units(layout, m, units, added, n_jobs)


def stage_states(layout: Layout, markets=MARKETS, n_jobs=8):
    for m in markets:
        units, added = _universe.units(layout, m)
        _state.collect(layout, m, units, added, n_jobs)
        _state.build(layout, m, units)


def stage_train(layout: Layout, n_seeds=N_SEEDS, run_name=None):
    run_name = run_name or run_names(layout)[0]
    return train_all(layout, run_name, TrainConfig(), TARGET_MARKETS, FOLDS, list(range(n_seeds)))


def stage_initial(layout: Layout, run_name=None):
    """Initial network (paper 4.3 before tuning): squared-error loss, maturity standardised per market, the 17 inputs, one seed."""
    run_name = run_name or run_names(layout)[1]; cfg = TrainConfig.initial()
    run = train_all(layout, run_name, cfg, TARGET_MARKETS, FOLDS, [0])
    T.training_curves(run.path).to_csv(f'{run.path}/curves.csv', index=False)
    T.permutation_importance(layout, run.path, cfg, TARGET_MARKETS, FOLDS).to_csv(f'{run.path}/importance.csv')
    return run


def evaluation(layout: Layout, run_name, cfg, seeds, markets=MARKETS, with_gb=True, log=print, periods=None):
    ev = Evaluation(layout, run_name, cfg, markets, seeds, periods or PAPER_PERIODS, with_gb=with_gb, log=log)
    gate = ev.run_forecasts()
    assert all(ev.gate_ok.get(m) for m in markets), f'reproduction gate failed: {ev.gate_ok}'
    ev.tables()
    return ev, gate


def stage_paper(layout: Layout, n_seeds=N_SEEDS, out=None, log=print, markets=MARKETS, periods=None):
    """Re-scores the final and initial networks and writes every table of the paper to `out`."""
    final_run, init_run = run_names(layout)
    out = out or layout.results_dir('paper'); os.makedirs(out, exist_ok=True); note = T.Numbers(); t0 = time.time()
    ev, gate = evaluation(layout, final_run, TrainConfig(), list(range(n_seeds)), markets, log=log, periods=periods)
    gate.to_csv(f'{out}/gate_final.csv', index=False)
    for k, t_ in {'table6_gains': ev.T6, 'table7_chosen': ev.T7, 'loo_best_variant': ev.T9, 'c1_diebold_mariano': ev.C1, 'c2_2026_bh': ev.C2, 'us_home': ev.USH}.items():
        t_.to_csv(f'{out}/{k}.csv', index=False)
    pd.DataFrame(ev.gb_rows).to_csv(f'{out}/gb_trees_k.csv', index=False)
    val_final = pd.concat([T.validation_skills(ev, [0], 'final network, seed 0'), T.validation_skills(ev, ev.seeds, f'final network, {n_seeds} seeds')], ignore_index=True)
    val_final.to_csv(f'{out}/appendix_a_validation_final.csv', index=False)
    for fn in (T.table1, T.top1_share):
        fn(ev, out, note)
    T.data_quality(ev, layout, out, note); T.retention(ev, out, note); T.output_scales(ev, out, note); T.diagnostics_2026(ev, out, note)
    T.robustness_items(ev, out, note); T.bootstrap(ev, out, note)
    R = T.Robustness(ev, layout, out, note, log)
    R.strict(); R.equal_weight(); R.leave_one_name_out(); R.surface_fits(); R.benchmarks(); R.benchmark_bootstrap(); R.by_year()
    T.score_decomposition(ev, out, note); T.single_seeds(ev, out, note); T.roll_check(ev, out, note)
    log(f'final network done ({(time.time() - t0) / 60:.0f} min)')
    ev_i, gate_i = evaluation(layout, init_run, TrainConfig.initial(), [0], markets, with_gb=False, log=log, periods=periods)
    gate_i.to_csv(f'{out}/gate_initial.csv', index=False)
    T.initial_network_items(ev_i, layout.run_dir(init_run), out, note, val_final)
    note.save(f'{out}/paper_numbers.csv')
    log(f'all tables in {out} ({(time.time() - t0) / 60:.0f} min)')
    return ev


def us_chain_files(layout: Layout):
    if isinstance(layout, DriveLayout):
        dv = f'{layout.root}/deepvol'
        return [p for p in (f'{dv}/us_raw/us_chains_2020_2025.parquet', f'{dv}/us_raw/us_chains_2026.parquet', f'{dv}/xa/chains/us/us_chains_xa.parquet') if os.path.exists(p)]
    return sorted(p for p in (layout.chain_file('US', y) for y in SPECS['US'].years) if os.path.exists(p))


def stage_trading(layout: Layout, n_seeds=N_SEEDS, out=None, cache=None, log=print, periods=None):
    final_run, _ = run_names(layout); periods = periods or PAPER_PERIODS
    out = out or layout.results_dir('paper'); cache = cache or f'{layout.results_dir("trading")}'; os.makedirs(out, exist_ok=True); os.makedirs(cache, exist_ok=True)
    note = T.Numbers(); fp = f'{cache}/forecasts.parquet'
    if not os.path.exists(fp):
        ev = Evaluation(layout, final_run, TrainConfig(), ['US'], list(range(n_seeds)), periods, with_gb=True, log=log); ev.run_forecasts()
        _trading.us_forecasts(ev, layout).to_parquet(fp, index=False)
    FC_T = pd.read_parquet(fp)
    ends = {'2023-25': pd.Timestamp('2025-12-31'), '2026': pd.Timestamp('2026-10-12')}
    S_all, CAL = _trading.candidates(layout, FC_T, ends['2026'])
    T2RAW = _trading.price_all(layout, S_all, CAL, us_chain_files(layout), f'{cache}/trades', log=log)
    log(_trading.evaluate(FC_T, T2RAW, S_all, CAL, periods, ends, out, note))
    note.save(f'{out}/paper_numbers_trading.csv')


def stage_figures(out):
    from .figures import figure2
    figure2(f'{out}/bootstrap_gains.csv', f'{out}/table6_gains.csv', f'{out}/figure2.png')
