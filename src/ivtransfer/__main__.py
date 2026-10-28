"""Command line: python -m ivtransfer <stage> [--root DIR] [--drive] [--markets US BR ...] [--seeds 10] [--jobs N]

Stages: universe, surfaces, states, train, initial, paper, trading, figures, all (= every stage after the market scripts).
--drive reads and writes the folder layout of the original Google Drive run (root = .../pm_universe).
"""
import argparse

from . import pipeline as P
from .config import MARKETS, DriveLayout, Layout


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ivtransfer', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['universe', 'surfaces', 'states', 'train', 'initial', 'paper', 'trading', 'figures', 'all'])
    ap.add_argument('--root', default='./data', help='data root (default ./data)')
    ap.add_argument('--drive', action='store_true', help='use the folder layout of the original Drive run')
    ap.add_argument('--markets', nargs='+', default=list(MARKETS))
    ap.add_argument('--seeds', type=int, default=P.N_SEEDS)
    ap.add_argument('--jobs', type=int, default=None, help='parallel processes for the surface fits (default: all cores)')
    a = ap.parse_args(argv)
    L = DriveLayout(a.root) if a.drive else Layout(a.root)
    st = a.stage
    if st in ('universe', 'all'): P.stage_universe(L, a.markets)
    if st in ('surfaces', 'all'): P.stage_surfaces(L, a.markets, a.jobs)
    if st in ('states', 'all'): P.stage_states(L, a.markets)
    if st in ('train', 'all'): P.stage_train(L, a.seeds)
    if st in ('initial', 'all'): P.stage_initial(L)
    if st in ('paper', 'all'): P.stage_paper(L, a.seeds)
    if st in ('trading', 'all'): P.stage_trading(L, a.seeds)
    if st in ('figures', 'all'): P.stage_figures(L.results_dir('paper'))


if __name__ == '__main__':
    main()
