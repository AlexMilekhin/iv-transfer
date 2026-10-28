"""Markets, sample design and folder layout.

Every setting here is the one used for the paper. Two folder layouts are provided:
``Layout`` (the layout this repository writes) and ``DriveLayout`` (the folders of the original
Google Drive run, so that the published states and models can be re-scored with this code).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

MARKETS = ('US', 'BR', 'IN', 'HK', 'CA')
TARGET_MARKETS = ('BR', 'IN', 'HK', 'CA')
SOURCE_MARKET = 'US'


@dataclass(frozen=True)
class MarketSpec:
    code: str
    name: str
    years: Tuple[int, ...]            # years with surfaces (first year = start of the history)
    test_years: Tuple[int, ...]       # walk-forward test years (2026 = holdout)
    buckets: Tuple[float, ...]        # constant maturities in years (0.08 = 29 days, 0.12 = 44 days)
    otm_only: bool                    # keep out-of-the-money quotes only (American-style options)
    forward: str                      # 'parity+carry' (US) or 'carry' (F = S exp(rT))
    chain_format: str                 # 'dolt' (US option-chain export) or 'chain_v2' (exchange files)
    cross_listed: Tuple[str, ...] = ()   # underlyings left out (company also in the US universe)


SPECS = {
    'US': MarketSpec('US', 'United States', tuple(range(2020, 2027)), (2022, 2023, 2024, 2025, 2026), (0.08, 0.12), True, 'parity+carry', 'dolt'),
    'BR': MarketSpec('BR', 'Brazil', tuple(range(2015, 2027)), (2022, 2023, 2024, 2025, 2026), (0.08, 0.12), False, 'carry', 'chain_v2'),
    'IN': MarketSpec('IN', 'India', tuple(range(2018, 2027)), (2023, 2024, 2025, 2026), (0.08,), True, 'carry', 'chain_v2'),
    'HK': MarketSpec('HK', 'Hong Kong', tuple(range(2018, 2027)), (2022, 2023, 2024, 2025, 2026), (0.08,), True, 'carry', 'chain_v2',
                     cross_listed=('ALB', 'BIU', 'JDC')),          # Alibaba, Baidu, JD
    'CA': MarketSpec('CA', 'Canada', tuple(range(2017, 2027)), (2022, 2023, 2024, 2025, 2026), (0.08,), True, 'carry', 'chain_v2',
                     cross_listed=('AMD', 'TSLA', 'OVV')),         # depositary receipts on AMD and Tesla, Ovintiv
}

# ---- sample design (paper Sections 3.4-3.5 and 4.1) ----
GRID_WEEKDAYS = (0, 2, 4)        # Monday / Wednesday / Friday grid
SEQ_LEN = 10                     # L: grid dates in the input window
HORIZON = 6                      # h: grid dates ahead (two calendar weeks)
MAX_FILLED = 2                   # isolated missing input dates filled from the previous grid date, at most this many per window
SPAN_IN_MAX = 30                 # calendar days covered by the input window, at most
SPAN_H_MAX = 20                  # calendar days covered by the horizon, at most
COVID_WINDOW = ('2020-02-15', '2020-06-01')   # left out of every market
MIN_TRAIN_PER_NAME = 50          # an underlying needs this many training samples to enter a fold's training block
MAX_NAME_YEAR_IV = 1.0           # name-years with a median ATM vol above 100% are left out

# ---- network and training (paper Section 4.3, preset 'final_season') ----
ALL17 = ['ATM_IV', 'ATM_skew', 'ATM_curvature', 'T', 'total_var', 'IV_percentile', 'IV_momentum', 'IV_acceleration', 'skew_percentile', 'term_slope',
         'ATM_IV_zscore', 'ATM_skew_zscore', 'total_var_change_lag1', 'total_var_change_lag2', 'total_var_change_lag3', 'total_var_change_lag5',
         'total_var_change_lag7']
GATING = ['IV_percentile', 'IV_momentum', 'ATM_IV_zscore', 'T', 'total_var_change_lag1']
LEAN12 = [c for c in ALL17 if c not in ('ATM_skew', 'ATM_curvature', 'total_var_change_lag2', 'total_var_change_lag5', 'total_var_change_lag7')]


@dataclass
class TrainConfig:
    """Network and training settings. Defaults = the final configuration; ``initial()`` = the configuration before tuning (paper 4.3)."""
    features: list = field(default_factory=lambda: LEAN12 + ['season'])
    gating: list = field(default_factory=lambda: list(GATING))
    hidden: int = 16
    lstm_layers: int = 1
    expert_hidden: int = 16
    n_experts: int = 4
    top_k: int = 2
    gate_hidden: int = 32
    dropout: float = 0.3
    gate_dropout: float = 0.1
    t_mode: str = 'fixed'          # 'fixed': maturity input (T - 0.10) / 0.02 in every market | 'per_market': standardised like the other inputs
    loss: str = 'mae'              # 'mae' | 'mse' | 'huber'
    huber_delta: float = 1.0
    lr: float = 1e-3
    ft_lr: float = 3e-4            # fine-tuning (variant C)
    src_lr: Optional[float] = None
    c_freeze: str = 'none'
    d_us_share: float = 0.5        # pooled training (variant D): share of draws from US samples
    weight_decay: float = 3e-4
    batch: int = 256
    max_epochs: int = 150
    patience: int = 30
    min_improvement: float = 0.001
    lr_patience: int = 5
    load_balance: float = 0.2
    momentum_weight: float = 0.0
    eval_every: int = 0
    ema_decay: float = 0.0
    sample_weights: bool = True
    max_name_year_iv: Optional[float] = MAX_NAME_YEAR_IV
    k_cap: float = 1.0

    @classmethod
    def initial(cls) -> 'TrainConfig':
        return cls(features=list(ALL17), loss='mse', t_mode='per_market')

    def legacy_dict(self) -> dict:
        """The run folder's config.json keys of the original notebooks (used to check that a saved run matches)."""
        return {'HIDDEN': self.hidden, 'LSTM_LAYERS': self.lstm_layers, 'EXPERT_HIDDEN': self.expert_hidden, 'N_EXPERTS': self.n_experts,
                'TOP_K': self.top_k, 'GATE_HIDDEN': self.gate_hidden, 'DROPOUT': self.dropout, 'GATE_DROPOUT': self.gate_dropout,
                'FEATURES': list(self.features), 'GATING': list(self.gating), 'T_MODE': self.t_mode, 'MAX_NAME_YEAR_IV': self.max_name_year_iv,
                'K_CAP': self.k_cap}


# ---- folder layouts ----
class Layout:
    """Folders written by this repository under one data root."""

    def __init__(self, root: str):
        self.root = root.rstrip('/')

    def raw(self, m):                return f'{self.root}/raw/{m.lower()}'
    def chain_file(self, m, year, added=False):
        return f'{self.root}/chains/{m.lower()}/chain_{year}.parquet'
    def rates_file(self, m):         return f'{self.root}/raw/us/dtb3.parquet'   # US only (DTB3); other markets carry r in the chain
    def universe_file(self, m):      return f'{self.root}/universe/{m.lower()}.parquet'
    def in_year_file(self, m):       return f'{self.root}/universe/{m.lower()}_in_year.parquet'     # market scripts: liquidity statistics + in_universe / strict
    def surface_dir(self, m):        return f'{self.root}/surfaces/{m.lower()}'
    def surface_dirs_to_read(self, m, added):   # folders searched for a finished unit, in order (the last one is written)
        return [self.surface_dir(m)]
    def collected_dir(self, m):      return f'{self.root}/collected/{m.lower()}'
    def state_dir(self, m):          return f'{self.root}/states/{m.lower()}'
    def run_dir(self, name):         return f'{self.root}/runs/{name}'
    def results_dir(self, name):     return f'{self.root}/results/{name}'


class DriveLayout(Layout):
    """Folders of the original run on Google Drive (root = .../pm_universe). Read-only use: re-scoring the published states and models."""
    TAG = {'US': '', 'BR': '_mwf', 'IN': '', 'HK': '', 'CA': ''}
    RAW = {'IN': 'in_raw', 'HK': 'hk_raw', 'CA': 'ca2_raw'}

    def chain_file(self, m, year, added=False):
        dv = f'{self.root}/deepvol'
        if m == 'US':
            if added: return f'{dv}/xa/chains/us/us_chains_xa.parquet'
            return f'{dv}/us_raw/us_chains_2026.parquet' if year == 2026 else f'{dv}/us_raw/us_chains_2020_2025.parquet'
        if added and m != 'CA': return f'{dv}/xa/chains/{m.lower()}/chain_v2_{year}.parquet'
        if m == 'BR': return f'{self.root}/b3/iv/chain_v2_{year}.parquet'
        return f'{dv}/{self.RAW[m]}/chain_v2/chain_v2_{year}.parquet'
    def rates_file(self, m):         return f'{self.root}/deepvol/us_raw/dtb3.parquet'
    def universe_file(self, m):
        return f'{self.root}/deepvol/xa/' + ('universe_ca2_xa.parquet' if m == 'CA' else f'universe_{m.lower()}.parquet')
    def in_year_file(self, m):
        return {'US': f'{self.root}/deepvol/us_raw/us_universe.parquet', 'BR': f'{self.root}/b3/liquidity_v2.parquet',
                'IN': f'{self.root}/deepvol/in_raw/liquidity_in.parquet', 'HK': f'{self.root}/deepvol/hk_raw/liquidity_hk.parquet',
                'CA': f'{self.root}/deepvol/ca2_raw/liquidity_ca2.parquet'}[m]
    def surface_dir(self, m):        return f'{self.root}/deepvol/{m.lower()}_fin/surfaces'
    def surface_dirs_to_read(self, m, added):
        dv = f'{self.root}/deepvol'
        if m == 'CA': return [self.surface_dir(m)]
        return ([f'{dv}/{m.lower()}_xa/surfaces'] if added else [f'{dv}/{m.lower()}/surfaces', f'{dv}/{m.lower()}_h26/surfaces']) + [self.surface_dir(m)]
    def collected_dir(self, m):      return f'{self.root}/deepvol/{m.lower()}_fin'
    def state_dir(self, m):          return f'{self.root}/deepvol/{m.lower()}_fin/state{self.TAG[m]}'
    def run_dir(self, name):         return f'{self.root}/deepvol/tune/{name}'
    def results_dir(self, name):     return f'{self.root}/deepvol/xa/{name}'


def layout_from_env(default_root: str = './data') -> Layout:
    """IVT_ROOT = data root; IVT_LAYOUT = 'drive' to read the original Drive folders."""
    root = os.environ.get('IVT_ROOT', default_root)
    return DriveLayout(root) if os.environ.get('IVT_LAYOUT', '').lower() == 'drive' else Layout(root)
