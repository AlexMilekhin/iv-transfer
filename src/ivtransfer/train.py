"""Training of the US source model and of variants A-D in the target markets (paper Sections 4.3-4.5).

Walk-forward fold for test year Y: training samples whose target ends by 31 Dec Y-2, validation samples that start after that date
and whose target ends by 31 Dec Y-1, test samples that start in Y. Every model is saved with its validation and test forecasts in
<run>/models and <run>/pred, so a run can be resumed and re-scored.

Variants per target market, fold and seed: A local only; B the US model unchanged (zero-shot); C the US model fine-tuned on local
data; D a model trained on local and US 29-day samples (half of the draws each). E (a combination of A and D) is formed at scoring.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ALL17, MIN_TRAIN_PER_NAME, TrainConfig, Layout
from .model import MoE, Norm, predict, seed_all

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================= data =======================
def season_input(mt: pd.DataFrame, y: np.ndarray, steps: int) -> np.ndarray:
    """Seasonal input: the same name and maturity's realised target about 13, 26 and 52 weeks earlier (nearest grid date within
    3 days; mean of those found, 0 if none), repeated over the input window."""
    bk = np.round(mt['target_T'].to_numpy(float) * 365).astype(int); a0 = pd.to_datetime(mt['asof'])
    D_ = pd.DataFrame({'tk': mt['ticker'].astype(str).values, 'bk': bk, 'asof': a0.values, 'y': y}); S_ = D_.sort_values('asof'); out = []
    for L in (91, 182, 364):
        q = D_[['tk', 'bk']].assign(look=D_['asof'] - pd.Timedelta(days=L)).reset_index().sort_values('look')
        mm = pd.merge_asof(q, S_.rename(columns={'asof': 'look'})[['tk', 'bk', 'look', 'y']], on='look', by=['tk', 'bk'], direction='nearest', tolerance=pd.Timedelta(days=3))
        out.append(mm.set_index('index')['y'].reindex(D_.index).to_numpy())
    V_ = np.vstack(out); n_ = np.isfinite(V_).sum(axis=0); s_ = np.where(n_ > 0, np.nansum(V_, axis=0) / np.maximum(n_, 1), 0.0)
    return np.repeat(np.nan_to_num(s_, nan=0.0)[:, None], steps, axis=1)


def model_inputs(X17: np.ndarray, mt: pd.DataFrame, y: np.ndarray, features) -> np.ndarray:
    return np.concatenate([X17[:, :, [ALL17.index(c)]] if c in ALL17 else season_input(mt, y, X17.shape[1])[:, :, None] if c == 'season'
                           else _raise(c) for c in features], axis=2).astype(np.float32)


def _raise(c):
    raise ValueError(f'unknown input {c}')


def added_units(layout: Layout, m: str) -> set:
    U_ = pd.read_parquet(layout.universe_file(m))
    return set(zip(U_.loc[U_['added'], 'name'].astype(str), U_.loc[U_['added'], 'year'].astype(int)))


def load_market(layout: Layout, m: str, cfg: TrainConfig, verbose=True) -> dict:
    """Training view of a market's state: name-years with a median ATM vol above 100% and the ex-ante test additions (test names
    only) are left out."""
    st = layout.state_dir(m)
    A_ = np.load(f'{st}/arrays.npz', allow_pickle=True); assert A_['X'].shape[2] == 17, f'{m}: unexpected input width {A_["X"].shape}'
    mt = pd.read_parquet(f'{st}/meta.parquet').reset_index(drop=True); keep = np.ones(len(mt), bool)
    if cfg.max_name_year_iv is not None:
        ny = pd.DataFrame({'tk': mt['ticker'].astype(str), 'yr': pd.to_datetime(mt['asof']).dt.year, 'iv': A_['X'][:, -1, ALL17.index('ATM_IV')]})
        keep = (ny.groupby(['tk', 'yr'])['iv'].transform('median') <= cfg.max_name_year_iv).to_numpy()
        if verbose: print(f'{m}: {int((~keep).sum()):,} samples of name-years above {cfg.max_name_year_iv:.0%} median ATM vol left out')
    if os.path.exists(layout.universe_file(m)):
        add_ = added_units(layout, m)
        drop_ = np.array([u in add_ for u in zip(mt['ticker'].astype(str), pd.to_datetime(mt['asof']).dt.year)], bool); keep = keep & ~drop_
        if verbose: print(f'{m}: {int(drop_.sum()):,} samples of {len(add_)} ex-ante test additions left out of the training data')
    mt = mt[keep].reset_index(drop=True); X17 = A_['X'][keep].astype(np.float32); yk = A_['y'][keep].astype(np.float64)
    X = model_inputs(X17, mt, yk, cfg.features)
    return dict(m=m, X=X, y=yk, w=A_['w'][keep].astype(np.float32), tk=mt['ticker'].astype(str).to_numpy(),
                asof=pd.to_datetime(mt['asof']).astype('datetime64[ns]'), afut=pd.to_datetime(mt['asof_future']).astype('datetime64[ns]'),
                tT=mt['target_T'].to_numpy(float), mech=mt['mech'].to_numpy(float), same=mt['same'].to_numpy(bool) if 'same' in mt else None)


def masks(D: dict, Y: int, min_n: int = MIN_TRAIN_PER_NAME):
    """Training, validation and test masks of fold Y; an underlying needs min_n training samples to be trained on."""
    tr_e, va_e = pd.Timestamp(f'{Y - 2}-12-31'), pd.Timestamp(f'{Y - 1}-12-31')
    TR = (D['afut'] <= tr_e).values; VA = ((D['asof'] > tr_e) & (D['afut'] <= va_e)).values; TE = (D['asof'].dt.year == Y).values
    c_ = pd.Series(D['tk'][TR]).value_counts(); TR = TR & np.isin(D['tk'], c_.index[c_ >= min_n])
    return TR, VA, TE, (TR.sum() >= 2000 and VA.sum() >= 200 and TE.sum() > 0)


# ======================= training =======================
class TrainSet:
    """Training block on the device. Each epoch draws n samples with replacement, with probabilities proportional to
    1 / sqrt(name's sample count) (names balanced); for D the two markets get d_us_share and 1 - d_us_share of the draws.
    The global RNG draw per epoch keeps dropout masks as in the published runs (which used a DataLoader)."""

    def __init__(self, X, G, y, w, names, seed, batch, market=None, us_share=0.5):
        nm = pd.Series(names); wn = 1.0 / np.sqrt(nm.map(nm.value_counts()).to_numpy())
        if market is not None:
            wn = wn / pd.Series(wn).groupby(np.asarray(market)).transform('sum').to_numpy() * np.where(np.asarray(market) == 1, us_share, 1 - us_share)
        self.wts, self.gen, self.n, self.batch = torch.as_tensor(wn, dtype=torch.double), torch.Generator().manual_seed(seed), len(nm), batch
        self.X, self.G, self.y, self.w = (torch.as_tensor(np.asarray(a, np.float32), device=DEV) for a in (X, G, y, w))

    def batches(self):
        torch.empty((), dtype=torch.int64).random_()
        idx = torch.multinomial(self.wts, self.n, True, generator=self.gen).to(DEV)
        for b in range(self.n // self.batch):
            i = idx[b * self.batch:(b + 1) * self.batch]; yield self.X[i], self.G[i], self.y[i], self.w[i]


class ValSet:
    def __init__(self, X, G, y): self.X, self.G, self.y = (torch.as_tensor(np.asarray(a, np.float32), device=DEV) for a in (X, G, y))


def loss_fn(cfg: TrainConfig, p, y, w, load, lag1=None):
    """Sample-weighted absolute (or squared / Huber) error + load-balancing penalty on the experts' average weights."""
    e = (p - y) ** 2 if cfg.loss == 'mse' else (p - y).abs() if cfg.loss == 'mae' else F.huber_loss(p, y, reduction='none', delta=cfg.huber_delta)
    if not cfg.sample_weights: w = torch.ones_like(w)
    l_ = (e * w).sum() / w.sum() + cfg.load_balance * ((load - 1.0 / cfg.n_experts) ** 2).mean()
    if cfg.momentum_weight > 0 and lag1 is not None:
        l_ = l_ + cfg.momentum_weight * torch.relu((p / (p.std() + 1e-8) * (lag1 / (lag1.std() + 1e-8))).mean())
    return l_


def fit(cfg: TrainConfig, model: MoE, trs: TrainSet, vas: ValSet, lr: float) -> dict:
    """AdamW; learning rate halved after lr_patience flat epochs; early stopping on validation skill against the roll
    (1 - mean |error| / mean |y|) after `patience` epochs without a gain of min_improvement; the best weights are restored."""
    i_lag1 = cfg.features.index('total_var_change_lag1') if 'total_var_change_lag1' in cfg.features else None
    model.to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=cfg.lr_patience, min_lr=1e-6)
    best, best_state, bad, best_ep, ep, hist, hist_sub = -np.inf, None, 0, 0, 0, [], []
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg.ema_decay > 0 else None
    nsteps = trs.n // cfg.batch

    def val_skill():
        if ema is not None: raw_ = {k: v.detach().clone() for k, v in model.state_dict().items()}; model.load_state_dict(ema)
        model.eval()
        with torch.no_grad(): P_ = torch.cat([model(vas.X[i:i + 4096], vas.G[i:i + 4096])[0] for i in range(0, len(vas.y), 4096)])
        sk_ = float(1 - (P_ - vas.y).abs().mean() / vas.y.abs().mean())
        snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema is not None: model.load_state_dict(raw_)
        return sk_, snap

    for ep in range(1, cfg.max_epochs + 1):
        model.train(); tl, nb, improved = 0.0, 0, False
        for X, G, y, w in trs.batches():
            opt.zero_grad(); p, load = model(X, G); l_ = loss_fn(cfg, p, y, w, load, X[:, -1, i_lag1] if i_lag1 is not None else None); l_.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); opt.step(); tl += l_.item(); nb += 1
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items(): ema[k].mul_(cfg.ema_decay).add_(v.detach(), alpha=1 - cfg.ema_decay)
            if cfg.eval_every and nb % cfg.eval_every == 0 and nb < nsteps:
                sk_i, snap = val_skill(); model.train(); hist_sub.append([ep - 1 + nb / nsteps, sk_i])
                if sk_i > best + cfg.min_improvement: best, best_ep, best_state, improved = sk_i, round(ep - 1 + nb / nsteps, 3), snap, True
        torch.empty((), dtype=torch.int64).random_()   # the validation DataLoader's draw in the published runs (keeps seeds identical)
        sk, snap = val_skill(); sch.step(sk)
        hist.append([ep, tl / max(nb, 1), float(sk), opt.param_groups[0]['lr']])
        if sk > best + cfg.min_improvement: best, best_ep, best_state, improved = sk, ep, snap, True
        if improved: bad = 0
        else:
            bad += 1
            if bad >= cfg.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return {'epochs': ep, 'best_epoch': best_ep, 'val_skill_scaled': float(best), 'hist': hist, 'hist_sub': hist_sub}


class Run:
    """A run folder: config.json, models/<name>.pt, pred/<name>.parquet (+ .json with the training record)."""

    def __init__(self, path: str, cfg: TrainConfig, extra: dict = None):
        self.path, self.cfg = path, cfg
        os.makedirs(f'{path}/pred', exist_ok=True); os.makedirs(f'{path}/models', exist_ok=True)
        c_ = cfg.legacy_dict() | (extra or {})
        if os.path.exists(f'{path}/config.json'):
            old = json.load(open(f'{path}/config.json'))
            diff = [k for k in cfg.legacy_dict() if k in old and old[k] != json.loads(json.dumps(c_[k]))]
            assert not diff, f'run folder {path} was trained with different settings ({diff})'
        else:
            json.dump(c_, open(f'{path}/config.json', 'w'), indent=1)

    def done(self, name):
        return os.path.exists(f'{self.path}/pred/{name}.parquet')

    def save(self, name, idx_va, pv, idx_te, pt, info, model=None):
        if model is not None: torch.save(model.state_dict(), f'{self.path}/models/{name}.pt')
        pd.DataFrame({'part': ['val'] * len(idx_va) + ['test'] * len(idx_te), 'i': np.r_[idx_va, idx_te], 'pred': np.r_[pv, pt]}).to_parquet(f'{self.path}/pred/{name}.parquet', index=False)
        json.dump(info, open(f'{self.path}/pred/{name}.json', 'w'))

    def us_model(self, Y, s):
        seed_all(s); m_ = MoE(self.cfg)
        m_.load_state_dict(torch.load(f'{self.path}/models/US_{Y}_s{s}.pt', map_location='cpu', weights_only=False)['state']); return m_.to(DEV)


def train_source(run: Run, U: dict, folds, seeds):
    """US model per fold and seed (also scored on its own test year)."""
    cfg = run.cfg; t0 = time.time()
    for Y in folds:
        TR, VA, TE, ok = masks(U, Y)
        if not ok: print(f'US fold {Y}: too few samples, skipped'); continue
        nU = Norm(cfg).fit(U['X'][TR]); Xtr, Gtr = nU.tr(U['X'][TR]); Xva, Gva = nU.tr(U['X'][VA]); Xte, Gte = nU.tr(U['X'][TE]); sy = float(np.std(U['y'][TR]))
        for s in seeds:
            name = f'US_A_{Y}_s{s}'
            if run.done(name): continue
            t1 = time.time(); seed_all(s); mdl = MoE(cfg)
            info = fit(cfg, mdl, TrainSet(Xtr, Gtr, U['y'][TR] / sy, U['w'][TR], U['tk'][TR], s, cfg.batch), ValSet(Xva, Gva, U['y'][VA] / sy), cfg.src_lr or cfg.lr)
            torch.save({'state': mdl.state_dict(), 'sy': sy}, f'{run.path}/models/US_{Y}_s{s}.pt')
            run.save(name, np.where(VA)[0], predict(mdl, Xva, Gva, DEV) * sy, np.where(TE)[0], predict(mdl, Xte, Gte, DEV) * sy, info | {'sy': sy})
            print(f'US fold {Y} seed {s}: {info["epochs"]} epochs (best {info["best_epoch"]}), val skill {info["val_skill_scaled"]:+.3f} | {time.time() - t1:.0f}s', flush=True)
    print(f'US source done | {time.time() - t0:.0f}s')


def train_targets(run: Run, DATA: dict, markets, folds, seeds, variants=('A', 'B', 'C', 'D')):
    cfg = run.cfg; U = DATA['US']
    U29 = np.isclose(U['tT'], 0.08)   # D pools the US 29-day samples only (the maturity every market has)
    U29D = {k: (v[U29] if isinstance(v, np.ndarray) else v[U29].reset_index(drop=True) if isinstance(v, pd.Series) else v) for k, v in U.items()}
    for m in markets:
        D_ = DATA[m]; t0 = time.time()
        for Y in folds:
            TR, VA, TE, ok = masks(D_, Y)
            if not ok: print(f'{m} fold {Y}: too few samples, skipped'); continue
            nB = Norm(cfg).fit(D_['X'][TR]); Xtr, Gtr = nB.tr(D_['X'][TR]); Xva, Gva = nB.tr(D_['X'][VA]); Xte, Gte = nB.tr(D_['X'][TE])
            sy = float(np.std(D_['y'][TR])); iva, ite = np.where(VA)[0], np.where(TE)[0]
            va_ld = ValSet(Xva, Gva, D_['y'][VA] / sy)
            mk_tr = lambda s: TrainSet(Xtr, Gtr, D_['y'][TR] / sy, D_['w'][TR], D_['tk'][TR], s, cfg.batch)
            out = lambda mdl: (predict(mdl, Xva, Gva, DEV) * sy, predict(mdl, Xte, Gte, DEV) * sy)
            for s in seeds:
                if 'A' in variants and not run.done(f'{m}_A_{Y}_s{s}'):
                    seed_all(s); mdl = MoE(cfg); info = fit(cfg, mdl, mk_tr(s), va_ld, cfg.lr); pv, pt = out(mdl); run.save(f'{m}_A_{Y}_s{s}', iva, pv, ite, pt, info, mdl)
                    print(f'{m} {Y} s{s} A: {info["epochs"]} ep (best {info["best_epoch"]}), val {info["val_skill_scaled"]:+.3f}', flush=True)
                if not os.path.exists(f'{run.path}/models/US_{Y}_s{s}.pt'): continue
                if 'B' in variants and not run.done(f'{m}_B_{Y}_s{s}'):
                    pv, pt = out(run.us_model(Y, s)); run.save(f'{m}_B_{Y}_s{s}', iva, pv, ite, pt, {'epochs': 0, 'best_epoch': 0})
                if 'C' in variants and not run.done(f'{m}_C_{Y}_s{s}'):
                    mdl = run.us_model(Y, s)
                    if cfg.c_freeze == 'encoder':
                        for p_ in mdl.encoder.parameters(): p_.requires_grad_(False)
                    info = fit(cfg, mdl, mk_tr(s), va_ld, cfg.ft_lr); pv, pt = out(mdl); run.save(f'{m}_C_{Y}_s{s}', iva, pv, ite, pt, info, mdl)
                    print(f'{m} {Y} s{s} C: {info["epochs"]} ep (best {info["best_epoch"]}), val {info["val_skill_scaled"]:+.3f}', flush=True)
                if 'D' in variants and not run.done(f'{m}_D_{Y}_s{s}'):
                    TRu, _, _, _ = masks(U29D, Y)
                    Xu, yu, wu, tku = U29D['X'][TRu], U29D['y'][TRu], U29D['w'][TRu], U29D['tk'][TRu]
                    nU = Norm(cfg).fit(Xu); Xun, Gun = nU.tr(Xu); syu = float(np.std(yu))
                    seed_all(s); mdl = MoE(cfg)
                    ld = TrainSet(np.concatenate([Xtr, Xun]), np.concatenate([Gtr, Gun]), np.r_[D_['y'][TR] / sy, yu / syu], np.r_[D_['w'][TR], wu],
                                  np.r_[D_['tk'][TR], np.char.add('US:', tku.astype(str))], s, cfg.batch,
                                  market=np.r_[np.zeros(TR.sum(), int), np.ones(len(yu), int)], us_share=cfg.d_us_share)
                    info = fit(cfg, mdl, ld, va_ld, cfg.lr) | {'n_local': int(TR.sum()), 'n_us': int(len(yu))}; pv, pt = out(mdl); run.save(f'{m}_D_{Y}_s{s}', iva, pv, ite, pt, info, mdl)
                    print(f'{m} {Y} s{s} D: {info["epochs"]} ep (best {info["best_epoch"]}), val {info["val_skill_scaled"]:+.3f} ({info["n_local"]:,} local + {info["n_us"]:,} US)', flush=True)
        print(f'{m} done | {time.time() - t0:.0f}s')


def train_all(layout: Layout, run_name: str, cfg: TrainConfig, markets, folds, seeds, variants=('A', 'B', 'C', 'D'), deterministic=True):
    """Train the US source models, then every variant in every target market. Full float32 on the GPU (no TF32), so that the CPU
    re-scoring reproduces the saved forecasts up to rounding."""
    if deterministic:
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    run = Run(layout.run_dir(run_name), cfg, {'MARKETS': list(markets), 'FOLDS': list(folds), 'SEEDS': list(seeds), 'VARIANTS': list(variants)})
    DATA = {m: load_market(layout, m, cfg) for m in ['US'] + list(markets)}
    train_source(run, DATA['US'], folds, seeds)
    train_targets(run, DATA, markets, folds, seeds, variants)
    return run
