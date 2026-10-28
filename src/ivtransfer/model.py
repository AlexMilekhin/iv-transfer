"""The mixture-of-experts recurrent network (paper Section 4.2, Figure 1) and input standardisation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig

T_CENTER, T_SCALE = 0.10, 0.02     # fixed maturity scale: (T - 0.10) / 0.02 in every market


class Norm:
    """Standardise inputs with the training block's statistics; with t_mode 'fixed' maturity is on one scale in every market."""

    def __init__(self, cfg: TrainConfig):
        self.feat, self.gate, self.t_mode = list(cfg.features), list(cfg.gating), cfg.t_mode

    def fit(self, X):
        f_ = X.reshape(-1, X.shape[-1]); self.mu, self.sd = f_.mean(axis=0), f_.std(axis=0) + 1e-8
        g_ = X[:, -1, [self.feat.index(c) for c in self.gate]]; self.gmu, self.gsd = g_.mean(axis=0), g_.std(axis=0) + 1e-8
        if self.t_mode == 'fixed' and 'T' in self.feat:
            self.mu[self.feat.index('T')], self.sd[self.feat.index('T')] = T_CENTER, T_SCALE
            if 'T' in self.gate: self.gmu[self.gate.index('T')], self.gsd[self.gate.index('T')] = T_CENTER, T_SCALE
        return self

    def tr(self, X):
        """-> (standardised input windows, standardised gating inputs of the last date)."""
        return ((X - self.mu) / self.sd).astype(np.float32), ((X[:, -1, [self.feat.index(c) for c in self.gate]] - self.gmu) / self.gsd).astype(np.float32)


class MoE(nn.Module):
    """LSTM encoder of the input window -> n_experts heads; a gating network on the last date's gating inputs keeps the top_k experts
    and weights them by a softmax over their logits. Layer order fixed so that seeds give the published initial weights."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.encoder = nn.LSTM(input_size=len(cfg.features), hidden_size=cfg.hidden, num_layers=cfg.lstm_layers, batch_first=True,
                               dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0, bidirectional=False)
        self.gating = nn.Sequential(nn.Linear(len(cfg.gating), cfg.gate_hidden), nn.ReLU(), nn.Dropout(cfg.gate_dropout), nn.Linear(cfg.gate_hidden, cfg.n_experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(cfg.hidden, cfg.expert_hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.expert_hidden, 1))
                                      for _ in range(cfg.n_experts)])

    def forward(self, x, g):
        """-> (forecast, average expert weight per expert = load)."""
        _, (h, _) = self.encoder(x); h = h[-1]
        lg = self.gating(g); tv, ti = torch.topk(lg, self.top_k, dim=-1)
        wts = torch.zeros_like(lg).scatter(1, ti, F.softmax(tv, dim=-1))
        return (wts * torch.stack([e(h).squeeze(-1) for e in self.experts], dim=1)).sum(dim=1), wts.mean(dim=0)


def seed_all(s: int):
    torch.manual_seed(s); np.random.seed(s)


@torch.no_grad()
def predict(model: MoE, X, G, device=torch.device('cpu')):
    model.eval()
    if len(X) == 0: return np.zeros(0)
    return np.concatenate([model(torch.from_numpy(X[i:i + 4096]).to(device), torch.from_numpy(G[i:i + 4096]).to(device))[0].cpu().numpy()
                           for i in range(0, len(X), 4096)]).astype(float)


def n_parameters(cfg: TrainConfig) -> int:
    return sum(p.numel() for p in MoE(cfg).parameters())
