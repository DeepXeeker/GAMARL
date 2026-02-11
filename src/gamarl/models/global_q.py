from __future__ import annotations

import torch
import torch.nn as nn

from .layers import MLP, GraphConvolution


class CorridorEncoder(nn.Module):
    """CCG encoder: feature projection (Eq. 10) + degree-normalized GCN (Eq. 11) + mask (Eq. 12)."""

    def __init__(self, df: int, d_model: int, gcn_hidden: int, gcn_activation: str = "relu"):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(df, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.gcn = GraphConvolution(d_model, gcn_hidden, activation=gcn_activation)

    def forward(self, F: torch.Tensor, A: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # F: [B,N,df]
        H = self.fcn(F)  # [B,N,d]
        G = self.gcn(H, A)  # [B,N,gcn_hidden]
        # mask: m in {0,1}^N -> diag(m)G
        Z = G * m.unsqueeze(-1)
        return Z


class FactorizedQHead(nn.Module):
    """Per-node Q-values over a discrete phase-duration action set."""

    def __init__(self, in_dim: int, hidden: int, action_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,N,D] -> [B,N,A]
        B, N, D = z.shape
        out = self.mlp(z.reshape(B * N, D)).reshape(B, N, -1)
        return out
