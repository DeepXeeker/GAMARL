from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.tensor_ops import gcn_norm, add_self_loops


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:, :T]


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.pe = PositionalEncoding(d_model)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        y = self.enc(x)
        return y[:, -1]  # last token as summary


class GraphConvolution(nn.Module):
    """Single-layer degree-normalized GCN (Eq. 11 in the paper)."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = activation

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # H: [B,N,D] or [N,D]; A: [B,N,N] or [N,N]
        if A.dim() == 2:
            A_hat = add_self_loops(A)
            An = gcn_norm(A_hat)
            out = An @ self.lin(H) + self.bias
        else:
            A_hat = add_self_loops(A)
            An = gcn_norm(A_hat)
            out = An @ self.lin(H) + self.bias
        if self.activation == "relu":
            return F.relu(out)
        if self.activation == "tanh":
            return torch.tanh(out)
        return out
