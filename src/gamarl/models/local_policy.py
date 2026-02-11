from __future__ import annotations

import torch
import torch.nn as nn

from .layers import TemporalTransformer, MLP
from .comm import GatedAttentionComm


class LocalPolicyNet(nn.Module):
    """Local (per-intersection) policy encoder.

    It produces per-node embeddings using:
      - observation projection (E)
      - temporal transformer (Eq. 7)
      - gated-attention communication (Eqs. 8–9)

    The final embedding is intended to be consumed by a Q head (value-based learning).
    """

    def __init__(
        self,
        df: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        gate_hidden: int,
        attn_hidden: int,
        gumbel_tau: float,
        hard_gates: bool,
        history_len: int,
    ):
        super().__init__()
        self.history_len = history_len
        self.obs_proj = nn.Linear(df, d_model)
        self.temporal = TemporalTransformer(d_model, n_heads, n_layers)
        self.comm = GatedAttentionComm(d_model, gate_hidden, attn_hidden, gumbel_tau, hard_gates)
        self.post = nn.LayerNorm(2 * d_model)

    def forward(self, F_hist: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """F_hist: [B,N,T,df] history of node features.

        Returns:
          h_local: [B,N,2*d_model] (concat of h and context)
          g, a: [B,N,N]
        """
        B, N, T, df = F_hist.shape
        x = self.obs_proj(F_hist)  # [B,N,T,D]
        x = x.reshape(B * N, T, -1)
        h = self.temporal(x).reshape(B, N, -1)  # [B,N,D]

        c, g, a = self.comm(h, A)
        h_local = self.post(torch.cat([h, c], dim=-1))
        return h_local, g, a
