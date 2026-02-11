from __future__ import annotations

import torch
import torch.nn as nn

from .local_policy import LocalPolicyNet
from .global_q import CorridorEncoder, FactorizedQHead


class GAMARLModel(nn.Module):
    """End-to-end model.

    - Local temporal encoder + gated-attn comm over DIG
    - Corridor encoder (CCG) with masked degree-normalized GCN
    - Factorized per-node Q head

    Inputs
      F_hist: [B,N,T,df] (history window; for synthetic we can tile current F)
      F:      [B,N,df]
      A:      [B,N,N]
      m:      [B,N]

    Output
      Q: [B,N,action_dim]
    """

    def __init__(
        self,
        df: int,
        d_model: int,
        n_heads: int,
        n_transformer_layers: int,
        history_len: int,
        gate_hidden: int,
        attn_hidden: int,
        gumbel_tau: float,
        hard_gates: bool,
        gcn_hidden: int,
        gcn_activation: str,
        q_hidden: int,
        action_dim: int,
    ):
        super().__init__()
        self.local = LocalPolicyNet(
            df=df,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            gate_hidden=gate_hidden,
            attn_hidden=attn_hidden,
            gumbel_tau=gumbel_tau,
            hard_gates=hard_gates,
            history_len=history_len,
        )
        # Map local embedding into corridor embedding space and merge
        self.local_to_corr = nn.Linear(2 * d_model, d_model)

        self.corridor = CorridorEncoder(df=df, d_model=d_model, gcn_hidden=gcn_hidden, gcn_activation=gcn_activation)

        self.q_head = FactorizedQHead(in_dim=gcn_hidden + d_model, hidden=q_hidden, action_dim=action_dim)

    def forward(self, F_hist: torch.Tensor, F: torch.Tensor, A: torch.Tensor, m: torch.Tensor):
        h_local, g, a = self.local(F_hist, A)
        h_local_p = self.local_to_corr(h_local)

        Z = self.corridor(F, A, m)

        # concatenate corridor embedding with local temporal/context embedding
        fused = torch.cat([Z, h_local_p], dim=-1)
        Q = self.q_head(fused)
        return Q, {"gates": g, "attn": a, "Z": Z}
