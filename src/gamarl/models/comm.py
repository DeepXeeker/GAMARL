from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.tensor_ops import masked_softmax, gumbel_sigmoid


class GatedAttentionComm(nn.Module):
    """Two-stage communication: hard gating then attention-weighted fusion (Eqs. 8–9).

    Given node embeddings h~_i, compute:
      g_{i<-j} = M_gate([h_i, h_j])  in {0,1} (or relaxed)
      a_{i<-j} = softmax( M_att([W_s h_i, h_j]) )
      c_i = sum_j g * a * h_j

    Implementation details:
      - We mask with adjacency A (potential dependencies).
      - Gates are learned per ordered pair.
    """

    def __init__(self, d: int, gate_hidden: int, attn_hidden: int, gumbel_tau: float = 1.0, hard_gates: bool = True):
        super().__init__()
        self.Ws = nn.Linear(d, d, bias=False)

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * d, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )
        self.tau = gumbel_tau
        self.hard = hard_gates

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return context c, gates g, attention weights a.

        h: [B,N,D]
        A: [B,N,N] or [N,N]
        """
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(h.size(0), -1, -1)

        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)

        # gate logits per pair
        gate_logits = self.gate_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # [B,N,N]
        # adjacency mask
        gate_mask = A
        g = gumbel_sigmoid(gate_logits, tau=self.tau, hard=self.hard) * gate_mask

        # attention logits per pair (mask with adjacency)
        hi_proj = self.Ws(h).unsqueeze(2).expand(B, N, N, D)
        attn_logits = self.attn_mlp(torch.cat([hi_proj, hj], dim=-1)).squeeze(-1)
        a = masked_softmax(attn_logits, mask=gate_mask, dim=-1)  # [B,N,N]

        alpha = g * a
        c = torch.einsum("bij,bjd->bid", alpha, h)  # [B,N,D]
        return c, g, a
