from __future__ import annotations

import torch
import torch.nn.functional as F


def td_loss(q_sa: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(q_sa, target)
