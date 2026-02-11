from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Transition:
    # global observation y^k = (F, A, m)
    F: torch.Tensor          # [M, df]
    A: torch.Tensor          # [M, M]
    m: torch.Tensor          # [M]
    action: torch.Tensor     # [M] discrete action per intersection
    reward: torch.Tensor     # scalar
    F_next: torch.Tensor
    A_next: torch.Tensor
    m_next: torch.Tensor
    done: torch.Tensor       # scalar (0/1)


@dataclass
class Batch:
    F: torch.Tensor
    A: torch.Tensor
    m: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    F_next: torch.Tensor
    A_next: torch.Tensor
    m_next: torch.Tensor
    done: torch.Tensor

    @property
    def device(self):
        return self.F.device
