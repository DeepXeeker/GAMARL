from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..data.replay_buffer import ReplayBuffer
from ..data.batch import Transition
from ..utils.schedulers import LinearSchedule
from .losses import td_loss


@dataclass
class DQNConfig:
    gamma: float
    lr: float
    target_update_steps: int
    warmup_steps: int
    batch_size: int


class DQNAgent:
    def __init__(
        self,
        model: nn.Module,
        target: nn.Module,
        action_dim: int,
        device: torch.device,
        cfg: DQNConfig,
        eps: LinearSchedule,
    ):
        self.model = model.to(device)
        self.target = target.to(device)
        self.action_dim = int(action_dim)
        self.device = device
        self.cfg = cfg
        self.eps = eps

        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.steps = 0

    @torch.no_grad()
    def act(self, Q: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
        """Epsilon-greedy per node.

        Q: [1,N,A]
        """
        eps = float(self.eps.value(self.steps))
        if np.random.rand() < eps:
            a = np.random.randint(0, self.action_dim, size=(Q.shape[1],), dtype=np.int64)
            return a
        a = torch.argmax(Q, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
        return a

    def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
        if len(buffer) < max(self.cfg.warmup_steps, self.cfg.batch_size):
            return {}

        batch = buffer.sample(self.cfg.batch_size, self.device)

        # Model needs F_hist; we approximate history by repeating current F in training buffer
        B, N, df = batch.F.shape
        T = 8
        F_hist = batch.F.unsqueeze(2).repeat(1, 1, T, 1)
        F_hist_next = batch.F_next.unsqueeze(2).repeat(1, 1, T, 1)

        Q, _ = self.model(F_hist, batch.F, batch.A, batch.m)
        # gather chosen actions per node
        q_sa = torch.gather(Q, dim=-1, index=batch.action.unsqueeze(-1)).squeeze(-1)  # [B,N]
        # reduce across nodes to match corridor reward scalar (mean over nodes)
        q_sa = q_sa.mean(dim=1)  # [B]

        with torch.no_grad():
            Qn, _ = self.target(F_hist_next, batch.F_next, batch.A_next, batch.m_next)
            max_qn = Qn.max(dim=-1).values.mean(dim=1)  # [B]
            target = batch.reward.squeeze(-1) + self.cfg.gamma * (1.0 - batch.done.squeeze(-1)) * max_qn

        loss = td_loss(q_sa, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()

        self.steps += 1
        if self.steps % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.model.state_dict())

        return {"loss": float(loss.item()), "epsilon": float(self.eps.value(self.steps))}