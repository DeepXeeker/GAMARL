from __future__ import annotations

import random
from dataclasses import asdict
from typing import List

import torch

from .batch import Transition, Batch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, t: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        batch = random.sample(self.data, batch_size)
        # stack
        def stack(key: str):
            return torch.stack([getattr(b, key) for b in batch], dim=0).to(device)

        return Batch(
            F=stack("F"),
            A=stack("A"),
            m=stack("m"),
            action=stack("action"),
            reward=stack("reward"),
            F_next=stack("F_next"),
            A_next=stack("A_next"),
            m_next=stack("m_next"),
            done=stack("done"),
        )
