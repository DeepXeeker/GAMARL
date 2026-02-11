from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class StepOutput:
    F: np.ndarray      # [M, df]
    A: np.ndarray      # [M, M]
    m: np.ndarray      # [M]
    reward: float
    done: bool
    info: dict


class BaseEnv(ABC):
    """Base environment returning the global structured input y^k=(F^k,A,m^k)."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: np.ndarray) -> StepOutput:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_intersections(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def df(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError
