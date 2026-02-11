from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class SeedState:
    seed: int


def set_seed(seed: int) -> SeedState:
    """Seed python, numpy, and torch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return SeedState(seed=seed)
