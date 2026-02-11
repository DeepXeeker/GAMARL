from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FeatureSpec:
    df: int = 11


class ObservationBuilder:
    """Build node features F^k in the paper (df=11).

    Features are grouped as in Table 2:
      - Intersection state (2): residual queue q_i^k, waiting-time/delay proxy w_i^k
      - Stream context (7): direction one-hot (4), rolling speed vbar, speed deviation dv, position proxy ell
      - Road/lane (2): inbound link length Lin, inbound lane count n_lane

    In SUMO, these should be computed from detectors/lanes; in synthetic env we generate proxies.
    """

    def __init__(self, n: int, spec: FeatureSpec | None = None):
        self.n = int(n)
        self.spec = spec or FeatureSpec()

    def build(self,
              q: np.ndarray,
              w: np.ndarray,
              direction_onehot: np.ndarray,
              vbar: np.ndarray,
              dv: np.ndarray,
              pos_proxy: np.ndarray,
              link_len: np.ndarray,
              n_lane: np.ndarray) -> np.ndarray:
        # Shapes: [N] scalars, direction_onehot [N,4]
        F = np.concatenate([
            q.reshape(-1, 1),
            w.reshape(-1, 1),
            direction_onehot,
            vbar.reshape(-1, 1),
            dv.reshape(-1, 1),
            pos_proxy.reshape(-1, 1),
            link_len.reshape(-1, 1),
            n_lane.reshape(-1, 1),
        ], axis=1)
        assert F.shape[1] == self.spec.df
        return F.astype(np.float32)
