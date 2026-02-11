from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .base_env import BaseEnv, StepOutput
from .observation_builder import ObservationBuilder


def _make_adjacency(n: int, kind: str) -> np.ndarray:
    if kind == "line":
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        return A
    if kind == "grid":
        # square-ish grid
        side = int(np.ceil(np.sqrt(n)))
        A = np.zeros((n, n), dtype=np.float32)
        def idx(r,c):
            return r*side + c
        for r in range(side):
            for c in range(side):
                i = idx(r,c)
                if i >= n:
                    continue
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < side and 0 <= cc < side:
                        j = idx(rr,cc)
                        if j < n:
                            A[i,j] = 1.0
        return A
    raise ValueError(f"Unknown adjacency kind: {kind}")


@dataclass
class SyntheticParams:
    episode_horizon: int
    decision_interval_s: int
    traffic_load_range: Tuple[int, int]
    adjacency: str


class SyntheticTINEnv(BaseEnv):
    """A lightweight synthetic corridor environment.

    It is NOT a traffic simulator; it produces plausible queue / delay dynamics to test learning.

    State variables per intersection i:
      q_i: residual queue
      w_i: waiting-time proxy
      v_i: speed proxy (rolling + deviation)

    Actions: discrete phase-duration index.
    We map action to a "service rate" that reduces queues, but can create spillback-like coupling.
    """

    def __init__(self, n_intersections: int, df: int, action_dim: int, params: SyntheticParams):
        self._n = int(n_intersections)
        self._df = int(df)
        self._action_dim = int(action_dim)
        self.params = params

        self.A = _make_adjacency(self._n, params.adjacency)
        self.m = np.ones((self._n,), dtype=np.float32)
        self.obs = ObservationBuilder(self._n)

        self.rng = np.random.default_rng(0)
        self.t = 0

        # per-node static descriptors
        self.link_len = self.rng.uniform(80.0, 320.0, size=(self._n,)).astype(np.float32)
        self.n_lane = self.rng.integers(1, 4, size=(self._n,)).astype(np.float32)

        self.entry = 0  # entry intersection index for downstream waiting penalty (Eq. 5)

    @property
    def n_intersections(self) -> int:
        return self._n

    @property
    def df(self) -> int:
        return self._df

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0

        load_lo, load_hi = self.params.traffic_load_range
        base = self.rng.uniform(load_lo, load_hi)
        # q and w start modest
        self.q = self.rng.gamma(shape=2.0, scale=3.0, size=(self._n,)).astype(np.float32)
        self.w = (self.q * self.rng.uniform(1.0, 3.0, size=(self._n,))).astype(np.float32)
        self.vbar = self.rng.uniform(6.0, 13.0, size=(self._n,)).astype(np.float32)
        self.dv = self.rng.normal(0.0, 0.5, size=(self._n,)).astype(np.float32)

        # direction: one-hot, cycle by index
        dirs = np.zeros((self._n, 4), dtype=np.float32)
        dirs[np.arange(self._n), np.mod(np.arange(self._n), 4)] = 1.0
        self.dirs = dirs

        self.pos_proxy = (self.link_len - (self.vbar + self.dv) * 1.0).astype(np.float32)

        F = self.obs.build(self.q, self.w, self.dirs, self.vbar, self.dv, self.pos_proxy, self.link_len, self.n_lane)
        return F, self.A.copy(), self.m.copy(), {"t": self.t}

    def _service_from_action(self, a: np.ndarray) -> np.ndarray:
        # map action index to a service factor in [0.2, 1.0]
        frac = (a.astype(np.float32) + 1) / float(self._action_dim)
        return 0.2 + 0.8 * frac

    def step(self, actions: np.ndarray) -> StepOutput:
        actions = actions.reshape(-1)
        assert actions.shape[0] == self._n

        self.t += 1

        # arrivals: upstream -> downstream coupling
        arrivals = self.rng.poisson(lam=1.0 + 0.15 * self.q).astype(np.float32)
        # spillback coupling: if downstream is congested, reduce effective service upstream
        neighbor_cong = (self.A @ (self.q / (1.0 + self.q))).astype(np.float32)

        service = self._service_from_action(actions) * (1.0 - 0.15 * np.tanh(neighbor_cong))
        discharged = np.minimum(self.q, self.rng.uniform(0.5, 1.5, size=(self._n,)).astype(np.float32) * service * 3.0)

        self.q = np.clip(self.q + arrivals - discharged, 0.0, 200.0).astype(np.float32)
        # waiting proxy increases with queue, decreases with discharge
        self.w = np.clip(self.w + 0.25 * self.q - 0.1 * discharged, 0.0, 1e6).astype(np.float32)

        # speed proxies
        self.dv = (0.8 * self.dv + self.rng.normal(0.0, 0.3, size=(self._n,))).astype(np.float32)
        self.vbar = np.clip(self.vbar + 0.1 * (service - 0.6) - 0.05 * (self.q / 50.0), 2.0, 20.0).astype(np.float32)
        self.pos_proxy = (self.link_len - (self.vbar + self.dv) * 1.0).astype(np.float32)

        F = self.obs.build(self.q, self.w, self.dirs, self.vbar, self.dv, self.pos_proxy, self.link_len, self.n_lane)

        # rewards: local (Eq. 4) and corridor (Eq. 5) approximations
        local = -(self.q + 0.01 * self.w)  # proxy for q + delay
        pi = 0.02 * self.w  # downstream waiting penalty
        pi[self.entry] = 0.0
        R_net = float(np.mean(local - pi))

        done = self.t >= self.params.episode_horizon
        info = {"t": self.t, "mean_q": float(self.q.mean()), "mean_w": float(self.w.mean())}
        return StepOutput(F=F, A=self.A.copy(), m=self.m.copy(), reward=R_net, done=done, info=info)
