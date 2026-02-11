from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

from .base_env import BaseEnv, StepOutput
from .observation_builder import ObservationBuilder


@dataclass
class SumoScenario:
    sumo_binary: str
    sumo_config: str
    network_file: str | None = None
    route_file: str | None = None


class SumoTINEnv(BaseEnv):
    """SUMO environment wrapper via TraCI.

    This implementation is intentionally conservative: it provides the interface and common
    hooks you'll need, but the exact lane/phase mapping is scenario-dependent.

    You should customize:
      - how intersections are enumerated
      - how per-intersection queues/delays are computed
      - how actions map to SUMO traffic light programs

    See SUMO TraCI docs for details.
    """

    def __init__(self, scenario: SumoScenario, n_intersections: int, df: int, action_dim: int, episode_horizon: int):
        self.scenario = scenario
        self._n = int(n_intersections)
        self._df = int(df)
        self._action_dim = int(action_dim)
        self.episode_horizon = int(episode_horizon)

        self.obs = ObservationBuilder(self._n)
        self.t = 0

        self._traci = None
        self.A = np.eye(self._n, dtype=np.float32)  # TODO: build from network topology
        self.m = np.ones((self._n,), dtype=np.float32)

        # static descriptors placeholders
        self.link_len = np.ones((self._n,), dtype=np.float32) * 150.0
        self.n_lane = np.ones((self._n,), dtype=np.float32) * 2.0
        self.dirs = np.zeros((self._n, 4), dtype=np.float32)
        self.dirs[:, 0] = 1.0

        self.entry = 0

    @property
    def n_intersections(self) -> int:
        return self._n

    @property
    def df(self) -> int:
        return self._df

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _ensure_traci(self):
        if self._traci is not None:
            return
        try:
            import traci
        except ImportError as e:
            raise RuntimeError(
                "TraCI not available. Ensure SUMO tools are installed and on PYTHONPATH. "
                "Commonly: export SUMO_HOME=/path/to/sumo && export PYTHONPATH=$SUMO_HOME/tools"
            ) from e
        self._traci = traci

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        self._ensure_traci()
        if seed is not None:
            np.random.seed(int(seed))
        self.t = 0

        # Start SUMO
        cmd = [self.scenario.sumo_binary, "-c", self.scenario.sumo_config, "--start", "--quit-on-end"]
        self._traci.start(cmd)

        # TODO: discover controlled tls IDs and build A from net
        return self._get_obs(), self.A.copy(), self.m.copy(), {"t": self.t}

    def close(self):
        if self._traci is not None:
            try:
                self._traci.close(False)
            except Exception:
                pass

    def _get_obs(self) -> np.ndarray:
        # Very generic proxy: without scenario-specific lane mapping, we cannot compute true queues.
        # Provide safe defaults to keep interface usable.
        q = np.zeros((self._n,), dtype=np.float32)
        w = np.zeros((self._n,), dtype=np.float32)
        vbar = np.ones((self._n,), dtype=np.float32) * 10.0
        dv = np.zeros((self._n,), dtype=np.float32)
        pos_proxy = self.link_len - vbar
        return self.obs.build(q, w, self.dirs, vbar, dv, pos_proxy, self.link_len, self.n_lane)

    def _apply_actions(self, actions: np.ndarray) -> None:
        # TODO: map discrete phase-duration to SUMO TLS logic using traci.trafficlight.
        # For now, we advance time; actions are ignored.
        pass

    def step(self, actions: np.ndarray) -> StepOutput:
        self._ensure_traci()
        self._apply_actions(actions)

        # advance 1 second for decision interval handling is left to scenario config
        self._traci.simulationStep()
        self.t += 1

        F = self._get_obs()

        # reward placeholder; should be computed from queues/waiting-time as in paper
        reward = 0.0

        done = self.t >= self.episode_horizon
        info: Dict[str, Any] = {"t": self.t}
        if done:
            self.close()
        return StepOutput(F=F, A=self.A.copy(), m=self.m.copy(), reward=reward, done=done, info=info)
