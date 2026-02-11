from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .traffic_metrics import EpisodeMetrics


@dataclass
class RunningStats:
    n: int = 0
    wait: float = 0.0
    travel: float = 0.0
    stops: float = 0.0

    def add(self, m: EpisodeMetrics):
        self.n += 1
        self.wait += m.avg_waiting_time
        self.travel += m.avg_travel_time
        self.stops += m.avg_stops

    def mean(self) -> EpisodeMetrics:
        if self.n == 0:
            return EpisodeMetrics(0.0, 0.0, 0.0)
        return EpisodeMetrics(self.wait / self.n, self.travel / self.n, self.stops / self.n)
