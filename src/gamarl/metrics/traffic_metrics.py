from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    avg_waiting_time: float
    avg_travel_time: float
    avg_stops: float


class TrafficMetrics:
    """Metric aggregator.

    For SUMO you should compute these from vehicle trajectories.
    For the synthetic env, we produce proxy metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.wait_sum = 0.0
        self.travel_sum = 0.0
        self.stops_sum = 0.0
        self.steps = 0

    def update_proxy(self, mean_q: float, mean_w: float):
        # proxy: waiting increases with w, travel with q, stops with q and variability
        self.wait_sum += mean_w
        self.travel_sum += mean_q * 5.0 + mean_w * 0.1
        self.stops_sum += min(5.0, 0.05 * mean_q)
        self.steps += 1

    def compute(self) -> EpisodeMetrics:
        if self.steps == 0:
            return EpisodeMetrics(0.0, 0.0, 0.0)
        return EpisodeMetrics(
            avg_waiting_time=self.wait_sum / self.steps,
            avg_travel_time=self.travel_sum / self.steps,
            avg_stops=self.stops_sum / self.steps,
        )
