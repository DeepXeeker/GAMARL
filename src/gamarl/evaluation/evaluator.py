from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from ..metrics.traffic_metrics import TrafficMetrics


@dataclass
class EvalResult:
    metrics: dict


class Evaluator:
    def __init__(self, env, model, device: torch.device, history_len: int):
        self.env = env
        self.model = model
        self.device = device
        self.history_len = history_len

    @torch.no_grad()
    def run(self, episodes: int = 5) -> EvalResult:
        stats = []
        for ep in range(episodes):
            F, A, m, _ = self.env.reset(seed=ep)
            metrics = TrafficMetrics()

            done = False
            while not done:
                Ft = torch.tensor(F, dtype=torch.float32, device=self.device).unsqueeze(0)
                At = torch.tensor(A, dtype=torch.float32, device=self.device).unsqueeze(0)
                mt = torch.tensor(m, dtype=torch.float32, device=self.device).unsqueeze(0)
                F_hist = Ft.unsqueeze(2).repeat(1, 1, self.history_len, 1)

                Q, _ = self.model(F_hist, Ft, At, mt)
                act = torch.argmax(Q, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)

                out = self.env.step(act)
                F, A, m = out.F, out.A, out.m
                done = out.done

                metrics.update_proxy(out.info.get("mean_q", 0.0), out.info.get("mean_w", 0.0))

            stats.append(metrics.compute())

        # mean
        mean = {
            "avg_waiting_time": float(np.mean([s.avg_waiting_time for s in stats])),
            "avg_travel_time": float(np.mean([s.avg_travel_time for s in stats])),
            "avg_stops": float(np.mean([s.avg_stops for s in stats])),
        }
        return EvalResult(metrics=mean)
