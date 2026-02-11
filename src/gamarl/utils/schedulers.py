from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LinearSchedule:
    start: float
    end: float
    duration: int

    def value(self, t: int) -> float:
        if self.duration <= 0:
            return self.end
        frac = min(max(t / self.duration, 0.0), 1.0)
        return self.start + frac * (self.end - self.start)
