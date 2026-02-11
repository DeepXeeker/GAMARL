from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Curriculum:
    enabled: bool
    m0: int
    add_every_episodes: int

    def active_count(self, episode: int, max_m: int) -> int:
        if not self.enabled:
            return max_m
        inc = episode // max(1, self.add_every_episodes)
        return min(max_m, self.m0 + inc)
