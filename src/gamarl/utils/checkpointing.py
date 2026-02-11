from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class CheckpointPaths:
    latest: str
    best: str


def make_ckpt_paths(out_dir: str) -> CheckpointPaths:
    os.makedirs(out_dir, exist_ok=True)
    return CheckpointPaths(
        latest=os.path.join(out_dir, "latest.pt"),
        best=os.path.join(out_dir, "best.pt"),
    )


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: str | None = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
