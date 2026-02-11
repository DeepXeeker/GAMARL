from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


def save_report(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
