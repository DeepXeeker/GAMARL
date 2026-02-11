from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter


@dataclass
class LogHandles:
    logger: logging.Logger
    tb: SummaryWriter


def make_logger(log_dir: str, name: str = "gamarl") -> LogHandles:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    tb = SummaryWriter(log_dir=log_dir)
    return LogHandles(logger=logger, tb=tb)
