"""Experiment logger writing to logs/<experiment>.log and to the console."""

from __future__ import annotations

import logging as _std_logging
import sys

from . import ROOT

LOG_DIR = ROOT / "logs"


def get_logger(experiment: str, console: bool = True) -> _std_logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = _std_logging.getLogger(f"iers.{experiment}")
    if logger.handlers:
        return logger
    logger.setLevel(_std_logging.INFO)
    logger.propagate = False
    fmt = _std_logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = _std_logging.FileHandler(LOG_DIR / f"{experiment}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if console:
        stream = _std_logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    return logger
