"""
Logging utilities for FNO training.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT   = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def setup_logging(cfg) -> logging.Logger:
    """Initialise logger with console + file output."""
    log_path = Path(cfg.log_dir) / "train_fno.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)
    logger = logging.getLogger("heat_fno")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        fh = logging.FileHandler(str(log_path), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.propagate = False
    logger.info("Log file: %s", log_path)
    return logger


def log_metrics(logger, epoch, train_loss, val_metrics, cfg) -> None:
    """Write one epoch summary to logger."""
    logger.info(
        "Epoch %4d | train=%.5f | val=%.5f | MAE=%.2fK | "
        "R2=%.4f | W5K=%.1f%% | W10K=%.1f%%",
        epoch,
        train_loss,
        val_metrics.get("loss",       float("nan")),
        val_metrics.get("mae",        float("nan")),
        val_metrics.get("r2",         float("nan")),
        val_metrics.get("within_5K",  float("nan")),
        val_metrics.get("within_10K", float("nan")),
    )
