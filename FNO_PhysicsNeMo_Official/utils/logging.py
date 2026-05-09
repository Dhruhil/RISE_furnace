"""
Logging utilities for the FNO training runs.

Two helpers live here:
  setup_logging(cfg)  -> returns a Python logger with both console
                         and file handlers attached
  log_metrics(...)    -> writes one epoch summary line to the logger

Mirror of utils/logging.py from the GNN pipeline. Kept separate so
the FNO log lands in train_fno.log instead of train.log, which makes
side-by-side log inspection easier when both architectures are
training back-to-back on the cluster.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


# Shared format strings — kept at module level so both handlers
# write identical-looking lines and grep'ing across console + file
# logs stays simple.
_FORMAT   = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def setup_logging(cfg) -> logging.Logger:
    """
    Set up the FNO logger with both console and file handlers.

    Returns the 'heat_fno' logger directly. propagate=False keeps
    its messages out of the root logger so any other module that
    happens to use the standard logging stack (e.g. PyTorch's own
    debug logs) doesn't get tangled in with the training output.
    """
    # The training log lives next to the FNO checkpoints so a
    # single output_dir holds everything from a given run.
    log_path = Path(cfg.log_dir) / "train_fno.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)
    logger = logging.getLogger("heat_fno")
    logger.setLevel(logging.INFO)

    # Guard against double-attach — setup_logging can get called
    # more than once (e.g. when a notebook re-imports the module),
    # and adding handlers a second time would duplicate every line.
    if not logger.handlers:
        # Console handler — what shows up live on stdout while
        # the SLURM job is running.
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        # File handler — appended to (mode="a") so a crashed-and-
        # resumed run keeps a continuous log instead of clobbering
        # the previous attempt.
        fh = logging.FileHandler(str(log_path), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)

    # Don't bubble up to the root logger — keeps the training output
    # uncluttered by anything other modules might be logging.
    logger.propagate = False
    logger.info("Log file: %s", log_path)
    return logger


def log_metrics(logger, epoch, train_loss, val_metrics, cfg) -> None:
    """
    Write one epoch summary line to the logger.

    Missing keys in val_metrics fall back to NaN so the format
    string never blows up mid-training — handy when an early epoch
    skips part of the validation pipeline (e.g. when the physics
    branch is still warming up).
    """
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