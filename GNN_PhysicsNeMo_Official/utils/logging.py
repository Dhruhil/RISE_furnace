"""
Logging utilities for the GNN PhysicsNeMo heat-treatment training runs.

Two helpers live here:
  setup_logging(cfg)  -> returns a Python logger, optionally fires up a W&B run
  log_metrics(...)    -> writes one epoch row to the log file (and W&B, if on)

The split between console and file handlers is intentional. Console
output is what shows up on the Alvis terminal during a live run, and
the file handler keeps everything around for post-mortem inspection
once the SLURM job has finished.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from datetime import datetime


# Shared format strings — kept at module level so both handlers
# write identical-looking lines and grep'ing across console + file
# logs stays simple.
_FORMAT     = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT   = "%H:%M:%S"

# Guard flag — handlers only get attached once even if setup_logging
# is called multiple times (e.g. from a notebook re-import).
_CONFIGURED = False


def setup_logging(cfg) -> logging.Logger:
    """
    Set up the root logger with both console and file handlers, and
    optionally start a Weights & Biases run when cfg.use_wandb is True.

    Returns
    -------
    logging.Logger
        Named 'heat_gnn' — every other module just grabs its own
        sub-logger off this one (e.g. 'heat_gnn.checkpoint').
    """
    global _CONFIGURED

    # The training log lives next to the checkpoints so a single
    # output_dir holds everything from a given run.
    log_path = Path(cfg.log_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not _CONFIGURED:
        fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

        # Console handler — what shows up live on stdout while
        # the SLURM job is running.
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)

        # File handler — appended to (mode="a") so a crashed-and-
        # resumed run keeps a continuous log instead of clobbering
        # the previous attempt.
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)

        root = logging.getLogger()
        root.addHandler(ch)
        root.addHandler(fh)
        root.setLevel(logging.INFO)
        _CONFIGURED = True

    logger = logging.getLogger("heat_gnn")

    # ---- optional W&B run ------------------------------------------
    # W&B is off by default — only flip it on for production runs
    # that are worth tracking long-term.
    if getattr(cfg, "use_wandb", False):
        try:
            import wandb
            wandb.init(
                project = cfg.wandb_project,
                name    = cfg.wandb_run_name,
                config  = {
                    "epochs":         cfg.n_epochs,
                    "lr":             cfg.learning_rate,
                    "batch_size":     cfg.batch_size,
                    "hidden":         cfg.hidden_features,
                    "n_layers":       cfg.n_message_passing_layers,
                    "node_in":        cfg.node_in_features,
                    "edge_in":        cfg.edge_in_features,
                    "k_neighbors":    cfg.graph_k_neighbors,
                    "weight_decay":   cfg.weight_decay,
                    "grad_clip":      cfg.grad_clip,
                },
            )
            logger.info("W&B run started: %s / %s",
                        cfg.wandb_project, cfg.wandb_run_name)
        except ImportError:
            # Don't crash the run just because wandb is missing on
            # the cluster — drop a warning and carry on.
            logger.warning("wandb not installed — skipping W&B logging.")

    logger.info("Log file: %s", log_path)
    return logger


def log_metrics(
    logger:      logging.Logger,
    epoch:       int,
    train_loss:  float,
    val_metrics: dict,
    cfg,
) -> None:
    """
    Write one epoch summary line to the logger and (optionally) push
    the same numbers to W&B.

    Parameters
    ----------
    logger      : logger returned by setup_logging()
    epoch       : current epoch number (1-based)
    train_loss  : average training loss for this epoch
    val_metrics : dict with keys: loss, mae, rmse, r2, within_5K, within_10K
    cfg         : BaseConfig instance (used for the use_wandb flag)
    """
    # One-line summary that gets greppable by epoch later. Keys that
    # might not be in val_metrics fall back to NaN so the format
    # string never blows up mid-training.
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

    # Same numbers, different sink. Only push if W&B was actually
    # initialised — the wandb.run check covers the case where
    # cfg.use_wandb=True but the import failed earlier.
    if getattr(cfg, "use_wandb", False):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch":            epoch,
                    "train/loss":       train_loss,
                    "val/loss":         val_metrics.get("loss",       0.0),
                    "val/mae_K":        val_metrics.get("mae",        0.0),
                    "val/rmse_K":       val_metrics.get("rmse",       0.0),
                    "val/r2":           val_metrics.get("r2",         0.0),
                    "val/within_5K":    val_metrics.get("within_5K",  0.0),
                    "val/within_10K":   val_metrics.get("within_10K", 0.0),
                }, step=epoch)
        except ImportError:
            # Same fallback as above — silently skip if wandb isn't
            # importable, since the metrics already went to the file
            # logger by the time this branch runs.
            pass