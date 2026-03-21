"""
Logging utilities for GNN PhysicsNeMo heat treatment training.

Provides:
  setup_logging(cfg)  → returns a Python logger, optionally initialises W&B
  log_metrics(...)    → writes one epoch row to the log file + W&B if enabled
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from datetime import datetime


_FORMAT     = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT   = "%H:%M:%S"
_CONFIGURED = False


def setup_logging(cfg) -> logging.Logger:
    """
    Initialise root logger + file handler.
    Optionally starts a W&B run if cfg.use_wandb is True.

    Returns
    -------
    logging.Logger  — named 'heat_gnn'
    """
    global _CONFIGURED

    log_path = Path(cfg.log_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not _CONFIGURED:
        fmt     = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)
        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        # file handler  (appends — survives restart)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)

        root = logging.getLogger()
        root.addHandler(ch)
        root.addHandler(fh)
        root.setLevel(logging.INFO)
        _CONFIGURED = True

    logger = logging.getLogger("heat_gnn")

    # ── Optional W&B ──────────────────────────────────────────────────
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
    Write one epoch summary line to the logger and (optionally) to W&B.

    Parameters
    ----------
    logger      : logger returned by setup_logging()
    epoch       : current epoch number  (1-based)
    train_loss  : average training loss for this epoch
    val_metrics : dict with keys: loss, mae, rmse, r2, within_5K, within_10K
    cfg         : BaseConfig instance (used for use_wandb flag)
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
            pass