"""
Checkpoint helpers for GNN PhysicsNeMo heat treatment training.

Provides:
    save_checkpoint(...)   → saves model + optimiser + scheduler + metrics
    load_checkpoint(...)   → restores model (and optionally optimiser/scheduler)
    list_checkpoints(dir)  → sorted list of all checkpoint .pt files
    get_best_checkpoint(dir) → path to best_model.pt if it exists
    CheckpointManager      → tracks best model, auto-saves on improvement
                             (imported by training/train.py)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("heat_gnn.checkpoint")


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:      str,
    model:     torch.nn.Module,
    epoch:     int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler                                   = None,
    metrics:   Optional[dict]                  = None,
    extra:     Optional[dict]                  = None,
) -> None:
    """Save a full training checkpoint (model + optimiser + scheduler + metrics)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics":         metrics or {},
        "extra":           extra   or {},
    }

    # Store lightweight model config so the checkpoint is self-describing
    if hasattr(model, "cfg"):
        c = model.cfg
        payload["model_cfg"] = {
            "node_in_features":          c.node_in_features,
            "edge_in_features":          c.edge_in_features,
            "hidden_features":           c.hidden_features,
            "n_message_passing_layers":  c.n_message_passing_layers,
            "output_features":           c.output_features,
        }

    torch.save(payload, path)
    mae_str = (
        f"  val_MAE={metrics['mae']:.3f} K"
        if metrics and "mae" in metrics else ""
    )
    logger.info("Checkpoint saved → %s  (epoch %d)%s", path, epoch, mae_str)


def load_checkpoint(
    path:      str,
    model:     torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler                                   = None,
    device:    str                             = "cpu",
    strict:    bool                            = True,
) -> dict:
    """
    Load a checkpoint into model and optionally optimizer/scheduler.
    Returns the full checkpoint dict (epoch, metrics, extra, …).
    """
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.to(device)

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    epoch   = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    mae_str = (
        f"  val_MAE={metrics['mae']:.3f} K"
        if "mae" in metrics else ""
    )
    logger.info("Checkpoint loaded ← %s  (epoch %d)%s", path, epoch, mae_str)
    return ckpt


def list_checkpoints(checkpoint_dir: str) -> list[Path]:
    """Return all checkpoint_epoch*.pt files sorted by epoch number."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return []
    return sorted(d.glob("checkpoint_epoch*.pt"))


def get_best_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """Return path to best_model.pt if it exists, else None."""
    p = Path(checkpoint_dir) / "best_model.pt"
    return p if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Tracks the best validation MAE and saves checkpoints automatically.

    Used by training/train.py:
        ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)

        for epoch in range(cfg.n_epochs):
            ...
            is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_metrics)
            if epoch % cfg.save_every_n_epochs == 0:
                ckpt_mgr.save_periodic(model, optimizer, scheduler, epoch, val_metrics)

    Attributes:
        best_mae   : best validation MAE seen so far
        best_epoch : epoch at which best_mae was achieved
        best_path  : path to best_model.pt
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.best_mae       = float("inf")
        self.best_epoch     = -1
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "best_model.pt")

    def update(
        self,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch:     int,
        metrics:   dict,
    ) -> bool:
        """
        Check if current epoch is the best so far.
        If yes, save best_model.pt and return True.
        Otherwise return False.
        """
        mae = metrics.get("mae", float("inf"))
        if mae < self.best_mae:
            self.best_mae   = mae
            self.best_epoch = epoch
            save_checkpoint(
                path      = self.best_path,
                model     = model,
                epoch     = epoch,
                optimizer = optimizer,
                scheduler = scheduler,
                metrics   = metrics,
            )
            return True
        return False

    def save_periodic(
        self,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch:     int,
        metrics:   dict,
    ) -> None:
        """Save a periodic checkpoint named checkpoint_epoch{epoch:04d}.pt."""
        path = str(
            Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch:04d}.pt"
        )
        save_checkpoint(
            path      = path,
            model     = model,
            epoch     = epoch,
            optimizer = optimizer,
            scheduler = scheduler,
            metrics   = metrics,
        )