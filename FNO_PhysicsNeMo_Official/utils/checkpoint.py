"""
Checkpoint manager for FNO training.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Tracks best model, auto-saves on improvement.
"""
from __future__ import annotations

from pathlib import Path
import torch


class CheckpointManager:
    """Track best validation MAE and save checkpoints automatically."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.best_mae       = float("inf")
        self.best_epoch     = -1
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "best_model.pt")

    def update(self, model, optimizer, scheduler, epoch, metrics) -> bool:
        """Save if current epoch is best. Returns True if saved."""
        mae = metrics.get("mae", float("inf"))
        if mae < self.best_mae:
            self.best_mae   = mae
            self.best_epoch = epoch
            model.save(
                self.best_path, epoch,
                optimizer.state_dict() if optimizer else None,
                scheduler.state_dict() if scheduler else None,
                metrics,
            )
            return True
        return False

    def save_periodic(self, model, optimizer, scheduler, epoch, metrics):
        """Save a periodic checkpoint."""
        path = str(Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch:04d}.pt")
        model.save(
            path, epoch,
            optimizer.state_dict() if optimizer else None,
            scheduler.state_dict() if scheduler else None,
            metrics,
        )
