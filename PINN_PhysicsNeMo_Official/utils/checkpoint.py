"""Checkpoint manager for PINN — same interface as GNN/FNO."""
from __future__ import annotations
from pathlib import Path
import torch

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.best_mae = float("inf")
        self.best_epoch = -1
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "best_model.pt")

    def update(self, model, optimizer, scheduler, epoch, metrics) -> bool:
        mae = metrics.get("mae", float("inf"))
        if mae < self.best_mae:
            self.best_mae = mae
            self.best_epoch = epoch
            model.save(self.best_path, epoch,
                       optimizer.state_dict() if optimizer else None,
                       scheduler.state_dict() if scheduler else None,
                       metrics)
            return True
        return False

    def save_periodic(self, model, optimizer, scheduler, epoch, metrics):
        path = str(Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch:05d}.pt")
        model.save(path, epoch,
                   optimizer.state_dict() if optimizer else None,
                   scheduler.state_dict() if scheduler else None,
                   metrics)
