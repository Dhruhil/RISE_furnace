"""Checkpoint helpers for GNN_PhysicsNeMo_Official."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger("heat_gnn.checkpoint")


def save_checkpoint(path, model, epoch, optimizer=None, scheduler=None, metrics=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics":         metrics or {},
        "extra":           extra   or {},
    }
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
    mae_str = f"  val_MAE={metrics['mae']:.3f} K" if metrics and "mae" in metrics else ""
    logger.info("Checkpoint saved -> %s  (epoch %d)%s", path, epoch, mae_str)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu", strict=True):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.to(device)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.best_mae       = float("inf")
        self.best_epoch     = -1
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self):
        return str(Path(self.checkpoint_dir) / "best_model.pt")

    def update(self, model, optimizer, scheduler, epoch, metrics):
        mae = metrics.get("mae", float("inf"))
        if mae < self.best_mae:
            self.best_mae   = mae
            self.best_epoch = epoch
            save_checkpoint(self.best_path, model, epoch, optimizer, scheduler, metrics)
            return True
        return False

    def save_periodic(self, model, optimizer, scheduler, epoch, metrics):
        path = str(Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch:04d}.pt")
        save_checkpoint(path, model, epoch, optimizer, scheduler, metrics)
