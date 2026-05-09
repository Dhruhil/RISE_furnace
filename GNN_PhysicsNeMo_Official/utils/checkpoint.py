"""
Checkpoint helpers for GNN_PhysicsNeMo_Official.

Wraps torch.save / torch.load with a couple of conveniences:
  - bundles model_cfg into the checkpoint so it can be reloaded
    later without needing the exact original config file
  - a small CheckpointManager that tracks the best-so-far validation
    MAE and overwrites best_model.pt whenever a new low is hit
  - logs through the heat_gnn.checkpoint logger instead of print so
    the long Alvis training logs stay searchable
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger("heat_gnn.checkpoint")


def save_checkpoint(path, model, epoch, optimizer=None, scheduler=None, metrics=None, extra=None):
    """
    Dump the full training state to a single .pt file.

    Saves the model weights along with optimiser and scheduler state
    so training can resume from exactly where it left off if a job
    gets killed (which happens on shared clusters more often than
    one would like).
    """
    # Make sure the target directory exists — saves a head-scratch
    # later when a job dies on the very first checkpoint write.
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics":         metrics or {},
        "extra":           extra   or {},
    }

    # Bundle the model's structural config too, so a checkpoint
    # is self-contained — no need to keep the exact original
    # base_config.py around to reload it later.
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
    """
    Restore weights (and optionally optimiser/scheduler state) from
    a checkpoint produced by save_checkpoint().

    `strict=False` is sometimes useful when loading older checkpoints
    after a small architecture change — e.g. when a new feature gets
    added to the node encoder and the old weights are missing for it.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.to(device)

    # Only restore optimiser / scheduler state when both the object
    # and the saved state are available — keeps the function usable
    # for inference-only loads (where no optimiser exists).
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


class CheckpointManager:
    """
    Tiny helper that keeps track of the best-so-far validation MAE
    and overwrites best_model.pt whenever a new minimum is reached.

    Periodic checkpoints (one per N epochs) are kept separately
    under checkpoint_epoch####.pt — useful for plotting training
    dynamics or rolling back to an earlier weight set.
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        # Start at +inf so any finite MAE on epoch 1 counts as an improvement
        self.best_mae       = float("inf")
        self.best_epoch     = -1
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self):
        # Single canonical location for the best checkpoint —
        # the eval/rollout scripts default to looking here.
        return str(Path(self.checkpoint_dir) / "best_model.pt")

    def update(self, model, optimizer, scheduler, epoch, metrics):
        """
        Check the latest validation MAE and overwrite best_model.pt
        if it's a new low. Returns True when a new best is saved,
        False otherwise — handy for early-stopping logic.
        """
        mae = metrics.get("mae", float("inf"))
        if mae < self.best_mae:
            self.best_mae   = mae
            self.best_epoch = epoch
            save_checkpoint(self.best_path, model, epoch, optimizer, scheduler, metrics)
            return True
        return False

    def save_periodic(self, model, optimizer, scheduler, epoch, metrics):
        """
        Drop a regular checkpoint at this epoch, regardless of the
        validation MAE. Filename includes the epoch index so the
        files sort nicely on disk.
        """
        path = str(Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch:04d}.pt")
        save_checkpoint(path, model, epoch, optimizer, scheduler, metrics)