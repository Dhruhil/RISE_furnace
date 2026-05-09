"""
Checkpoint manager for FNO training.

Tiny helper that keeps track of the best-so-far validation MAE
and overwrites best_model.pt whenever a new minimum is reached.
Periodic checkpoints (one per N epochs) are kept separately under
checkpoint_epoch####.pt — useful for plotting training dynamics
or rolling back to an earlier weight set.

Mirror of utils/checkpoint.py from the GNN pipeline, kept separate
here because the FNO model owns its own .save() method (it bundles
the FNO-specific backend tag into the checkpoint payload, which
the generic save_checkpoint() in the GNN utils doesn't do).
"""
from __future__ import annotations

from pathlib import Path
import torch


class CheckpointManager:
    """
    Track the best validation MAE seen so far and save checkpoints
    automatically when a new minimum is hit.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        # Start at +inf so any finite MAE on epoch 1 counts as
        # an improvement.
        self.best_mae       = float("inf")
        self.best_epoch     = -1
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> str:
        # Single canonical location for the best checkpoint —
        # the eval/rollout scripts default to looking here.
        return str(Path(self.checkpoint_dir) / "best_model.pt")

    def update(self, model, optimizer, scheduler, epoch, metrics) -> bool:
        """
        Check the latest validation MAE and overwrite best_model.pt
        if it's a new low. Returns True when a new best was saved,
        False otherwise — handy for early-stopping logic in the
        training loop.
        """
        mae = metrics.get("mae", float("inf"))
        if mae < self.best_mae:
            self.best_mae   = mae
            self.best_epoch = epoch
            # Delegate the actual write to the model so the
            # checkpoint payload includes the FNO-specific backend
            # tag and any other model-side metadata.
            model.save(
                self.best_path, epoch,
                optimizer.state_dict() if optimizer else None,
                scheduler.state_dict() if scheduler else None,
                metrics,
            )
            return True
        return False

    def save_periodic(self, model, optimizer, scheduler, epoch, metrics):
        """
        Drop a regular checkpoint at this epoch, regardless of the
        validation MAE. Filename includes the epoch index so the
        files sort nicely on disk.
        """
        path = str(Path(self.checkpoint_dir) / f"checkpoint_epoch{epoch:04d}.pt")
        model.save(
            path, epoch,
            optimizer.state_dict() if optimizer else None,
            scheduler.state_dict() if scheduler else None,
            metrics,
        )