"""
Checkpoint helpers for the DeepONet training loop.

Counterpart to the GNN / FNO checkpoint managers, with a slightly
different on-disk naming convention:
  - best-so-far weights live at  best.pt
  - periodic snapshots land at   epoch_####.pt

Note: the GNN/FNO managers use best_model.pt and
checkpoint_epoch####.pt — kept different here because the DeepONet
runs predate the convention shake-up, and renaming them would
break a stack of evaluation scripts that already point at best.pt.
"""
from __future__ import annotations

from pathlib import Path
import torch


class CheckpointManager:
    """
    Tiny helper that owns the DeepONet checkpoint directory and
    knows how to write best-of-run vs periodic snapshots.

    Stays small on purpose — the training loop tracks the best
    metric itself and decides when to call save_best, so this
    class doesn't carry any of that bookkeeping.
    """

    def __init__(self, ckpt_dir):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _payload(self, model, optimizer, scheduler, epoch, metrics):
        """
        Build the standard checkpoint dict used by both save methods.

        Bundles model, optimiser, and scheduler state so a crashed
        SLURM job can pick up exactly where it left off rather than
        restarting from scratch (which on the cluster can mean
        hours of lost compute).
        """
        return {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            # Some lr schedulers (e.g. raw lambdas) don't expose
            # state_dict — guard so the call site doesn't have to.
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "metrics":   metrics or {},
        }

    def save_best(self, model, optimizer, scheduler, epoch, metrics):
        """
        Overwrite best.pt with the current weights.

        The training loop calls this only when val_mae beats the
        previous best, so this file always reflects the best
        observed checkpoint across the run.
        """
        torch.save(self._payload(model, optimizer, scheduler, epoch, metrics),
                   self.dir / "best.pt")

    def save_epoch(self, model, optimizer, scheduler, epoch, metrics):
        """
        Write a periodic snapshot at this epoch.

        Filename is zero-padded so the files sort correctly in
        directory listings (epoch_0001 ... epoch_0099 ... epoch_0100).
        """
        torch.save(self._payload(model, optimizer, scheduler, epoch, metrics),
                   self.dir / f"epoch_{epoch:04d}.pt")


def load_best(model, path, device):
    """
    Load weights from a checkpoint into an existing model.

    Used by the rollout / evaluation scripts at the start of every
    inference run. The path is expected to be best.pt by default
    but any payload produced by CheckpointManager will work.

    Missing-file case is a warning rather than an error so a quick
    "did training even start?" check doesn't crash the eval script.
    """
    p = Path(path)
    if not p.exists():
        print(f"[WARN] checkpoint not found: {p}")
        return
    ckpt = torch.load(p, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] loaded checkpoint: {p}  (epoch {ckpt.get('epoch', '?')})")