from __future__ import annotations

from pathlib import Path
import torch


class CheckpointManager:
    def __init__(self, ckpt_dir):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _payload(self, model, optimizer, scheduler, epoch, metrics):
        return {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "metrics":   metrics or {},
        }

    def save_best(self, model, optimizer, scheduler, epoch, metrics):
        torch.save(self._payload(model, optimizer, scheduler, epoch, metrics),
                   self.dir / "best.pt")

    def save_epoch(self, model, optimizer, scheduler, epoch, metrics):
        torch.save(self._payload(model, optimizer, scheduler, epoch, metrics),
                   self.dir / f"epoch_{epoch:04d}.pt")


def load_best(model, path, device):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] checkpoint not found: {p}")
        return
    ckpt = torch.load(p, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] loaded checkpoint: {p}  (epoch {ckpt.get('epoch', '?')})")
