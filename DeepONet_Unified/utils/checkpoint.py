"""
utils/checkpoint.py
-------------------
Checkpoint manager. Mirrors GNN_Unified/utils/checkpoint.py.
"""

import os
import torch
import numpy as np


class CheckpointManager:
    def __init__(self, ckpt_dir: str, model, optimizer, scheduler=None):
        self.ckpt_dir   = ckpt_dir
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.best_loss  = float("inf")
        os.makedirs(ckpt_dir, exist_ok=True)

    def save(self, epoch: int, val_loss: float, normalizer: dict, tag: str = ""):
        state = {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "normalizer": normalizer,
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()

        path = os.path.join(self.ckpt_dir, f"ckpt_epoch{epoch:04d}{tag}.pt")
        torch.save(state, path)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            best = os.path.join(self.ckpt_dir, "best_model.pt")
            torch.save(state, best)
            return True
        return False

    def load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        return state.get("epoch", 0), state.get("val_loss", float("inf")), state.get("normalizer", {})

    def load_best(self):
        best = os.path.join(self.ckpt_dir, "best_model.pt")
        if os.path.exists(best):
            return self.load(best)
        raise FileNotFoundError(f"No best model found at {best}")
