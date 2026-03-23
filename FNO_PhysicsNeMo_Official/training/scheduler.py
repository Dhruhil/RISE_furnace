"""
Learning rate scheduler for FNO training.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI
"""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(optimizer, cfg) -> ReduceLROnPlateau:
    """ReduceLROnPlateau — reduces LR when validation loss plateaus."""
    return ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = cfg.lr_decay_factor,
        patience = cfg.lr_patience,
        min_lr   = 1e-6,
    )
