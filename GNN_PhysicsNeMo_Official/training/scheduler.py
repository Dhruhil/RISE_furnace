"""
Learning rate scheduler for heat treatment GNN training.

FIX: Removed 'verbose' parameter from ReduceLROnPlateau.
It was deprecated and removed in newer PyTorch versions (>=2.2),
causing: TypeError: ReduceLROnPlateau.__init__() got an unexpected
keyword argument 'verbose'
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg,
) -> ReduceLROnPlateau:
    """
    Build a ReduceLROnPlateau learning rate scheduler.

    Monitors validation loss — reduces LR by factor when no improvement
    for patience epochs. Prevents overshooting near convergence.

    Args:
        optimizer : Adam optimizer from training setup
        cfg       : BaseConfig with lr_decay_factor and lr_patience

    Returns:
        ReduceLROnPlateau — call scheduler.step(val_loss) each epoch
    """
    return ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = cfg.lr_decay_factor,
        patience = cfg.lr_patience,
        min_lr   = 1e-6,
        # NOTE: 'verbose' removed — deprecated in PyTorch >= 2.2
    )