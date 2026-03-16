"""
Learning rate scheduler for heat treatment GNN training.

Uses ReduceLROnPlateau:
  - Monitors validation loss each epoch
  - If val loss does not improve for `patience` epochs, multiply LR by `factor`
  - This prevents over-shooting once the model is close to a good solution

Example with default config values:
  Initial LR  = 0.001
  After 20 epochs no improvement  → LR = 0.001 * 0.5 = 0.0005
  After another 20 no improvement → LR = 0.0005 * 0.5 = 0.00025
  ... and so on down to min_lr = 1e-6
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

    Args:
        optimizer : the Adam optimizer from training setup
        cfg       : BaseConfig with lr_decay_factor and lr_patience fields

    Returns:
        ReduceLROnPlateau scheduler — call scheduler.step(val_loss) each epoch
    """
    return ReduceLROnPlateau(
        optimizer,
        mode      = "min",        # reduce LR when monitored metric stops decreasing
        factor    = cfg.lr_decay_factor,   # multiply LR by this when patience exceeded
        patience  = cfg.lr_patience,       # number of epochs with no improvement before reducing
        min_lr    = 1e-6,          # never go below this learning rate
        verbose   = False,         # set True if you want LR changes printed
    )