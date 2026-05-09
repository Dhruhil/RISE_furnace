"""
LR scheduler builder for the DeepONet training loop.

Single function — separated out so the training script doesn't
need to know which scheduler is in use, and so the choice can be
swapped (e.g. cosine, OneCycle) without touching the loop itself.
"""
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(optimizer, cfg):
    """
    Build a ReduceLROnPlateau scheduler from the config.

    Plateau scheduling is well-suited to this surrogate-training
    setup — the validation MAE plateaus visibly between drops, and
    halving the LR after `cfg.lr_patience` epochs without
    improvement consistently squeezes a bit more accuracy out of
    the tail of training.

    The training loop holds this off until after the LR warmup is
    done so the warmup gets a clean ramp.
    """
    return ReduceLROnPlateau(
        optimizer,
        mode="min",                      # MAE: lower is better
        factor=cfg.lr_decay_factor,      # 0.5 by default — halve the LR on plateau
        patience=cfg.lr_patience,        # 15 epochs of no improvement
    )