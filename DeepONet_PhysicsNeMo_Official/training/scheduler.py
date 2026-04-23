from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(optimizer, cfg):
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_decay_factor,
        patience=cfg.lr_patience,
    )
