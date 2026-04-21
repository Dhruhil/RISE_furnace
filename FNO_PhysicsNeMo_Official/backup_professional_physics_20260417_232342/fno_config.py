"""
3D FNO configuration.
All 12 regions interpolated onto one regular grid.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

_BASE = "/mimer/NOBACKUP/groups/revar"

@dataclass
class FNOConfig:
    # Paths
    dataset_path:   str = f"{_BASE}/FNO_PhysicsNeMo_Official/dataset_all_regions_clean.h5"
    output_dir:     str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/logs"

    # Regions
    all_regions: list = field(default_factory=lambda: [
        "steel_cylinder", "inner_box",
        "heater_1", "heater_2", "heater_3", "heater_4",
        "heater_5", "heater_6", "heater_7", "heater_8",
        "brick_heater", "outer_box",
    ])
    n_regions: int = 12

    # 3D grid resolution (furnace: x=0.206, y=0.36, z=0.39)
    grid_x: int = 30   # ~8.6mm resolution
    grid_y: int = 36   # ~9.0mm resolution
    grid_z: int = 54   # ~8.9mm resolution

    # Furnace bounds (from .geo file)
    x_min: float = 0.0;   x_max: float = 0.206
    y_min: float = 0.06;  y_max: float = 0.30
    z_min: float = 0.005; z_max: float = 0.385

    # FNO architecture
    # Input channels: T_norm, T_set_norm, region_id/11, time,
    #                 is_heater, kappa/100, Cp/1000, rho/10000 = 8
    # (single region_id channel instead of 12 one-hot masks)
    fno_in_channels:  int = 8
    fno_out_channels: int = 1    # delta_T normalised
    fno_modes:        list = field(default_factory=lambda: [15, 18, 27])  # modes per dim
    fno_layers:       int = 3
    fno_latent:       int = 32
    fno_decoder_layers:     int = 2
    fno_decoder_layer_size: int = 32

    # Training
    batch_size:      int   = 4
    n_epochs:        int   = 100
    learning_rate:   float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 20
    weight_decay:    float = 0.0
    grad_clip:       float = 1.0

    # Data splits
    val_fraction:  float = 0.14
    test_fraction: float = 0.10

    # Time
    dt:               float = 10.0
    t_total:          float = 3460.0
    train_time_end:   float = 2760.0
    predict_time_end: float = 3460.0

    # Logging
    log_every_n_epochs:  int  = 1
    save_every_n_epochs: int  = 10

    @property
    def n_train_steps(self) -> int:
        return int(self.train_time_end / self.dt)

    def __post_init__(self):
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/evaluation"]:
            Path(p).mkdir(parents=True, exist_ok=True)

CONFIG = FNOConfig()
