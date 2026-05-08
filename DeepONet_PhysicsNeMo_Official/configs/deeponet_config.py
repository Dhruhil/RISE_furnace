"""
DeepONet config — mirrors FNO_PhysicsNeMo_Official/configs/fno_config.py
so that all three surrogates (GNN / FNO / DeepONet) share dataset paths,
time windows and split fractions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DeepONetConfig:
    # ── Paths ────────────────────────────────────────────────────────
    dataset_path: str = (
        "/mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official/"
        "dataset_v2_all_regions_clean.h5"
    )
    output_dir:     str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir:        str = "outputs/logs"

    # ── 3D sampling grid for the trunk net ───────────────────────────
    # DeepONet operates on a point cloud, not a dense grid. We still
    # define a bounding box to sample query points uniformly across
    # the furnace domain per batch.
    x_min: float = -0.30; x_max: float = 0.30
    y_min: float = -0.10; y_max: float = 0.50
    z_min: float = -0.10; z_max: float = 0.50

    # Number of query points passed to the trunk per sample
    n_query_points: int = 1024

    # ── Branch net input (sensor / function representation) ─────────
    # A fixed set of sensor locations encodes the current field u(x).
    # We take a coarse regular lattice across the furnace.
    sensor_grid_x: int = 10
    sensor_grid_y: int = 12
    sensor_grid_z: int = 18

    @property
    def n_sensors(self) -> int:
        return self.sensor_grid_x * self.sensor_grid_y * self.sensor_grid_z

    # Channels per sensor:
    #   T_norm, region_id/11, is_heater, kappa/100, Cp/1000, rho/10000
    branch_in_channels: int = 6

    # Extra scalars concatenated to the branch output:
    #   T_set_norm, time/t_total          (2 values)
    branch_scalar_inputs: int = 7  # was 2 (T_set, time); now 7 (+ cx, cy, cz, radius, height)

    # ── Trunk net input ──────────────────────────────────────────────
    # Per query point: (x, y, z, region_id/11, is_heater,
    #                   kappa/100, Cp/1000, rho/10000)
    trunk_in_features: int = 8

    # ── DeepONet architecture ────────────────────────────────────────
    latent_dim:        int = 128            # branch & trunk output dim
    branch_hidden:     List[int] = field(default_factory=lambda: [256, 256, 256])
    trunk_hidden:      List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    activation:        str = "gelu"
    use_bias_decoder:  bool = True          # adds a learnable bias to each prediction

    # ── Training ─────────────────────────────────────────────────────
    batch_size:      int   = 4
    n_epochs:        int   = 200
    learning_rate:   float = 1e-4
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 15
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0

    # ── Data splits ──────────────────────────────────────────────────
    val_fraction:  float = 0.13
    test_fraction: float = 0.10

    # ── Time window (same as FNO / GNN) ──────────────────────────────
    dt:               float = 10.0
    t_total:          float = 3460.0
    train_time_end:   float = 2760.0
    predict_time_end: float = 3460.0

    # ── Physics regularisation (gentle, like FNO) ────────────────────
    lambda_physics:   float = 0.003

    # ── Logging ──────────────────────────────────────────────────────
    log_every_n_epochs:  int = 1
    save_every_n_epochs: int = 10

    @property
    def n_train_steps(self) -> int:
        return int(self.train_time_end / self.dt)

    def __post_init__(self):
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/evaluation",
                  self.output_dir + "/plots",
                  self.output_dir + "/rollout_results"]:
            Path(p).mkdir(parents=True, exist_ok=True)


CONFIG = DeepONetConfig()
