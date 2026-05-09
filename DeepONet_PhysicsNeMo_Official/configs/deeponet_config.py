"""
DeepONet config — mirrors the FNO and GNN configs so all three
surrogates share the same dataset paths, time windows, and
train/val/test split fractions. Keeping these in sync is what
makes the comparison numbers in the thesis tables fair.

DeepONet is structured differently from the other two: instead
of operating on a mesh (GNN) or a voxel grid (FNO), it learns the
operator T(x, t) -> T(x, t+dt) point-by-point. Two networks make
that work — a branch net that encodes the current field from a
fixed set of sensor locations, and a trunk net that maps the
query coordinates into the same latent space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DeepONetConfig:
    # ---- paths ---------------------------------------------------------
    # Same HDF5 dataset as the FNO and GNN, just under a different
    # project root on Mimer. Output dirs are relative to wherever the
    # training script is launched, so each SLURM job can keep its
    # own outputs without clobbering anyone else's.
    dataset_path: str = (
        "/mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official/"
        "dataset_v2_all_regions_clean.h5"
    )
    output_dir:     str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir:        str = "outputs/logs"

    # ---- 3D sampling box for the trunk net -----------------------------
    # DeepONet works on a point cloud — there's no dense grid like the
    # FNO has. The bounding box below defines the volume that query
    # points get sampled from per batch.
    x_min: float = -0.30; x_max: float = 0.30
    y_min: float = -0.10; y_max: float = 0.50
    z_min: float = -0.10; z_max: float = 0.50

    # Number of query points fed to the trunk per sample. 1024 was
    # the largest value that still fit comfortably in GPU memory at
    # batch_size=4 on the Alvis A100s.
    n_query_points: int = 1024

    # ---- branch net input (sensor / function representation) ----------
    # The current temperature field u(x) gets encoded by sampling at
    # a fixed set of sensor locations — a coarse 10x12x18 lattice
    # across the furnace volume. The branch net never sees the full
    # mesh, only this compressed sensor view.
    sensor_grid_x: int = 10
    sensor_grid_y: int = 12
    sensor_grid_z: int = 18

    @property
    def n_sensors(self) -> int:
        return self.sensor_grid_x * self.sensor_grid_y * self.sensor_grid_z

    # Per-sensor channels (mirrors what each FNO voxel carries minus
    # the time scalar, which is appended via branch_scalar_inputs):
    #   T_norm, region_id/11, is_heater, kappa/100, Cp/1000, rho/10000
    branch_in_channels: int = 6

    # Extra scalar fields concatenated onto the branch output before
    # the inner product with the trunk. Currently:
    #   T_set_norm, time/t_total, cx, cy, cz, radius, height  -> 7
    # The cx/cy/cz/radius/height entries pass the per-sim cylinder
    # geometry through, since otherwise the branch can't tell two
    # cases at the same T_set apart from the sensor field alone.
    branch_scalar_inputs: int = 7

    # ---- trunk net input ----------------------------------------------
    # Per query point: spatial coordinates plus the same per-cell
    # static fields the FNO and GNN both use, so all three surrogates
    # see equivalent local context.
    #   (x, y, z, region_id/11, is_heater, kappa/100, Cp/1000, rho/10000)
    trunk_in_features: int = 8

    # ---- DeepONet architecture ----------------------------------------
    # Branch and trunk both project to the same latent_dim, which is
    # what lets the final inner product produce a scalar prediction.
    # Slightly deeper trunk than branch because the trunk has to
    # learn a richer per-point basis.
    latent_dim:        int = 128
    branch_hidden:     List[int] = field(default_factory=lambda: [256, 256, 256])
    trunk_hidden:      List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    activation:        str = "gelu"
    # Learnable scalar bias on the inner product — small effect, but
    # cheap to add and makes the predictor more flexible at the
    # expense of one extra parameter.
    use_bias_decoder:  bool = True

    # ---- training ------------------------------------------------------
    # Lower LR than the FNO since DeepONet's inner-product predictor
    # is sensitive to optimiser noise — too high and the latent
    # representations drift, too low and convergence stalls.
    batch_size:      int   = 4
    n_epochs:        int   = 200
    learning_rate:   float = 1e-4
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 15
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0

    # ---- data splits --------------------------------------------------
    # Match the FNO and GNN pipelines so the held-out cases line up
    # across architectures.
    val_fraction:  float = 0.13
    test_fraction: float = 0.10

    # ---- time stepping -------------------------------------------------
    # Match the FNO and GNN: rollout horizon 3460 s, training cuts
    # at the Phase-1 boundary at 2760 s, everything after is the
    # temporal-extrapolation window.
    dt:               float = 10.0
    t_total:          float = 3460.0
    train_time_end:   float = 2760.0
    predict_time_end: float = 3460.0

    # ---- physics regularisation ---------------------------------------
    # Gentle weight, same idea as the FNO — the OpenFOAM data
    # already encodes the full physics, so this term just nudges
    # the predictions toward energy-balance consistency.
    lambda_physics:   float = 0.003

    # ---- logging / checkpointing --------------------------------------
    log_every_n_epochs:  int = 1
    save_every_n_epochs: int = 10

    @property
    def n_train_steps(self) -> int:
        # Number of timesteps inside the Phase-1 (in-distribution) window
        return int(self.train_time_end / self.dt)

    def __post_init__(self):
        # Make sure all the output dirs exist before training kicks off
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/evaluation",
                  self.output_dir + "/plots",
                  self.output_dir + "/rollout_results"]:
            Path(p).mkdir(parents=True, exist_ok=True)


CONFIG = DeepONetConfig()