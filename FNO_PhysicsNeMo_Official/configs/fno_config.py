"""
3D FNO configuration.

All 12 regions get interpolated onto a single regular Cartesian
grid before training. The FNO operates on this voxel grid in
Fourier space, which is its natural home — but the resampling
loses some of the OpenFOAM mesh's geometric fidelity, especially
near material interfaces. That trade-off is one of the things
the thesis quantifies in Section 5.3.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

_BASE = "/mimer/NOBACKUP/groups/revar"


# -----------------------------------------------------------------------
# Physical constants used by the physics-informed loss
# -----------------------------------------------------------------------
SIGMA_SB = 5.67e-8           # Stefan-Boltzmann, W/(m^2*K^4)
EMISSIVITY_STEEL = 0.80      # taken for oxidized steel surfaces
H_CONV = 25.0                # natural convection coefficient (air, hot surface)
CHAR_THICKNESS = 0.0167      # V/A for the cylinder (r=50mm, h=100mm), see Incropera 2011


@dataclass
class FNOConfig:
    # ---- paths ---------------------------------------------------------
    # Same HDF5 dataset that the GNN uses, just re-rasterised to
    # a regular grid inside the FNO dataloader.
    dataset_path:   str = f"{_BASE}/FNO_PhysicsNeMo_Official/dataset_v2_all_regions_clean.h5"
    output_dir:     str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/logs"

    # ---- regions -------------------------------------------------------
    # Same 12 regions as the GNN — kept in the same order so region
    # IDs line up across the two pipelines.
    all_regions: list = field(default_factory=lambda: [
        "steel_cylinder", "inner_box",
        "heater_1", "heater_2", "heater_3", "heater_4",
        "heater_5", "heater_6", "heater_7", "heater_8",
        "brick_heater", "outer_box",
    ])
    n_regions: int = 12

    # ---- 3D voxel grid -------------------------------------------------
    # Furnace bounding box is roughly 0.206 x 0.36 x 0.39 m, so a
    # 30x36x54 grid lands at ~9 mm per voxel in each direction.
    # Resolution was bumped twice during development; this is the
    # smallest setting that still captured the cylinder gradient.
    grid_x: int = 30   # ~8.6mm resolution along x
    grid_y: int = 36   # ~9.0mm resolution along y
    grid_z: int = 54   # ~8.9mm resolution along z

    # Furnace bounds (taken straight from the .geo file). The y_min
    # offset of 0.06 m and the z_min offset of 0.005 m skip the
    # outer-enclosure padding so the grid sits tight around the
    # actual cavity geometry.
    x_min: float = 0.0;   x_max: float = 0.206
    y_min: float = 0.06;  y_max: float = 0.30
    z_min: float = 0.005; z_max: float = 0.385

    # ---- FNO architecture ---------------------------------------------
    # Input channels (one voxel feature each):
    #   T_norm, T_set_norm, region_id/11, time,
    #   is_heater, kappa/100, Cp/1000, rho/10000  -> 8 total
    # The single region_id channel is used instead of 12 one-hot
    # masks, which keeps the input tensor small at 30x36x54.
    fno_in_channels:  int = 8
    fno_out_channels: int = 1                    # normalised delta_T
    fno_modes:        list = field(default_factory=lambda: [15, 18, 27])  # per-dim Fourier modes
    fno_layers:       int = 3                    # spectral conv blocks
    fno_latent:       int = 32                   # channel width inside the blocks
    fno_decoder_layers:     int = 2
    fno_decoder_layer_size: int = 32

    # ---- training ------------------------------------------------------
    batch_size:      int   = 4
    n_epochs:        int   = 100
    learning_rate:   float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 20
    weight_decay:    float = 0.0                 # FNO trained fine without it
    grad_clip:       float = 1.0

    # ---- data splits --------------------------------------------------
    # Match the GNN pipeline so the held-out cases line up across
    # architectures and the comparison stays apples-to-apples.
    val_fraction:  float = 0.13
    test_fraction: float = 0.10

    # ---- time stepping -------------------------------------------------
    # Match the GNN: rollout horizon 3460 s, training cuts at the
    # Phase-1 boundary at 2760 s (everything after that is the
    # temporal-extrapolation window).
    dt:               float = 10.0
    t_total:          float = 3460.0
    train_time_end:   float = 2760.0
    predict_time_end: float = 3460.0

    # ---- logging / checkpointing --------------------------------------
    log_every_n_epochs:  int  = 1
    save_every_n_epochs: int  = 10

    @property
    def n_train_steps(self) -> int:
        # Number of timesteps inside the Phase-1 (in-distribution) window
        return int(self.train_time_end / self.dt)

    def __post_init__(self):
        # Make sure the output dirs exist before training kicks off
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/evaluation"]:
            Path(p).mkdir(parents=True, exist_ok=True)


CONFIG = FNOConfig()