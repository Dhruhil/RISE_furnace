"""
FNO configuration — All Regions dataset.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Reads dataset_all_regions.h5 (same file as GNN All Regions model).
Configured for Alvis HPC cluster (C3SE, Chalmers).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

_BASE = "/mimer/NOBACKUP/groups/revar"


@dataclass
class FNOConfig:

    # ── Paths (Alvis) ─────────────────────────────────────────────────
    dataset_path:   str = f"{_BASE}/GNN_PhysicsNeMo_Official/dataset_all_regions.h5"
    output_dir:     str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/logs"

    # ── Regions ───────────────────────────────────────────────────────
    all_regions: list = field(default_factory=lambda: [
        "steel_cylinder", "inner_box",
        "heater_1", "heater_2", "heater_3", "heater_4",
        "heater_5", "heater_6", "heater_7", "heater_8",
        "brick_heater",
        "outer_box",
    ])
    n_regions: int = 12

    # ── Data splits ───────────────────────────────────────────────────
    val_fraction:  float = 0.14
    test_fraction: float = 0.10

    # ── FNO architecture ──────────────────────────────────────────────
    # 1D FNO: each sample = temperature profile across cells of one region
    # Input channels:  T_current, T_set, region_id, time  → 4
    # Output channels: T_next                              → 1
    fno_in_channels:        int = 4
    fno_out_channels:       int = 1
    fno_modes:              int = 24   # Fourier modes kept in spectral conv
    fno_layers:             int = 6    # number of FNO spectral layers
    fno_latent:             int = 128   # latent channel width
    fno_decoder_layers:     int = 3
    fno_decoder_layer_size: int = 128

    # ── Training ──────────────────────────────────────────────────────
    batch_size:      int   = 8
    n_epochs:        int   = 300
    learning_rate:   float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 20
    weight_decay:    float = 1e-5
    grad_clip:       float = 1.0

    # ── Physics-informed loss (same weights as GNN all-regions) ───
    w_convection:      float = 0.5     # T ≤ T_set  (Newton cooling)
    w_conduction:      float = 0.3     # spectral smoothness (≡ diffusion)
    w_radiation:       float = 0.2     # Stefan-Boltzmann dT constraint
    sigma_sb:          float = 5.67e-8
    epsilon_steel:     float = 0.80
    char_thickness:    float = 0.01

    # ── Time / physics ────────────────────────────────────────────────
    dt:               float = 10.0     # seconds per timestep
    t_total:          float = 4000.0   # total simulation time
    train_time_end:   float = 3200.0   # training window end
    predict_time_end: float = 4000.0   # verification window end

    # ── Logging ───────────────────────────────────────────────────────
    log_every_n_epochs:  int  = 1
    save_every_n_epochs: int  = 10
    device: str = "cuda"

    @property
    def n_train_steps(self) -> int:
        return int(self.train_time_end / self.dt)

    @property
    def n_total_steps(self) -> int:
        return int(self.t_total / self.dt)

    @property
    def n_verify_steps(self) -> int:
        return self.n_total_steps - self.n_train_steps

    def __post_init__(self):
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/predictions",
                  self.output_dir + "/evaluation"]:
            Path(p).mkdir(parents=True, exist_ok=True)


CONFIG = FNOConfig()
