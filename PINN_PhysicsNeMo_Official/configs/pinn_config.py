"""
PINN configuration — All Regions, PhysicsNeMo Sym framework.
Master's Thesis: Digital Twin Modeling of Heat Treatment in Cast Metals
                 using OpenFOAM and Physics-Informed AI

Reads dataset_all_regions.h5 for validation/comparison.
The PINN solves the heat equation directly via PDE residual.
Configured for Alvis HPC cluster (C3SE, Chalmers).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

_BASE = "/mimer/NOBACKUP/groups/revar"


@dataclass
class PINNConfig:

    # ── Paths (Alvis) ─────────────────────────────────────────────────
    dataset_path:   str = f"{_BASE}/GNN_PhysicsNeMo_Official/dataset_all_regions.h5"
    output_dir:     str = f"{_BASE}/PINN_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/PINN_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/PINN_PhysicsNeMo_Official/outputs/logs"

    # ── Regions (same 12 as GNN/FNO) ─────────────────────────────────
    all_regions: list = field(default_factory=lambda: [
        "steel_cylinder", "inner_box",
        "heater_1", "heater_2", "heater_3", "heater_4",
        "heater_5", "heater_6", "heater_7", "heater_8",
        "brick_heater", "outer_box",
    ])
    n_regions: int = 12

    # ── Data splits (same as GNN/FNO) ────────────────────────────────
    val_fraction:  float = 0.14
    test_fraction: float = 0.10

    # ── PINN architecture ─────────────────────────────────────────────
    # MLP: input (x, y, z, t, T_set, region_id) → output T
    n_inputs:          int = 6    # x, y, z, t, T_set, region_id
    n_outputs:         int = 1    # T
    hidden_width:      int = 256
    n_hidden_layers:   int = 6    # 6 hidden layers (deep MLP)
    activation:        str = "siren"  # siren or tanh
    omega_0:         float = 30.0     # SIREN frequency (first layer)

    # ── Physics (heat equation PDE) ───────────────────────────────────
    # ρ·Cp·∂T/∂t = κ·∇²T  (conduction — Fourier's law)
    # BC: T_heaters = T_set (Dirichlet on heaters)
    # BC: Q_rad = ε·σ·(T_set⁴ - T⁴) (radiation on steel surface)
    kappa_steel:    float = 25.0     # W/(m·K) thermal conductivity
    rho_steel:      float = 7800.0   # kg/m³ density
    Cp_steel:       float = 450.0    # J/(kg·K) specific heat capacity
    kappa_air:      float = 0.06     # W/(m·K) for air regions
    rho_air:        float = 0.35     # kg/m³
    Cp_air:         float = 1100.0   # J/(kg·K)
    sigma_sb:       float = 5.67e-8  # Stefan-Boltzmann constant
    epsilon_steel:  float = 0.80     # emissivity

    # ── Training ──────────────────────────────────────────────────────
    batch_size:      int   = 16384
    n_epochs_pretrain: int = 3000    # Phase 1: data-only pretraining
    n_epochs_physics:  int = 5000    # Phase 2: physics-informed fine-tuning
    learning_rate:   float = 1e-3
    lr_pretrain:     float = 1e-3
    lr_physics:      float = 5e-4
    weight_decay:    float = 1e-5
    grad_clip:       float = 1.0

    # ── Physics loss curriculum ───────────────────────────────────────
    # Same smooth exponential as GNN/FNO: λ = 0.001 * exp(4.6 * p)
    lambda_max:      float = 0.10

    # ── Time / physics ────────────────────────────────────────────────
    dt:               float = 10.0
    t_total:          float = 4000.0
    train_time_end:   float = 3200.0
    predict_time_end: float = 4000.0

    # ── Logging ───────────────────────────────────────────────────────
    log_every:           int  = 500
    save_every:          int  = 1000
    device:              str  = "cuda"

    @property
    def alpha_steel(self) -> float:
        """Thermal diffusivity of steel [m²/s]."""
        return self.kappa_steel / (self.rho_steel * self.Cp_steel)

    @property
    def alpha_air(self) -> float:
        """Thermal diffusivity of air [m²/s]."""
        return self.kappa_air / (self.rho_air * self.Cp_air)

    @property
    def n_train_steps(self) -> int:
        return int(self.train_time_end / self.dt)

    @property
    def n_total_steps(self) -> int:
        return int(self.t_total / self.dt)

    def __post_init__(self):
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/predictions",
                  self.output_dir + "/evaluation"]:
            Path(p).mkdir(parents=True, exist_ok=True)


CONFIG = PINNConfig()
