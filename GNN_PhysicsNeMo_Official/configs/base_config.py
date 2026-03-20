"""
Base configuration — Option A temporal split.
Configured for Alvis HPC cluster (C3SE).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# Alvis path
_BASE = "/mimer/NOBACKUP/groups/revar"


@dataclass
class BaseConfig:

    # ── Paths (Alvis) ─────────────────────────────────────────────────────────
    dataset_path:   str = f"{_BASE}/dataset_cylinder_features.h5"
    output_dir:     str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs/logs"

    feature_cols: list = field(default_factory=lambda: [
        "x", "y", "z", "t",
        "T_set", "cx", "cy", "cz",
        "radius", "height", "volume", "mass",
        "kappa", "Cp", "rho",
    ])
    target_col: str = "T"
    n_features:  int = 15

    val_fraction:  float = 0.14
    test_fraction: float = 0.10

    graph_k_neighbors: int  = 16
    use_radius_graph:  bool = False

    node_in_features:         int = 10
    edge_in_features:         int = 4
    hidden_features:          int = 128
    n_message_passing_layers: int = 15
    output_features:          int = 1

    batch_size:      int   = 4
    n_epochs:        int   = 200
    learning_rate:   float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 20
    weight_decay:    float = 1e-5
    grad_clip:       float = 1.0

    dt:               float = 10.0
    t_total:          float = 4000.0
    train_time_end:   float = 3200.0
    predict_time_end: float = 4000.0

    sigma_sb:       float = 5.67e-8
    epsilon_steel:  float = 0.80
    w_conduction:   float = 0.3
    w_convection:   float = 0.5
    w_radiation:    float = 0.3
    char_thickness: float = 0.01

    log_every_n_epochs:  int  = 1
    save_every_n_epochs: int  = 10
    use_wandb:           bool = False
    wandb_project:       str  = "heat-treatment-gnn"
    wandb_run_name:      str  = "meshgraphnet_optionA_alvis_A100"

    device:      str = "cuda"
    num_workers: int = 0       # A100 on Alvis — use 4 workers

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
        for p in [
            self.checkpoint_dir,
            self.log_dir,
            self.output_dir + "/predictions",
            self.output_dir + "/plots",
            self.output_dir + "/verification",
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)


CONFIG = BaseConfig()
