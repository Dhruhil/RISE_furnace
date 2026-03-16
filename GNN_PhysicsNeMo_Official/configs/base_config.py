"""
Base configuration for GNN PhysicsNeMo heat treatment surrogate.

Your exact container path:
  root@c1c025623cd5:/workspace/rise_furnace/
    Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/
    GNN_PhysicsNeMo_Official/

Dataset is at:
  /workspace/rise_furnace/
    Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/
    Dataset_creation/dataset_cylinder_features.h5

Dataset split (50 LHS simulations):
  TRAIN : 38 sims  (76%)
  VAL   :  7 sims  (14%)
  TEST  :  5 sims  (10%)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ── Exact project root inside your container ────────────────────────────────
_BASE = (
    "/workspace/rise_furnace/"
    "Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM"
)


@dataclass
class BaseConfig:

    # ─── Paths (all using your exact container mount) ────────────────────────
    dataset_path: str = (
        f"{_BASE}/Dataset_creation/dataset_cylinder_features.h5"
    )
    output_dir:     str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs/logs"

    # ─── Feature layout ──────────────────────────────────────────────────────
    feature_cols: list = field(default_factory=lambda: [
        "x", "y", "z", "t",
        "T_set", "cx", "cy", "cz",
        "radius", "height", "volume", "mass",
        "kappa", "Cp", "rho",
    ])
    target_col: str = "T"
    n_features:  int = 15

    # ─── Split: 50 sims → Train 38 | Val 7 | Test 5 ─────────────────────────
    val_fraction:  float = 0.14
    test_fraction: float = 0.10

    # ─── Graph ───────────────────────────────────────────────────────────────
    graph_k_neighbors: int  = 16
    use_radius_graph:  bool = False

    # ─── Model ───────────────────────────────────────────────────────────────
    node_in_features:         int = 10
    edge_in_features:         int = 4
    hidden_features:          int = 128
    n_message_passing_layers: int = 15
    output_features:          int = 1

    # ─── Training ────────────────────────────────────────────────────────────
    batch_size:      int   = 4
    n_epochs:        int   = 200
    learning_rate:   float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 20
    weight_decay:    float = 1e-5
    grad_clip:       float = 1.0

    # ─── Temporal ────────────────────────────────────────────────────────────
    rollout_train_steps: int   = 1
    rollout_noise_std:   float = 0.003
    dt:                  float = 10.0
    t_start:             float = 0.0
    t_end:               float = 4000.0
    rollout_extra_steps: int   = 0

    # ─── Logging ─────────────────────────────────────────────────────────────
    log_every_n_epochs:  int  = 1
    save_every_n_epochs: int  = 10
    use_wandb:           bool = False
    wandb_project:       str  = "heat-treatment-gnn"
    wandb_run_name:      str  = "meshgraphnet_50sims"

    # ─── Device ──────────────────────────────────────────────────────────────
    device:      str = "cuda"
    num_workers: int = 4

    def __post_init__(self):
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/predictions",
                  self.output_dir + "/plots"]:
            Path(p).mkdir(parents=True, exist_ok=True)


CONFIG = BaseConfig()