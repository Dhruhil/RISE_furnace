"""
Base config for the unified GNN.
Set up for the Alvis cluster (C3SE).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

# Pick up the base path from env if set, otherwise use the default Mimer location
_BASE = os.environ.get("GNN_BASE_DIR", "/mimer/NOBACKUP/groups/revar")


# Material props pulled from the OpenFOAM thermophysicalProperties dicts.
# kappa = W/m.K, Cp = J/kg.K, rho = kg/m^3
# Note: inner_box is air (fluid). k=0.05 is used as a low-conduction
# placeholder since the GNN cannot really handle convection directly.
REGION_MATERIALS = {
    "steel_cylinder": {"kappa": 80.0,  "Cp": 450.0,  "rho": 7800.0},
    "inner_box":      {"kappa": 0.05,  "Cp": 1000.0, "rho": 1.2},     # air
    "outer_box":      {"kappa": 15.0,  "Cp": 1000.0, "rho": 867.0},
    "brick_heater":   {"kappa": 8.0,   "Cp": 450.0,  "rho": 7800.0},
    "heater_1":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_2":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_3":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_4":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_5":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_6":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_7":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_8":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
}

# -----------------------------------------------------------------------
# Physical constants used by the physics-informed loss
# -----------------------------------------------------------------------
SIGMA_SB = 5.67e-8           # Stefan-Boltzmann, W/(m^2*K^4)
EMISSIVITY_STEEL = 0.80      # taken for oxidized steel surfaces
H_CONV = 25.0                # natural convection coefficient (air, hot surface)
CHAR_THICKNESS = 0.0167      # V/A for the cylinder (r=50mm, h=100mm), see Incropera 2011


@dataclass
class BaseConfig:
    # ---- paths ---------------------------------------------------------
    # h5 dataset for cylinder-only runs (kept around for older experiments)
    dataset_path:   str = f"{_BASE}/dataset_cylinder_features.h5"
    # main dataset that has all 12 regions, used for the unified GNN
    all_regions_dataset_path: str = f"{_BASE}/GNN_PhysicsNeMo_Official/dataset_v2_all_regions_clean.h5"

    # outputs go under one folder so things stay tidy on Mimer
    output_dir:     str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/GNN_PhysicsNeMo_Official/outputs/logs"

    # ---- features ------------------------------------------------------
    # Node features fed into the GNN: 3D position, time, setpoint,
    # cylinder centroid, geometry stuff, and per-cell material props.
    feature_cols: list = field(default_factory=lambda: [
        "x", "y", "z", "t", "T_set", "cx", "cy", "cz",
        "radius", "height", "volume", "mass", "kappa", "Cp", "rho",
    ])

    # The 12 OpenFOAM regions (order matters for the cellToRegion mapping)
    all_regions: list = field(default_factory=lambda: [
        "steel_cylinder", "inner_box", "outer_box",
        "heater_1", "heater_2", "heater_3", "heater_4",
        "heater_5", "heater_6", "heater_7", "heater_8",
        "brick_heater",
    ])
    n_regions: int = 12

    target_col: str = "T"   # the model predicts temperature
    n_features:  int = 15   # length of feature_cols

    # ---- splits --------------------------------------------------------
    # 13% val, 10% test, rest goes to training. Stratified by T_set.
    val_fraction:  float = 0.13
    test_fraction: float = 0.10

    # ---- graph build ---------------------------------------------------
    # k-NN gave better results than a fixed radius for this geometry.
    # 12 neighbors is a sweet spot — fewer and the message passing
    # gets too local, more and far-away regions start to mix.
    graph_k_neighbors: int = 12
    use_radius_graph:  bool = False

    # ---- model dims ----------------------------------------------------
    node_in_features:         int = 16   # 15 feats + 1 region one-hot helper
    node_in_features_allregions: int = 7  # smaller variant for the all-regions setup
    edge_in_features:         int = 5     # dx, dy, dz, dist, region-pair flag
    hidden_features:          int = 128
    n_message_passing_layers: int = 4     # 4 was enough, 6 did not help
    output_features:          int = 1     # just the temperature delta

    # ---- training ------------------------------------------------------
    batch_size:      int   = 4            # GPU memory was tight at 6
    n_epochs:        int   = 200
    learning_rate:   float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 20           # ReduceLROnPlateau patience
    weight_decay:  float = 1e-5
    grad_clip:       float = 1.0          # had a couple of spikes early on, helps

    # ---- time stepping -------------------------------------------------
    dt:               float = 10.0        # OpenFOAM dump interval
    t_total:          float = 3460.0      # full rollout horizon
    train_time_end:   float = 2760.0      # cut training at Phase-1 boundary
    predict_time_end: float = 3460.0      # rollout goes to t_total

    # ---- physics loss constants (duplicated locally for convenience) ---
    sigma_sb:       float = 5.67e-8
    epsilon_steel:  float = 0.80

    # NOTE: the loss weights below are currently unused.
    # Actual weights are hardcoded in physics_loss_unified(): 0.5 / 0.3 / 0.15 / 0.05.
    # Kept here in case they need to be exposed later.
    w_conduction:   float = 0.3
    w_convection:   float = 0.5
    w_radiation:    float = 0.3
    char_thickness: float = 0.01

    # ---- logging / checkpointing --------------------------------------
    log_every_n_epochs:  int  = 1
    save_every_n_epochs: int  = 10

    # wandb is off by default — turn it on for production runs only
    use_wandb:           bool = False
    wandb_project:       str  = "heat-treatment-gnn"
    wandb_run_name:      str  = "unified_gnn_alvis"

    # ---- runtime -------------------------------------------------------
    device:      str = "cuda"
    num_workers: int = 0      # zero workers is more stable on Alvis

    # -------------------------------------------------------------------
    # Derived properties (computed from dt and the time bounds above)
    # -------------------------------------------------------------------
    @property
    def n_train_steps(self) -> int:
        # number of timesteps inside the Phase-1 (in-distribution) window
        return int(self.train_time_end / self.dt)

    @property
    def n_total_steps(self) -> int:
        # total timesteps across Phase 1 + Phase 2
        return int(self.t_total / self.dt)

    @property
    def n_verify_steps(self) -> int:
        # Phase-2 (verification / extrapolation) horizon
        return self.n_total_steps - self.n_train_steps

    def __post_init__(self):
        # make sure the output dirs exist before training kicks off
        for p in [
            self.checkpoint_dir, self.log_dir,
            self.output_dir + "/predictions",
            self.output_dir + "/plots",
            self.output_dir + "/verification",
        ]:
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except OSError:
                # mkdir can fail on some shared FS, that's fine — keep going
                pass


CONFIG = BaseConfig()