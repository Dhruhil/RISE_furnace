"""
configs/base_config.py
----------------------
Central configuration for DeepONet_Unified.
Mirrors GNN_Unified/configs/base_config.py exactly —
only model-specific hyperparameters differ.
"""

from dataclasses import dataclass, field
from typing import List

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "/mimer/NOBACKUP/groups/revar/jinis/deeponet_project/data/dataset_v2_all_regions_clean.h5"
OUTPUT_DIR   = "/mimer/NOBACKUP/groups/revar/jinis/deeponet_project/DeepONet_Unified/outputs"
CKPT_DIR     = f"{OUTPUT_DIR}/checkpoints_unified"
LOG_DIR      = f"{OUTPUT_DIR}/logs"
PLOT_DIR     = f"{OUTPUT_DIR}/plots"
PRED_DIR     = f"{OUTPUT_DIR}/predictions"
VERIF_DIR    = f"{OUTPUT_DIR}/verification"

# ── Dataset ───────────────────────────────────────────────────────────────────
TARGET_REGIONS = ["inner_box", "outer_box", "steel_cylinder", "brick_heater"]
ALL_REGIONS    = [
    "inner_box", "outer_box", "steel_cylinder", "brick_heater",
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
]
HEATER_REGIONS = [f"heater_{i}" for i in range(1, 9)]

# Branch input: case-level parameters
# [T_set, cx, cy, cz, radius, height, kappa_steel, Cp, rho, kappa_brick,
#  heater_1_T_mean, ..., heater_8_T_mean]  → 18 values
BRANCH_INPUT_DIM = 18

# Trunk input: (x, y, z, t) + one-hot region_id (4 target regions)
TRUNK_INPUT_DIM  = 8   # 4 coords + 4 one-hot

# Output: temperature T at query point
OUTPUT_DIM = 1

# Train / val / test split (by case, stratified by T_set)
TRAIN_RATIO = 0.77
VAL_RATIO   = 0.13
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# Max timesteps to sample per case per region during dataset loading
MAX_SAMPLES_PER_CASE = 500

# ── Model (DeepONet) ──────────────────────────────────────────────────────────
LATENT_DIM  = 256   # branch and trunk output dimension (dot-product size)
LAYER_SIZE  = 512   # hidden layer width
NR_LAYERS   = 6     # number of hidden layers in branch and trunk nets

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS      = 200
BATCH_SIZE  = 4096
LR          = 5e-5
WEIGHT_DECAY= 1e-5
GRAD_CLIP   = 1.0

# Physics loss coefficient (ramped via curriculum in training/train.py)
LAMBDA_PHYSICS = 0.003

# LR scheduler
LR_PATIENCE  = 20
LR_FACTOR    = 0.5
LR_MIN       = 1e-6

# Logging
LOG_EVERY    = 100    # steps
SAVE_EVERY   = 10     # epochs

# ── Evaluation ────────────────────────────────────────────────────────────────
# Two-phase evaluation split (seconds)
PHASE1_END   = 2760   # training window
PHASE2_START = 2760   # verification window
PHASE2_END   = 3600

# ── Region material properties (for physics loss) ─────────────────────────────
REGION_MATERIALS = {
    "steel_cylinder": {"kappa": 80.0,  "Cp": 450.0, "rho": 7800.0},
    "inner_box":      {"kappa": 0.026, "Cp": 1005.0,"rho": 1.2   },
    "outer_box":      {"kappa": 1.0,   "Cp": 800.0, "rho": 1600.0},
    "brick_heater":   {"kappa": 8.0,   "Cp": 800.0, "rho": 1800.0},
    **{f"heater_{i}": {"kappa": 20.0,  "Cp": 500.0, "rho": 7000.0}
       for i in range(1, 9)},
}

# Stefan-Boltzmann constant
SIGMA = 5.67e-8
# Steel emissivity
EPSILON_STEEL = 0.8
# Characteristic thickness for radiation loss term (V/A ratio)
CHAR_THICKNESS = 1.67e-2
