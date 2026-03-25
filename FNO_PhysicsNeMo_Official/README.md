# FNO PhysicsNeMo — Heat Treatment (All Regions)

### Master's Thesis — RISE Research Institutes of Sweden
**Fourier Neural Operator** for temperature prediction across all furnace regions.

---

## Overview

This is the second AI model for the thesis, complementing the MeshGraphNet (GNN).
Uses the official **NVIDIA PhysicsNeMo** 1D Fourier Neural Operator.

The FNO operates in Fourier (spectral) space, learning temperature evolution
patterns through spectral convolutions — ideal for smooth heat transfer fields.

### Architecture

```
Input:  (batch, 4, n_cells) = [T_current, T_set, region_id, time]
                ↓
        1D Fourier Neural Operator
        - Spectral convolutions (24 Fourier modes)
        - 6 FNO layers with residual connections
        - 128-dimensional latent space
                ↓
Output: (batch, 1, n_cells) = [T_next]
```

### Key Difference from GNN

| | GNN (MeshGraphNet) | FNO |
|---|---|---|
| **Approach** | Graph message passing | Spectral convolution |
| **Prediction** | δT per step | T_next directly |
| **Mesh handling** | Unstructured (native) | 1D signal per region |
| **Speed** | ~100x vs OpenFOAM | ~1000x vs OpenFOAM |
| **Strength** | Geometry-aware | Spectral smoothness |

### Dataset

Uses the same **`dataset_all_regions.h5`** (911 MB) as the GNN All Regions model.
No new data generation needed.

```
dataset_all_regions.h5
  case_000/
    times: (401,)
    steel_cylinder/T: (401, ~450)    → FNO input: (4, 450)
    inner_box/T:      (401, ~3000)   → FNO input: (4, 3000)
    heater_1/T:       (401, ~200)    → FNO input: (4, 200)
    ...
```

---

## Project Structure

```
FNO_PhysicsNeMo_Official/
├── configs/
│   └── fno_config.py              # Paths, FNO hyperparameters, Alvis config
├── data/
│   └── dataset.py                 # Reads dataset_all_regions.h5
├── models/
│   ├── fno_model.py               # PhysicsNeMo FNO + PyTorch fallback
│   └── rollout.py                 # Autoregressive rollout (all regions)
├── training/
│   └── scheduler.py               # ReduceLROnPlateau
├── evaluation/
│   └── evaluate.py                # Phase 1 + Phase 2 rollout evaluation
├── inference/
│   └── infer.py                   # Future prediction beyond 4000s
├── utils/
│   ├── metrics.py                 # MAE, RMSE, R² (same as GNN)
│   ├── logging.py                 # Console + file logging
│   └── checkpoint.py              # Best model tracking
├── train.py                       # Main training script
├── run_alvis_fno.sh               # SLURM job script (48h, A100)
└── README.md
```

---

## How to Run on Alvis

```bash
# 1. Upload and extract
cd /mimer/NOBACKUP/groups/revar
tar xzf FNO_PhysicsNeMo_Official.tar.gz

# 2. Verify dataset
ls -lh /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/dataset_all_regions.h5

# 3. Submit training job
cd FNO_PhysicsNeMo_Official
sbatch run_alvis_fno.sh

# 4. Monitor
squeue -u $USER
tail -f outputs/logs/fno_*.log

# 5. After training — check results
cat outputs/evaluation/fno_evaluation.json

# 6. Future prediction
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python inference/infer.py --target_time 8000 --device cuda
```

---

## Expected Results

Training produces Phase 1 (0–3200s) and Phase 2 (3200–4000s) metrics
in the same format as the GNN, enabling direct comparison in the thesis.

---

## Requirements

- NVIDIA PhysicsNeMo container (`physicsnemo_25.06.sif`)
- A100 GPU (Alvis HPC)
- `dataset_all_regions.h5` (shared with GNN project)
