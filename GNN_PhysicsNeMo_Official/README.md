# GNN_PhysicsNeMo_Official — Unified Multi-Region Graph Neural Network

**Master's Thesis** — *Simulating Heat Treatment of Cast Metal Products using OpenFOAM and AI*

A physics-informed MeshGraphNet built on **NVIDIA PhysicsNeMo** that predicts the transient temperature field of a cylindrical steel component undergoing heat treatment inside an industrial furnace. Unlike per-region surrogates, this model treats all 12 furnace regions as **one unified graph**, so the network learns cross-region heat transfer (conduction, convection, radiation) end-to-end.

---

## 1. Overview

The model is an autoregressive next-step predictor:

```
T(t)  ──▶  GNN  ──▶  T(t + Δt)
```

Each training sample is a single graph snapshot containing all 13,648 nodes of one OpenFOAM simulation at one timestep. Edges connect nodes within a region (k-nearest neighbours) and across physically adjacent regions (boundary edges, ≤ 2 cm threshold). At inference, the model is rolled out autoregressively for the full 3460 s horizon.

**Key features**

- Single unified graph per timestep covering all 12 regions (steel cylinder, inner air cavity, eight heaters, brick heater, outer box)
- 16 node features: coordinates, current T, T_set, region id, time, geometry (cx, cy, cz, radius, height), and per-region material properties (κ, Cp, ρ)
- 5 edge features: relative position (3), distance, edge type (intra-region vs boundary)
- MeshGraphNet backbone: 4 message-passing layers, 128 hidden units (~0.70 M parameters)
- Physics-informed loss: Fourier conduction + Newton convection + Stefan-Boltzmann radiation + energy balance
- Region-weighted data loss (steel ×10, inner_box ×3, outer_box ×0.1)
- Pushforward training for rollout stability
- Heater nodes are clamped to T_set (boundary conditions, never predicted)

---

## 2. Folder Layout

```
GNN_PhysicsNeMo_Official/
├── configs/
│   └── base_config.py              # CONFIG dataclass, REGION_MATERIALS, physics constants
├── data/
│   └── dataset_unified.py          # UnifiedDataset: builds graph + samples (sim, t)
├── models/
│   └── meshgraphnet.py             # HeatTreatmentGNN (PhysicsNeMo MGN + fallback impl)
├── utils/
│   ├── checkpoint.py               # CheckpointManager: best-model + latest tracking
│   ├── logging.py                  # setup_logging, log_metrics (W&B optional)
│   └── metrics.py                  # MAE, RMSE, R², within-tolerance helpers
├── evaluation/
│   ├── evaluate.py                 # Per-region rollout MAE → JSON (Phase 1 / Phase 2)
│   └── save_rollout_temps.py       # Save T_pred and T_true per timestep → HDF5
├── train_unified.py                # Main training entry point
│
│   ─── SBATCH scripts ───
├── run_alvis_gnn_v5_FIX.sh         # ★ FINAL TRAINING SCRIPT — use this for the thesis run
├── run_sanity_test.sh              # 1-batch smoke test before launching training
├── run_eval_gnn_v5_snapshot.sh     # Final evaluation against the v5_FIX snapshot
├── run_save_temps_gnn.sh           # Dump rollout temperatures for thesis figures
│
└── outputs/                        # Checkpoints, logs, evaluation results (created at runtime)
```

---

## 3. Hardware & Software Requirements

### 3.1 Hardware

The thesis run was performed on the **C3SE Alvis** HPC cluster.

| Resource                    | Used for the thesis run            | Minimum to reproduce            |
|-----------------------------|------------------------------------|---------------------------------|
| GPU                         | 1× NVIDIA A100fat (80 GB HBM2e)    | 1× GPU with ≥ 24 GB VRAM        |
| CPU                         | 16 cores (1 SLURM task)            | 8 cores                         |
| System RAM                  | ~64 GB available to the job        | ≥ 32 GB                         |
| Disk (working set)          | ~30 GB (dataset + checkpoints)     | ~20 GB                          |
| Wall-clock — training       | ~80 h (150 epochs, A100fat)        | depends on GPU                  |
| Wall-clock — sanity test    | ~5 min on A40                      | —                               |
| Wall-clock — evaluation     | ~30 min on A40                     | —                               |

**GPU notes.** The unified graph holds 13,648 nodes and ~150k edges per sample, so a single forward + pushforward pass keeps VRAM low (~6 GB at `batch=4`). The 80 GB A100fat was chosen for fast wall-clock, not for memory; an A40 (48 GB) or RTX 3090 (24 GB) reproduces the run at the same `batch=4`. Below 24 GB, drop to `batch=2` and double the epoch count or accept slower convergence.

**Without a GPU**, the code falls back to CPU automatically, but a full 150-epoch run would take weeks — not recommended.

### 3.2 Software

| Component                | Version used         | Notes                                                |
|--------------------------|----------------------|------------------------------------------------------|
| OS (cluster nodes)       | Linux (Alvis)        | Any modern x86-64 Linux works                        |
| CUDA                     | 12.x (in the sif)    | Provided by the Apptainer image                      |
| Python                   | 3.10                 | Provided by the image                                |
| PyTorch                  | 2.x                  | With CUDA support                                    |
| PyTorch Geometric        | latest in image      | For `Data`, `Batch`, `DataLoader`                    |
| NVIDIA PhysicsNeMo       | 25.06                | Primary MeshGraphNet backend                         |
| DGL                      | latest in image      | Required by PhysicsNeMo's MGN backend                |
| NumPy, SciPy             | latest               | `cKDTree` for KNN edge construction                  |
| h5py                     | latest               | HDF5 dataset I/O                                     |
| Apptainer / Singularity  | ≥ 1.1                | To exec the `.sif` image                             |
| SLURM                    | any recent version   | All `.sh` files are SBATCH scripts                   |

The complete software stack is bundled in a single Apptainer image:

```
/mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif
```

All `.sh` scripts in this folder execute training/evaluation inside that image via `apptainer exec --nv ... <script>`, so **you do not need to install any Python packages on the host node**. If `physicsnemo` is not importable for any reason, the model auto-detects this and silently switches to the in-house `_FallbackMGN` implementation — training still runs with the same dynamics.

### 3.3 Input dataset

Path is resolved from `configs/base_config.py`:

```
/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/dataset_v2_all_regions_clean.h5
```

Built upstream by `Dataset_creation/scripts/create_all_regions_dataset.py`. The dataset contains **78 OpenFOAM simulation cases** in total. Each HDF5 group is one simulation holding `times`, per-region `coords` and `T(t, n_cells)` arrays, plus parameter attributes (`T_set`, `cx`, `cy`, `cz`, `radius`, `height`).

---

## 4. Running on Alvis (SLURM)

All `.sh` files are SBATCH scripts targeting Alvis (project `NAISS2026-4-712`).

### 4.1 Sanity test (always first)

```bash
sbatch run_sanity_test.sh
```

Loads one batch, runs forward + physics-loss + backward. Verifies dataset, graph construction, and model wiring. Approx. 5 min on an A40. Always run this before launching the full training.

### 4.2 Full training — the main run

```bash
sbatch run_alvis_gnn_v5_FIX.sh
```

**This is the final training script for the thesis.** It trains for 150 epochs on an A100fat with:

| Setting          | Value         |
|------------------|---------------|
| Epochs           | 150           |
| Learning rate    | 5e-5          |
| Batch size       | 4             |
| Physics λ        | 0.003         |
| GPU              | A100fat (1×)  |
| Wall time budget | 88 h          |

Checkpoints land in:

```
outputs/GNN_v5_FIX_150ep_<timestamp>/checkpoints/
    ├── best_model.pt                  # selected on val data-only loss
    ├── best_model_eval_snapshot.pt    # pinned snapshot for paper-quality eval
    └── latest.pt                      # rolling — for resuming
```

The script verifies two source patches before launching, so it refuses to run if either `data/dataset_unified.py` (missing `_parse_mm`) or `train_unified.py` (missing `args.checkpoint_dir`) has reverted to a pre-fix state.

### 4.3 Evaluation

```bash
sbatch run_eval_gnn_v5_snapshot.sh        # full per-region rollout MAE → JSON
sbatch run_save_temps_gnn.sh              # dump T_pred / T_true HDF5 for figures
```

---

## 5. Training Details

### 5.1 Loss

```
L_total = (1 − λ) · L_data + λ · L_phys
```

**Data loss** is region-weighted MSE plus pushforward (predict step 2 from step-1's output to reduce rollout drift):

```
L_data = MSE(pred₁, T(t+Δt)) + w₂ · MSE(pred₂, T(t+2Δt))
```

with weights `steel=10×`, `inner_box=3×`, `outer_box=0.1×`. Heater nodes are masked out (clamped to T_set, never predicted).

**Physics loss** combines four residuals from the energy equation `ρ·Cp·dT/dt = κ·∇²T + h·(T_set − T)/δ + ε·σ·(T_set⁴ − T⁴)/δ`:

| Term     | Weight | Physics                              |
|----------|--------|--------------------------------------|
| `L_cond` | 0.4    | Fourier conduction (graph Laplacian) |
| `L_conv` | 0.3    | Newton convection                    |
| `L_rad`  | 0.2    | Stefan-Boltzmann radiation           |
| `L_eng`  | 0.1    | Combined energy balance              |

The convection term also includes an overshoot penalty `ReLU(T_pred − T_set)` to prevent the model from predicting temperatures above the heater setpoint.

### 5.2 Schedules

- **LR warmup:** 5 epochs from `0.1·lr → lr`
- **Pushforward weight:** zero for the first 10 % of epochs, then linearly ramped to 1.0
- **Physics λ:** held at 0.003 throughout (CLI flag `--lam` in `run_alvis_gnn_v5_FIX.sh`)

### 5.3 Splits (T_set-stratified)

A fixed shuffle splits the **78 cases** into train / val / test, stratified across the three setpoints (T_set = 1173 K, 1273 K, 1373 K). Normalisation statistics (`T_mean`, `T_std`, `dT_mean`, `dT_std`) are computed on the training split only.

---

## 6. Results

Rollout evaluation across the held-out test simulations. **Phase 1** is the in-distribution horizon the model was trained on (0 – 2760 s); **Phase 2** is the extrapolation horizon, 700 s beyond training (2760 – 3460 s). Heater regions are excluded — they are clamped to `T_set` and never predicted. *Best values per column in bold.*

|       |          | **Steel cylinder**         |||| **Overall non-heater**     ||||
|       |          | **Phase 1**     || **Phase 2**     || **Phase 1**     || **Phase 2**     ||
| Model | Params   | MAE   | R²       | MAE    | R²      | MAE   | R²       | MAE    | R²      |
|-------|----------|-------|----------|--------|---------|-------|----------|--------|---------|
| GNN   | 0.70 M   | **2.06** | **0.9996** | **31.28** | **< 0** | **2.71** | **0.9996** | **17.61** | **0.9577** |

MAE is reported in Kelvin (K).

### 6.1 Output artefacts

After running evaluation, the following files appear in `outputs/GNN_v5_FIX_150ep_<timestamp>/evaluation/`:

| File                          | Contents                                                  |
|-------------------------------|-----------------------------------------------------------|
| `gnn_rollout_results.json`    | Per-region Phase 1 / Phase 2 MAE, mean ± std across sims  |
| `rollout_temps.h5`            | `T_pred(t)` and `T_true(t)` per test simulation           |
| `logs/eval_<jobid>.log`       | Full SLURM log of the evaluation run                      |

The JSON file has this top-level structure:

```json
{
  "checkpoint":   "outputs/.../best_model_eval_snapshot.pt",
  "phase1_end_s": 2760,
  "phase2_end_s": 3460,
  "summary": {
    "steel_cylinder": {"p1_mae_mean": ..., "p1_mae_std": ..., "p2_mae_mean": ..., "p2_mae_std": ...},
    "inner_box":      {...},
    "outer_box":      {...}
  },
  "per_sim": {"sim_<i>": {...}}
}
```

---

## 7. Configuration

All hyperparameters live in `configs/base_config.py`. The most useful ones to tune:

| Field                      | Default | Description                              |
|----------------------------|---------|------------------------------------------|
| `hidden_features`          | 128     | MLP / message hidden width               |
| `n_message_passing_layers` | 4       | Number of MGN blocks                     |
| `graph_k_neighbors`        | 12      | KNN intra-region edge count              |
| `batch_size`               | 4       | OK on A100fat                            |
| `n_epochs`                 | 150     | CLI flag `--epochs` overrides            |
| `learning_rate`            | 5e-5    | CLI flag `--lr` overrides                |
| `dt`                       | 10.0    | Simulation timestep (s)                  |
| `train_time_end`           | 2760.0  | Phase 1 boundary (s)                     |
| `t_total`                  | 3460.0  | Full rollout horizon (s)                 |

Per-region material properties (κ, Cp, ρ) live in the `REGION_MATERIALS` dict at the top of the config — these are the values fed to the physics loss as raw SI quantities.

---

## 8. Reproducing the Thesis Result

```bash
# 1. Smoke test
sbatch run_sanity_test.sh

# 2. Full training (≈ 80 h on A100fat) — the main run
sbatch run_alvis_gnn_v5_FIX.sh

# 3. Pin the best checkpoint as the eval snapshot
cd outputs/GNN_v5_FIX_150ep_<timestamp>/checkpoints
cp best_model.pt best_model_eval_snapshot.pt

# 4. Evaluate
sbatch run_eval_gnn_v5_snapshot.sh
sbatch run_save_temps_gnn.sh
```

---

## 9. Troubleshooting

| Symptom                                              | Cause                                              | Resolution                                                                |
|------------------------------------------------------|----------------------------------------------------|---------------------------------------------------------------------------|
| `ImportError: physicsnemo`                           | Image not loaded or wrong sif                      | Confirm `physicsnemo_25.06.sif` is being used; fallback MGN runs anyway   |
| `run_alvis_gnn_v5_FIX.sh` aborts with "patch missing"| Stale `dataset_unified.py` or `train_unified.py`   | Re-pull the v5_FIX patch — script greps for `_parse_mm` and `args.checkpoint_dir` |
| `Expected 16 node features, got 15`                  | `node_in_features=15` in config                    | Confirm `base_config.py` has `node_in_features: int = 16`                 |
| Phase 2 MAE >> Phase 1 MAE                           | Pushforward weight stayed at 0                     | Check `get_pushforward_weight` — must ramp after 10 % of epochs           |

---

## 10. Notes

- **`run_alvis_gnn_v5_FIX.sh` is the canonical training script.** All other `.sh` files are auxiliary (sanity test, evaluation, figure data dump).
- **Heater clamping.** Predictions for heater regions are masked to zero in the loss and overwritten with the ground-truth `T_set` during rollout. This treats heaters as Dirichlet boundary conditions, mirroring the OpenFOAM setup.
- **Edge normalisation.** Edge distances are divided by their dataset-wide mean for training stability — same scaling applied at evaluation.
- **Two-step pushforward.** During training the model also predicts step `t+2Δt` from its own step `t+Δt` output (no gradient flow through the first prediction). This drastically reduces compounding rollout error.
- **Backend transparency.** The model autodetects `physicsnemo` and falls back to a pure-PyTorch implementation; both produce equivalent training dynamics on the test set within numerical noise.