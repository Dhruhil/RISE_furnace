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
├── README.md
├── configs/
│   └── base_config.py              # CONFIG dataclass, REGION_MATERIALS, physics constants
├── data/
│   └── dataset_unified.py          # UnifiedDataset: builds graph + samples (sim, t)
├── models/
│   └── meshgraphnet.py             # HeatTreatmentGNN (PhysicsNeMo MGN + fallback impl)
├── utils/
│   ├── checkpoint.py               # CheckpointManager: best-model + latest tracking
│   ├── logging.py                  # setup_logging, log_metrics (W&B optional)
│   └── metrics.py                  # MAE, R², within-tolerance helpers
├── evaluation/
│   ├── evaluate.py                 # Per-region rollout MAE → JSON (Phase 1 / Phase 2)
│   └── save_rollout_temps.py       # Save T_pred and T_true per timestep → HDF5
├── train_unified.py                # Main training entry point
└── outputs/                        # Checkpoints, logs, evaluation results (created at runtime)
```

---

## 3. Hardware & Software Requirements

### 3.1 Hardware

| Resource              | Recommended (training)              | Minimum to reproduce            |
|-----------------------|-------------------------------------|---------------------------------|
| GPU                   | 1× NVIDIA A100 (80 GB) or A40       | 1× GPU with ≥ 24 GB VRAM        |
| CPU                   | 16 cores                            | 8 cores                         |
| System RAM            | ≥ 64 GB                             | ≥ 32 GB                         |
| Disk (working set)    | ~30 GB (dataset + checkpoints)      | ~20 GB                          |

**GPU notes.** The unified graph holds 13,648 nodes and ~150k edges per sample, so a single forward + pushforward pass keeps VRAM low (~6 GB at `batch=4`). An A40 (48 GB) or RTX 3090 (24 GB) reproduces the run at the same `batch=4`. Below 24 GB, drop to `batch=2` and double the epoch count or accept slower convergence.

**Without a GPU**, the code falls back to CPU automatically, but a full 150-epoch run would take weeks — not recommended.

### 3.2 Software

| Component                | Version used         | Notes                                                |
|--------------------------|----------------------|------------------------------------------------------|
| OS                       | Linux                | Any modern x86-64 Linux works                        |
| CUDA                     | 12.x                 |                                                      |
| Python                   | 3.10                 |                                                      |
| PyTorch                  | 2.x                  | With CUDA support                                    |
| PyTorch Geometric        | latest               | For `Data`, `Batch`, `DataLoader`                    |
| NVIDIA PhysicsNeMo       | 25.06                | Primary MeshGraphNet backend                         |
| DGL                      | latest               | Required by PhysicsNeMo's MGN backend                |
| NumPy, SciPy             | latest               | `cKDTree` for KNN edge construction                  |
| h5py                     | latest               | HDF5 dataset I/O                                     |

The complete software stack can be bundled into a single Apptainer
(Singularity) image (`physicsnemo_25.06.sif`) so that no Python packages
need to be installed on the host. If `physicsnemo` is not importable for
any reason, the model auto-detects this and silently switches to the
in-house `_FallbackMGN` implementation — training still runs with the
same dynamics.

Alternatively, install Python dependencies directly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-physicsnemo dgl
pip install torch-geometric
pip install numpy scipy h5py pyyaml
```

### 3.3 Input dataset

The dataset path is configured in `configs/base_config.py`. The dataset contains **78 OpenFOAM simulation cases** in total, built upstream by `Dataset_creation/scripts/create_all_regions_dataset.py`. Each HDF5 group is one simulation holding `times`, per-region `coords` and `T(t, n_cells)` arrays, plus parameter attributes (`T_set`, `cx`, `cy`, `cz`, `radius`, `height`).

---

## 4. How to Run

### 4.1 Set the dataset path

Open `configs/base_config.py` and point `dataset_path` to your local copy
of the OpenFOAM-derived HDF5 dataset (78 cases). Output directories
(`outputs/`, `outputs/checkpoints`, `outputs/logs`) are created
automatically on the first run.

### 4.2 Train

The reference training command used for the thesis run:

```bash
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u train_unified.py \
    --epochs 150 \
    --lr 5e-5 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/GNN_<run_name>/checkpoints
```

If you have the dependencies installed directly on the host (no Apptainer),
the same flags apply:

```bash
python -u train_unified.py \
    --epochs 150 \
    --lr 5e-5 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/GNN_<run_name>/checkpoints
```

| Flag | Value | Description |
|---|---|---|
| `--epochs` | 150 | Number of training epochs |
| `--lr` | 5e-5 | Base learning rate (AdamW) |
| `--batch` | 4 | Mini-batch size |
| `--lam` | 0.003 | Physics-loss weight λ |
| `--checkpoint_dir` | `outputs/.../checkpoints` | Directory for `best_model.pt`, `best_model_eval_snapshot.pt`, `latest.pt` |

Reference wall-clock on a single NVIDIA A100: ≈ 80 hours for 150 epochs.

Checkpoints land in:

```
outputs/GNN_<run_name>/checkpoints/
    ├── best_model.pt                  # selected on val data-only loss
    ├── best_model_eval_snapshot.pt    # pinned snapshot for paper-quality eval
    └── latest.pt                      # rolling — for resuming
```

### 4.3 Evaluate (autoregressive rollout)

After training, evaluate the best checkpoint on the held-out test cases:

```bash
# Per-region rollout MAE → JSON (Phase 1 / Phase 2)
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u evaluation/evaluate.py \
    --checkpoint outputs/GNN_<run_name>/checkpoints/best_model.pt

# Dump T_pred / T_true HDF5 for thesis figures
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u evaluation/save_rollout_temps.py \
    --checkpoint outputs/GNN_<run_name>/checkpoints/best_model.pt
```

This produces per-region Phase 1 / Phase 2 MAE and R², saved as JSON
under `outputs/GNN_<run_name>/evaluation/`.

### 4.4 Quick sanity check (optional)

To verify the data pipeline and model wiring without running a full training,
launch with a reduced epoch count:

```bash
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u train_unified.py --epochs 1 --batch 4
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
- **Physics λ:** held at 0.003 throughout


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

After running evaluation, the following files appear in `outputs/GNN_<run_name>/evaluation/`:

| File                          | Contents                                                  |
|-------------------------------|-----------------------------------------------------------|
| `gnn_rollout_results.json`    | Per-region Phase 1 / Phase 2 MAE, mean ± std across sims  |
| `rollout_temps.h5`            | `T_pred(t)` and `T_true(t)` per test simulation           |

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
| `batch_size`               | 4       | OK on A100/A40                           |
| `n_epochs`                 | 150     | CLI flag `--epochs` overrides            |
| `learning_rate`            | 5e-5    | CLI flag `--lr` overrides                |
| `dt`                       | 10.0    | Simulation timestep (s)                  |
| `train_time_end`           | 2760.0  | Phase 1 boundary (s)                     |
| `t_total`                  | 3460.0  | Full rollout horizon (s)                 |

Per-region material properties (κ, Cp, ρ) live in the `REGION_MATERIALS` dict at the top of the config — these are the values fed to the physics loss as raw SI quantities.

---

## 8. Reproducing the Thesis Result

```bash
# 1. Full training (≈ 80 h on A100, 150 epochs)
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u train_unified.py \
    --epochs 150 --lr 5e-5 --batch 4 --lam 0.003 \
    --checkpoint_dir outputs/GNN_<run_name>/checkpoints

# 2. Pin the best checkpoint as the eval snapshot
cd outputs/GNN_<run_name>/checkpoints
cp best_model.pt best_model_eval_snapshot.pt

# 3. Evaluate
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u evaluation/evaluate.py \
    --checkpoint outputs/GNN_<run_name>/checkpoints/best_model_eval_snapshot.pt

apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u evaluation/save_rollout_temps.py \
    --checkpoint outputs/GNN_<run_name>/checkpoints/best_model_eval_snapshot.pt
```

---

## 9. Troubleshooting

| Symptom                                              | Cause                                              | Resolution                                                                |
|------------------------------------------------------|----------------------------------------------------|---------------------------------------------------------------------------|
| `ImportError: physicsnemo`                           | Package not installed                              | Use the Apptainer image, or `pip install nvidia-physicsnemo dgl`; fallback MGN runs anyway |
| `Expected 16 node features, got 15`                  | `node_in_features=15` in config                    | Confirm `base_config.py` has `node_in_features: int = 16`                 |
| Phase 2 MAE >> Phase 1 MAE                           | Pushforward weight stayed at 0                     | Check `get_pushforward_weight` — must ramp after 10 % of epochs           |
| OOM at `batch=4`                                     | GPU has < 24 GB VRAM                               | Drop to `batch=2` and double the epoch count                              |

---

