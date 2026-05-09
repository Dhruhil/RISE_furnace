# 3D FNO — Heat Treatment Digital Twin

### Master's Thesis — RISE Research Institutes of Sweden
**3D Fourier Neural Operator** for temperature prediction across all furnace regions of an industrial heat-treatment furnace.

## Why FNO?

The Fourier Neural Operator learns the solution operator of the underlying PDE
in spectral space, enabling resolution-invariant inference and constant-time
prediction regardless of trajectory length.

| Property | Value |
|---|---|
| **Approach** | 3D spectral convolution |
| **Domain** | Regular 3D grid (interpolated from OpenFOAM mesh) |
| **Prediction** | T_next directly (one-shot) |

## Software Requirements

| Component | Version |
|---|---|
| **OS** | Linux (Ubuntu 22.04 / WSL2) or HPC Linux |
| **Python** | 3.10+ |
| **PyTorch** | 2.x with CUDA 12.x |
| **NVIDIA PhysicsNeMo** | 25.06 |
| **CUDA Toolkit** | 12.1+ |
| **cuDNN** | 8.9+ |
| **OpenFOAM** | v2412 (for ground-truth data generation) |
| **Gmsh** | 4.13 (for mesh generation) |
| **HDF5** | 1.12+ (dataset I/O) |
| **NumPy / SciPy / h5py / PyYAML / matplotlib** | latest stable |

The complete software stack can be bundled into a single Apptainer
(Singularity) image (`physicsnemo_25.06.sif`) so that no Python packages
need to be installed on the host. Alternatively, install dependencies
directly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-physicsnemo
pip install numpy scipy h5py pyyaml matplotlib
```

## Hardware Requirements

### Minimum (sanity check / inference)
- **GPU**: NVIDIA GPU with >= 8 GB VRAM (e.g. RTX 3070, T4)
- **CPU**: 4 cores
- **RAM**: 16 GB
- **Disk**: 20 GB free

### Recommended (full training)
- **GPU**: NVIDIA A100 40 GB or A40 48 GB
- **CPU**: 8+ cores
- **RAM**: 64 GB
- **Disk**: 100 GB free (dataset + checkpoints + rollout outputs)

## Repository Structure

```
FNO_PhysicsNeMo_Official/
├── README.md
├── configs/         # Training / evaluation configuration
├── data/            # Data loading and preprocessing
├── dataset/         # OpenFOAM-derived training dataset (HDF5)
├── evaluation/      # Rollout evaluation scripts
├── models/          # 3D FNO architecture
├── outputs/         # Checkpoints, logs, figures
├── utils/           # Helpers (normalisation, region masks, etc.)
└── train.py         # Main training entry point
```

## How to Run

### 1. Set the dataset path

Open `configs/fno_config.py` and point `dataset_path` to your local copy of
the OpenFOAM-derived HDF5 dataset (78 cases). Output directories
(`outputs/`, `outputs/checkpoints`, `outputs/logs`) are created automatically
on the first run.

### 2. Train

The reference training command used for the thesis run:

```bash
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u train.py \
    --epochs 150 \
    --lr 1e-4 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/FNO_<run_name>/checkpoints
```

If you have the dependencies installed directly on the host (no Apptainer),
the same flags apply:

```bash
python -u train.py \
    --epochs 150 \
    --lr 1e-4 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/FNO_<run_name>/checkpoints
```

| Flag | Value | Description |
|---|---|---|
| `--epochs` | 150 | Number of training epochs |
| `--lr` | 1e-4 | Base learning rate (AdamW) |
| `--batch` | 4 | Mini-batch size |
| `--lam` | 0.003 | Physics-loss weight λ |
| `--checkpoint_dir` | `outputs/.../checkpoints` | Directory for `best_model.pt` and `latest.pt` |

Reference wall-clock on a single NVIDIA A100/A40: ≈ 24 hours.

### 3. Evaluate (autoregressive rollout)

After training, evaluate the best checkpoint with a 326-step autoregressive
rollout on the held-out test cases:

```bash
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u evaluation/evaluate.py \
    --checkpoint outputs/FNO_<run_name>/checkpoints/best_model.pt \
    --device cuda
```

This produces per-region Phase 1 / Phase 2 MAE and R², saved as JSON
under `outputs/FNO_<run_name>/evaluation/`.

### 4. Quick sanity check (optional)

To verify the data pipeline and model wiring without running a full training,
launch with a reduced epoch count:

```bash
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u train.py --epochs 1 --batch 4
```

## Key Results

### Training (one-step, validation set)
- Best steel MAE: **2.46 K**
- R²: **0.9998**

### Autoregressive Rollout

| Region | Phase 1 MAE [K] | Phase 1 R² | Phase 2 MAE [K] | Phase 2 R² |
|---|---|---|---|---|
| **Steel cylinder** | 55.86 | 0.7449 | 88.81 | < 0 |
| **Overall non-heater** | 81.25 | 0.7301 | 100.03 | 0.1598 |

- Phase 1 = in-distribution rollout (t ∈ [200, 2760] s)
- Phase 2 = temporal extrapolation (t ∈ (2760, 3460] s)