# DeepONet — Heat Treatment Digital Twin

### Master's Thesis — RISE Research Institutes of Sweden
**Deep Operator Network** built on the **official NVIDIA PhysicsNeMo Sym
`DeepONetArch`** for temperature prediction across all furnace regions of
an industrial heat-treatment furnace.

## Why DeepONet?

The Deep Operator Network learns an operator between infinite-dimensional
function spaces using a branch–trunk decomposition. The branch network
encodes the input field at fixed sensor locations, while the trunk network
encodes arbitrary query coordinates. The PhysicsNeMo implementation
combines the two encodings via element-wise product followed by a learned
linear output layer, allowing the model to evaluate the solution at any
point in the domain without grid interpolation.

| Property | Value |
|---|---|
| **Approach** | NVIDIA PhysicsNeMo Sym `DeepONetArch` (branch · trunk + output linear) |
| **Branch net** | `FullyConnectedArch`, 3×256, GELU |
| **Trunk net** | `FullyConnectedArch`, 4×256, GELU |
| **Latent dim** | 128 |
| **Combination** | element-wise product → `Linear(128 → 1)` |
| **Domain** | Continuous (queries at arbitrary (x, y, z)) |
| **Prediction** | T_next directly at query points |
| **Inter-region coupling** | Implicit (shared branch encoding) |
| **Discretisation** | Resolution-flexible at inference |

## Training

| Hyperparameter | Value |
|---|---|
| **Epochs** | 150 |
| **Batch size** | 4 |
| **Optimizer** | AdamW (weight decay 1e-4) |
| **Base learning rate** | 1e-4 |
| **LR scheduler** | ReduceLROnPlateau (factor 0.5, patience 15) |
| **Gradient clipping** | ‖g‖₂ ≤ 1.0 |
| **Physics λ** | 0.003 (gentle regulariser) |
| **Time step Δt** | 10 s |
| **Total simulation time** | 3,460 s |
| **Training horizon** | 0 → 2,760 s (in-distribution) |
| **Extrapolation horizon** | 2,760 → 3,460 s |
| **Validation fraction** | 0.14 |
| **Test fraction** | 0.10 |

## Software Requirements

| Component | Version |
|---|---|
| **OS** | Linux (Ubuntu 22.04 / WSL2) or HPC Linux |
| **Python** | 3.10+ |
| **PyTorch** | 2.x with CUDA 12.x |
| **NVIDIA PhysicsNeMo (Sym)** | 25.06 |
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
pip install nvidia-physicsnemo nvidia-physicsnemo-sym
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
DeepONet_PhysicsNeMo_Official/
├── README.md
├── configs/         # Python training/eval configuration (deeponet_config.py)
├── data/            # Data loading and preprocessing
├── evaluation/      # Rollout evaluation scripts
├── models/          # HeatTreatmentDeepONet (PhysicsNeMo Sym)
├── outputs/         # Checkpoints, logs
├── training/        # Training entry points
└── utils/           # Helpers (normalisation, region masks, etc.)
```

## How to Run

### 1. Set the dataset path

Open `configs/deeponet_config.py` and point `dataset_path` to your local
copy of the OpenFOAM-derived HDF5 dataset (78 cases). Output directories
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
    --checkpoint_dir outputs/DeepONet_<run_name>/checkpoints
```

If you have the dependencies installed directly on the host (no Apptainer),
the same flags apply:

```bash
python -u train.py \
    --epochs 150 \
    --lr 1e-4 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/DeepONet_<run_name>/checkpoints
```

| Flag | Value | Description |
|---|---|---|
| `--epochs` | 150 | Number of training epochs |
| `--lr` | 1e-4 | Base learning rate (AdamW) |
| `--batch` | 4 | Mini-batch size |
| `--lam` | 0.003 | Physics-loss weight λ |
| `--checkpoint_dir` | `outputs/.../checkpoints` | Directory for `best_model.pt` and `latest.pt` |

Reference wall-clock on a single NVIDIA A100: ≈ 60 hours for 150 epochs.

### 3. Evaluate (autoregressive rollout)

After training, evaluate the best checkpoint on the held-out test cases:

```bash
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u evaluation/evaluate.py \
    --checkpoint outputs/DeepONet_<run_name>/checkpoints/best_model.pt \
    --device cuda
```

This produces per-region Phase 1 / Phase 2 MAE and R², saved as JSON
under `outputs/DeepONet_<run_name>/evaluation/`.

### 4. Quick sanity check (optional)

To verify the data pipeline and model wiring without running a full training,
launch with a reduced epoch count:

```bash
apptainer exec --nv --cleanenv \
  <path_to>/physicsnemo_25.06.sif \
  python -u train.py --epochs 1 --batch 4
```

## Key Results

### Autoregressive Rollout (7 held-out test cases)

| Region | Phase 1 MAE [K] | Phase 1 R² | Phase 2 MAE [K] | Phase 2 R² |
|---|---|---|---|---|
| **Steel cylinder** | 381.83 | < 0 | 754.34 | < 0 |
| **Overall non-heater** | 255.06 | < 0 | 538.64 | < 0 |

- **Parameters**: 3.72 M
- Phase 1 = in-distribution rollout (t ∈ [200, 2760] s)
- Phase 2 = temporal extrapolation (t ∈ (2760, 3460] s)