# PINN PhysicsNeMo — Heat Treatment (All Regions)

### Master's Thesis — RISE Research Institutes of Sweden
**Physics-Informed Neural Network** for temperature prediction via exact PDE solving.

## Overview

This is the third AI model for the thesis, complementing GNN (MeshGraphNet) and FNO.
Uses a deep MLP with SIREN activation that directly solves the heat equation PDE
via automatic differentiation — the defining characteristic of PINNs.

### Architecture

```
Input:  (batch, 6) = [x, y, z, t, T_set, region_id]  (normalised)
                ↓
        Deep MLP with SIREN activation
        - 6 hidden layers, 256 neurons each
        - Sinusoidal activation (omega_0 = 30)
                ↓
Output: (batch, 1) = [T]  (normalised temperature)
```

### Key Difference from GNN/FNO

| | GNN (MeshGraphNet) | FNO | PINN |
|---|---|---|---|
| **Category** | Neural operator | Neural operator | Physics-informed NN |
| **Approach** | Graph message passing | Spectral convolution | MLP + PDE residual |
| **Physics** | Soft constraints in loss | Soft constraints in loss | Exact PDE via autograd |
| **Prediction** | δT per step | T_next directly | T(x,y,z,t) directly |
| **Data needed** | Yes (OpenFOAM) | Yes (OpenFOAM) | Optional (PDE is the loss) |
| **Speed** | ~100x vs OpenFOAM | ~1000x vs OpenFOAM | ~10x vs OpenFOAM |

### Heat Equation PDE

The PINN enforces the exact heat equation via automatic differentiation:

```
ρ·Cp·∂T/∂t = κ·∇²T = κ·(∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
```

Where:
- κ = 25 W/(m·K) — thermal conductivity of steel
- ρ = 7800 kg/m³ — density
- Cp = 450 J/(kg·K) — specific heat capacity
- α = κ/(ρ·Cp) = 7.12e-6 m²/s — thermal diffusivity

### Dataset

Uses the same `dataset_all_regions.h5` (911 MB) as GNN and FNO.

## Project Structure

```
PINN_PhysicsNeMo_Official/
├── configs/
│   └── pinn_config.py
├── data/
│   └── dataset.py               # Reads dataset_all_regions.h5
├── models/
│   ├── pinn_model.py             # PhysicsNeMo FullyConnected + fallback
│   └── physics.py                # PDE residual via autograd
├── utils/
│   ├── metrics.py
│   └── checkpoint.py
├── train.py                      # 2-phase: pretrain + physics
├── run_alvis_pinn.sh
└── README.md
```

## How to Run

```bash
cd /mimer/NOBACKUP/groups/revar/PINN_PhysicsNeMo_Official
sbatch run_alvis_pinn.sh
```

## Training Phases

**Phase A (3000 epochs)**: Pure data fitting — learn approximate T field.
**Phase B (5000 epochs)**: Physics-informed — enforce PDE with λ curriculum.

Lambda curriculum (same as GNN/FNO):
```
λ = 0.001 * exp(4.6 * epoch/n_epochs), cap at 0.10
```
