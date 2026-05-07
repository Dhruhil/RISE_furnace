# Simulating Heat Treatment of Cast Metal Products using OpenFOAM and AI

**Master's Thesis** — *RISE Research Institutes of Sweden / Jönköping University*

This repository contains the complete digital-twin pipeline for the heat
treatment of cast metal components: a validated OpenFOAM ground-truth
solver, a Latin-Hypercube simulation campaign producing **78 converged
cases**, and three physics-informed AI surrogate architectures
(**GNN**, **FNO**, **DeepONet**) implemented on top of NVIDIA PhysicsNeMo.

---

## Repository Structure

```
.
├── Dataset_creation/                  # OpenFOAM data-generation pipeline
│   └── ...                            # Geometry, meshing, CHT solve, HDF5 export
│
├── GNN_PhysicsNeMo_Official/          # MeshGraphNet surrogate (PhysicsNeMo)
│   ├── README.md
│   └── ...
│
├── FNO_PhysicsNeMo_Official/          # 3D Fourier Neural Operator surrogate
│   ├── README.md
│   └── ...
│
└── DeepONet_PhysicsNeMo_Official/     # Deep Operator Network surrogate
    ├── README.md
    └── ...
```

Each surrogate sub-folder contains its own `README.md` with full details
on the architecture, training command, and reproduction steps.

---

## Pipeline Overview

```
        ┌──────────────────────────────┐
        │   1. Dataset_creation/       │
        │   OpenFOAM CHT solver        │
        │   chtMultiRegionFoam         │
        │   89 LHS samples → 78 cases  │
        └──────────────┬───────────────┘
                       │  HDF5 dataset
                       ▼
   ┌────────────────────────────────────────────┐
   │  2. AI surrogate training (PhysicsNeMo)    │
   ├────────────────────────────────────────────┤
   │   GNN_PhysicsNeMo_Official  (MeshGraphNet) │
   │   FNO_PhysicsNeMo_Official  (3D FNO)       │
   │   DeepONet_PhysicsNeMo_Official            │
   └────────────────┬───────────────────────────┘
                    │  trained checkpoints
                    ▼
        ┌──────────────────────────────┐
        │   3. Autoregressive rollout  │
        │   evaluation/evaluate.py     │
        │   per-region MAE / R²        │
        └──────────────────────────────┘
```

---

## 1. Dataset_creation/

Generates the ground-truth dataset by running an automated OpenFOAM
pipeline over a Latin-Hypercube sample of the design space
($T_{\text{set}}, c_x, c_y$).

**Per-case stages:**

1. Draw $(T_{\text{set}}, c_x, c_y)$ from the LHS plan
2. Geometric feasibility check (cylinder fits inside furnace with clearance)
3. Build a self-contained OpenFOAM case directory (parameterized Gmsh script + boundary conditions)
4. Mesh, split into 12 regions, compute view-factor matrix, run transient CHT solve
5. Convergence filter (`N_t ≥ 300`, `t_last ≥ 3000` s) and HDF5 export

**Outputs:**
- `dataset_v2_all_regions_clean.h5` — 78 converged cases at Δt = 10 s
- One HDF5 group per simulation, with per-region `coords`, `T(t, n_cells)`,
  and parameter attributes (`T_set`, `cx`, `cy`, `cz`, `radius`, `height`)

This dataset is the shared input to all three surrogates.

---

## 2. AI Surrogate Models

All three surrogates target the same task — autoregressive next-step
temperature prediction across all 12 furnace regions — but differ
fundamentally in how they represent space.

| Surrogate | Spatial primitive | Backbone | Parameters | Steel MAE (Phase 1) | R² (Phase 1) |
|---|---|---|---|---|---|
| **GNN**       | Mesh node     | MeshGraphNet (PhysicsNeMo)   | 0.70 M   | **2.06 K**   | **0.9996** |
| **FNO**       | Voxel         | 3D Fourier Neural Operator   | 22.4 M   | 55.86 K      | 0.7449     |
| **DeepONet**  | Sensor + query| `DeepONetArch` (PhysicsNeMo) | 3.72 M   | 381.83 K     | < 0        |

Phase 1 = in-distribution rollout (t ∈ [200, 2760] s) on 7 held-out test
cases. Heater regions are excluded (clamped to `T_set`, never predicted).

The graph-based surrogate, despite having the smallest parameter count,
outperforms the FNO and DeepONet by one and two orders of magnitude
respectively — because its message-passing primitive directly mirrors
the local stencil structure of the finite-volume CHT solver.

### 2.1 GNN_PhysicsNeMo_Official/

Physics-informed MeshGraphNet that treats all 12 furnace regions as one
**unified graph** (~13.6 k nodes, ~150 k edges), so the network learns
cross-region heat transfer (conduction, convection, radiation) end-to-end.

- 4 message-passing layers, 128 hidden units
- 16 node features, 5 edge features
- Region-weighted data loss + 4-term physics loss (Fourier / Newton / Stefan-Boltzmann / energy balance)
- Pushforward training for autoregressive stability
- See [`GNN_PhysicsNeMo_Official/README.md`](GNN_PhysicsNeMo_Official/README.md)

### 2.2 FNO_PhysicsNeMo_Official/

3D Fourier Neural Operator that interpolates each per-cell field onto a
regular `30 × 36 × 54` Cartesian voxel grid aligned with the furnace
bounding box, then operates on a truncated frequency spectrum.

- 3 Fourier layers, modes `[15, 18, 27]`, 32 latent channels
- InstanceNorm3d + GELU activations
- One-shot prediction of $T_{\text{next}}$
- See [`FNO_PhysicsNeMo_Official/README.md`](FNO_PhysicsNeMo_Official/README.md)

### 2.3 DeepONet_PhysicsNeMo_Official/

Deep Operator Network built on the official NVIDIA PhysicsNeMo Sym
`DeepONetArch`, with a `FullyConnectedArch` branch and trunk.

- Branch: 3 × 256 MLP, 2,160 sensors (10 × 12 × 18 lattice)
- Trunk: 4 × 256 MLP, 1,024 query points per sample
- Latent dim 128, element-wise product + learned linear output
- See [`DeepONet_PhysicsNeMo_Official/README.md`](DeepONet_PhysicsNeMo_Official/README.md)

---

## 3. Software Requirements (shared)

| Component | Version |
|---|---|
| OS | Linux (Ubuntu 22.04 / WSL2) or HPC Linux |
| Python | 3.10+ |
| PyTorch | 2.x with CUDA 12.x |
| NVIDIA PhysicsNeMo (+ Sym for DeepONet) | 25.06 |
| CUDA Toolkit | 12.1+ |
| OpenFOAM | v2412 (Dataset_creation only) |
| Gmsh | 4.13 (Dataset_creation only) |
| HDF5 | 1.12+ |
| NumPy / SciPy / h5py / PyYAML / matplotlib | latest stable |

The complete software stack can be bundled into a single Apptainer
(Singularity) image (`physicsnemo_25.06.sif`). Each sub-project README
shows both the Apptainer-based and direct-host install paths.

---

## 4. Hardware Requirements (shared)

| Stage | Recommended | Minimum |
|---|---|---|
| OpenFOAM CHT solve (per case) | 16-core CPU | 8-core CPU |
| Surrogate training | 1× NVIDIA A100 / A40 | 1× GPU ≥ 24 GB VRAM |
| Surrogate inference | any CUDA-capable GPU | CPU also works |

A full OpenFOAM CHT solve takes ~10 hours per case on CPU; a trained
surrogate completes the same 326-step rollout in seconds.

---

## 6. Headline Results

| Aspect | OpenFOAM | Best surrogate (GNN) |
|---|---|---|
| Per-case wall-clock | ~10 hours | a few seconds |
| Steel cylinder MAE (Phase 1) | reference | **2.06 K** |
| Steel cylinder R² (Phase 1) | reference | **0.9996** |
| Industrial control resolution | n/a | within ±10 K (in-distribution) |

The graph-based surrogate is the only one of the three to meet the ±10 K
control resolution typical of industrial heat-treatment furnaces under
in-distribution rollout, while reducing per-case turnaround from
HPC-allocation scale to engineering-workstation scale.

---

