# Dataset Creation Pipeline

**Master's Thesis** — *Simulating Heat Treatment of Cast Metal Products using OpenFOAM and AI*

An automated pipeline for generating parameterised OpenFOAM simulations of a
cylindrical cast steel component inside an industrial heat-treatment furnace,
extracting the resulting temperature fields, and producing normalised HDF5
datasets for training Graph Neural Networks (GNN), Fourier Neural Operators
(FNO), and Deep Operator Networks (DeepONet).

---

## 1. Motivation

Heat treatment is an energy-intensive post-casting process in which components
are held at 900–1100 °C for a prescribed time, then cooled in a controlled
manner. Foundries currently optimise this process through experimental
trial-and-error, which is slow and expensive. A physics-based digital twin,
accelerated by neural surrogates, offers a complementary route that enables
rapid screening of heater setpoints, furnace loadings, and material parameters
at a fraction of the computational cost of direct CFD.

This package is the **data-generation stage** of that digital twin. It produces
the training corpus from which the neural surrogates learn the transient
conjugate heat-transfer response of the furnace.

---

## 2. Physical Setup

**Domain.** A three-dimensional cylindrical steel component is placed inside
the RISE heat-treatment furnace, which comprises an insulating brick lining,
an inner air cavity, and eight electrical heating elements.

**Governing physics.** OpenFOAM's `chtMultiRegionFoam` solver couples:

- **Conduction** in all solid and semi-solid regions
- **Natural convection** in the inner air cavity
- **Surface-to-surface radiation** via the view-factor model on the inner-box walls

**Regions (12 total).** `steel_cylinder`, `inner_box`, `heater_1`–`heater_8`,
`brick_heater`, `outer_box`.

**Sampled parameters.** Per-case parameters are drawn by Latin Hypercube
Sampling (LHS) from the discrete ranges defined in `configs/parameters.py`:

| Parameter            | Symbol               | Unit    | Range / Value                           |
| -------------------- | -------------------- | ------- | --------------------------------------- |
| Heater setpoint      | `T_set`              | K       | 1173.15 – 1373.15 (900 – 1100 °C)       |
| Cylinder centre (x)  | `cx`                 | m       | −0.14 … 0.14                            |
| Cylinder centre (y)  | `cy`                 | m       | 0.12, 0.15, 0.18, 0.21, 0.24            |
| Cylinder centre (z)  | `cz`                 | m       | 0.195                                   |
| Cylinder radius      | `radius`             | m       | 0.05                                    |
| Cylinder height      | `height`             | m       | 0.10                                    |
| Steel conductivity   | `kappa`              | W/m·K   | 80                                      |
| Steel heat capacity  | `Cp`                 | J/kg·K  | 450                                     |
| Steel density        | `rho`                | kg/m³   | 7800                                    |
| Brick conductivity   | `brick_heater_kappa` | W/m·K   | 8                                       |

**Target.** Temperature field `T` at every cell, at every saved timestep.

---

## 3. Pipeline Overview

```
   STEP 1            STEP 2             STEP 3                STEP 4             STEP 5
 ┌────────┐        ┌──────────┐       ┌──────────────┐      ┌──────────┐       ┌──────────┐
 │  LHS   │        │ Validate │       │ Mesh + Run   │      │  Build   │       │  Clean   │
 │ sample │  ───►  │  cases   │  ───► │   OpenFOAM   │ ───► │  HDF5    │ ───►  │ dataset  │
 │  45    │        │          │       │  (parallel)  │      │ datasets │       │          │
 └────────┘        └──────────┘       └──────────────┘      └──────────┘       └──────────┘
create_cases.py  validate_cases.py   run_all_openfoam.sh    create_dataset.py  clean_dataset.py
                                                          create_all_regions  (in OUTPUT_DIR)
                                                           _dataset.py
```

Two containers cooperate throughout the pipeline:

| Container                        | Role                                            |
| -------------------------------- | ----------------------------------------------- |
| `openfoam_2412.sif`              | Mesh generation, CHT solver, VTK export         |
| `physicsnemo_25.06.sif` (NVIDIA) | Python orchestration and HDF5 dataset assembly  |

---

## 4. Repository Structure

```
Dataset_creation/
│
├── README.md                       — this document
├── Makefile                        — local entry-point commands
├── pyproject.toml                  — packaging and dependencies
├── .env.example                    — template for local configuration
├── .env                            — local configuration (git-ignored)
├── case_manifest.json              — auto-generated: every case + status
│
├── configs/
│   ├── __init__.py
│   ├── defaults.py                 — PipelineConfig (reads .env)
│   ├── furnace.py                  — furnace bounds + heater region names
│   └── parameters.py               — sampling ranges (LHS source of truth)
│
├── scripts/
│   └── python/
│       ├── __init__.py
│       ├── create_cases.py                 — Step 1: LHS case generation
│       ├── validate_cases.py               — Step 2: pre-run sanity check
│       ├── create_dataset.py               — Step 4A: steel-cylinder HDF5
│       └── create_all_regions_dataset.py   — Step 4B: all-regions HDF5
│
├── src/
│   ├── core/
│   │   ├── case_builder.py         — assemble one OpenFOAM case
│   │   ├── dataset_builder.py      — orchestrate HDF5 dataset creation
│   │   └── manifest.py             — read / write case_manifest.json
│   │
│   ├── dataset/
│   │   ├── features.py             — build (N, 14) feature matrix
│   │   ├── normalizer.py           — z-score normalisation stats
│   │   └── writer.py               — serialise final HDF5 with metadata
│   │
│   ├── geometry/
│   │   ├── geo_patcher.py          — patch Gmsh .geo files per-case
│   │   ├── templates.py            — thermophysicalProperties template
│   │   └── validator.py            — cylinder-fits-in-furnace constraint
│   │
│   ├── openfoam/
│   │   ├── allmesh_writer.py       — regenerate Allmesh with viewFactor fix
│   │   ├── allrun_fixer.py         — clean Allrun for container execution
│   │   ├── case_cleaner.py         — remove old timesteps, VTK, logs
│   │   ├── heater_patcher.py       — set heater T = T_set in 0/heater_*/T
│   │   └── thermo_writer.py        — write steel + brick thermophysicals
│   │
│   ├── sampling/
│   │   └── lhs.py                  — Latin Hypercube Sampling + filtering
│   │
│   ├── vtk_io/
│   │   ├── reader.py               — extract steel_cylinder T(x,y,z,t)
│   │   ├── all_regions_reader.py   — extract all 12 furnace regions
│   │   └── hdf5_cache.py           — per-case HDF5 cache for re-use
│   │
│   └── utils/
│       ├── logging.py              — shared logger configuration
│       ├── naming.py               — deterministic case-name scheme
│       └── scripts.py              — generate run_all_openfoam.sh
│
└── tests/
    ├── test_feature_builder.py
    ├── test_geometry_validator.py
    └── test_lhs_sampling.py
```

---

## 5. Prerequisites

- Docker (local) or Apptainer (HPC)
- Custom image `openfoam-python` built from OpenFOAM 2412 + Python 3
- NVIDIA PhysicsNeMo 25.06 image for dataset assembly
- Gmsh ≥ 4.13 (external binary for `.geo` → `.msh` conversion)

### 5.1 Build the OpenFOAM + Python image (one-time)

The upstream OpenFOAM image does not include Python. Build a custom image once:

```bash
cat > ~/Dockerfile.openfoam << 'EOF'
FROM microfluidica/openfoam:2412
USER root
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-full python3-venv git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*
USER openfoam
WORKDIR /home/openfoam/rise_furnace
EOF

docker build -f ~/Dockerfile.openfoam -t openfoam-python .
```

### 5.2 Install the pipeline

```bash
cd Dataset_creation
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env        # edit paths afterwards
```

### 5.3 Configure (`.env`)

```bash
BASE_CASE=<path_to>/base_case_that_runs_chnage
OUTPUT_DIR=<path_to>/Dataset_k8
CONTAINER_BASE_DIR=<path_to>/Dataset_k8
N_LHS_SAMPLES=45
LHS_SEED=42
MAX_PARALLEL_JOBS=45
```

---

## 6. Usage

The pipeline is driven by Python entry-points and a `Makefile`. The five
steps below are normally executed in order.

### 6.1 Step 1 — Generate cases

```bash
make create-cases
# or:
python -m scripts.python.create_cases
```

Builds 45 OpenFOAM case directories under `$OUTPUT_DIR` and writes
`case_manifest.json`.

### 6.2 Step 2 — Validate cases

```bash
make validate
# or:
python -m scripts.python.validate_cases
```

Sanity-checks each case (geometry, mesh script, thermophysicals, parameters)
before submitting expensive simulations.

### 6.3 Step 3 — Mesh and solve (OpenFOAM)

After validation, `src/utils/scripts.py` writes a launcher
`run_all_openfoam.sh` into `$OUTPUT_DIR` that invokes meshing and
`chtMultiRegionFoam` for every case in parallel. From within the
`openfoam-python` container:

```bash
cd $OUTPUT_DIR
bash run_all_openfoam.sh
```

This is the longest step (dominated by `chtMultiRegionFoam`). On a
single 16-core HPC node, expect ≈ 10 hours wall-clock for 45 cases.

### 6.4 Step 4 — Build HDF5 datasets

```bash
make create-dataset
# or:
python -m scripts.python.create_dataset                  # 4A: steel-cylinder
python -m scripts.python.create_all_regions_dataset      # 4B: all 12 regions
```

Two HDF5 files are produced:

- `dataset_cylinder_features.h5` (steel cylinder only, 14-feature point cloud)
- `dataset_v2_all_regions.h5` (all 12 regions, hierarchical schema)

### 6.5 Step 5 — Clean dataset

`clean_dataset.py` lives in `$OUTPUT_DIR` (not in this package) and applies
filters to `dataset_v2_all_regions.h5`:

```bash
cd $OUTPUT_DIR
apptainer exec <path_to>/physicsnemo_25.06.sif python3 clean_dataset.py \
    --input  dataset_v2_all_regions.h5 \
    --output dataset_v2_all_regions_clean.h5 \
    --min-timesteps 300 \
    --min-final-time 3000 \
    --max-t-kelvin 1773
```

| Filter              | Default   | Action                                      |
| ------------------- | --------- | ------------------------------------------- |
| `--min-timesteps`   | 300       | Drop cases whose simulation crashed early   |
| `--min-final-time`  | 3000.0 s  | Drop cases that did not reach the end time  |
| `--max-t-kelvin`    | 1773.0 K  | Replace cells above this with `NaN`         |

The output `dataset_v2_all_regions_clean.h5` is the file consumed by all
three downstream surrogates (GNN, FNO, DeepONet).

### 6.6 Makefile reference

| Command              | Description                                   |
| -------------------- | --------------------------------------------- |
| `make create-cases`  | Generate OpenFOAM parameter-study cases       |
| `make validate`      | Sanity-check cases before running             |
| `make create-dataset`| Build steel-cylinder HDF5 dataset             |
| `make test`          | Run the test suite (`pytest tests/`)          |
| `make lint`          | Run `ruff` on `src/ scripts/ configs/`        |
| `make clean`         | Remove Python cache artifacts                 |

---

## 7. Execution Flow in Detail

### Step 1 — Latin Hypercube Sampling

`scripts/python/create_cases.py` enumerates all geometrically valid parameter
combinations (filtered by `geometry/validator.py` to ensure the cylinder fits
inside the furnace cavity), samples 45 unique cases using a fixed seed for
reproducibility, and builds one OpenFOAM case directory per sample. For each
case, `core/case_builder.py` performs eight actions: copy the base case, clean
stale artefacts, patch heater temperatures, write the steel and brick
thermophysical properties, patch the Gmsh `.geo` file, rewrite `Allmesh`, fix
`Allrun`, and persist `cylinder_params.json`. The manifest
(`case_manifest.json`) becomes the single source of truth for the run.

### Step 2 — Validation

`scripts/python/validate_cases.py` walks the manifest and verifies that every
case has the expected directory structure, a patched `.geo` file, an executable
`Allmesh`, both `thermophysicalProperties` files (steel and brick), and a
valid `cylinder_params.json`. Failures are logged but do not abort the pipeline.

### Step 3 — Meshing and Simulation

For each case, in parallel (one per CPU), the launcher performs:

1. **Gmsh meshing** — native binary, `.geo` → `.msh`
2. **OpenFOAM meshing** — `gmshToFoam`, `topoSet`, `splitMeshRegions`
3. **View-factor wall fix** — `sed` on `constant/inner_box/polyMesh/boundary`
4. **Baffle creation** — `createBaffles -region rightFluid`
5. **View-factor generation** — `viewFactorsGen -region inner_box`
6. **Transient CHT solve** — `chtMultiRegionFoam` (dominant runtime)
7. **VTK export** — `foamToVTK -allRegions`

A case is considered successful if the `VTK/` directory exists after this loop.

### Step 4 — Dataset Construction

Two HDF5 datasets are produced from the VTK output:

**4A — `dataset_cylinder_features.h5`** (steel cylinder only). Built by
`create_dataset.py` via `core/dataset_builder.py`. For each case, the VTK
time-series is read once (cached per-case in `steel_cylinder_T_timeseries.h5`)
and converted into a 14-column feature matrix:

```
[ x, y, z, t, T_set, cx, cy, cz, radius, height,
  kappa, Cp, rho, brick_heater_kappa ]  →  T
```

All cases are concatenated; z-score normalisation statistics are computed on
the combined matrix and saved alongside the raw and normalised arrays. An
inline outlier filter drops rows where `T ≥ 1773 K`.

**4B — `dataset_v2_all_regions.h5`** (consumed by GNN, FNO, and DeepONet).
Built by `create_all_regions_dataset.py` via `vtk_io/all_regions_reader.py`.
All 12 regions are extracted and stored in a hierarchical HDF5 layout:

```
case_XXX/
    attrs: name, T_set
    times
    steel_cylinder/{coords, T}
    inner_box/{coords, T}
    heater_1 … heater_8/{coords, T}
    brick_heater/{coords, T}
    outer_box/{coords, T}
```

### Step 5 — Dataset Cleaning

`clean_dataset.py` (located in `$OUTPUT_DIR`, not in this package) writes
`dataset_v2_all_regions_clean.h5` after applying the three filters listed
in §6.5.

Note that Steps 4 and 5 apply different outlier strategies by design:

| Strategy         | Script                  | Rationale                                           |
| ---------------- | ----------------------- | --------------------------------------------------- |
| Drop row         | `dataset_builder.py`    | Acceptable for point-cloud regression               |
| Replace with NaN | `clean_dataset.py`      | Preserves mesh topology required by GNN, FNO, DeepONet |

---

## 8. Dataset Specification

The v2 filename convention denotes the second revision of the all-regions
schema (steel κ fixed at 80, brick κ fixed at 8, expanded `cx` sampling).

### 8.1 `dataset_cylinder_features.h5`

```
/X_raw              (N, 14)   float32   raw feature matrix
/Y_raw              (N, 1)    float32   raw temperatures [K]
/X_norm, /Y_norm    z-score normalised copies
/X_mean, /X_std     (14,)     normalisation statistics
/Y_mean, /Y_std     scalars
/sim_start_indices  (n_sims,) int64     per-simulation row boundaries
attrs:
  feature_cols      JSON list of 14 column names
  target_col        "T"
  n_simulations, total_points, n_features
  case_summary      JSON per-case parameter record
```

Loading example:

```python
import h5py, json, numpy as np

with h5py.File("dataset_cylinder_features.h5", "r") as f:
    X_norm       = f["X_norm"][:]
    Y_norm       = f["Y_norm"][:]
    feature_cols = json.loads(f.attrs["feature_cols"])
    sim_starts   = f["sim_start_indices"][:]

# Leak-free train/val split by simulation
n_sims = len(sim_starts)
val_sims = [n_sims - 1]
val_mask = np.zeros(len(X_norm), dtype=bool)
for s in val_sims:
    start = sim_starts[s]
    end   = sim_starts[s + 1] if s + 1 < n_sims else len(X_norm)
    val_mask[start:end] = True

X_train, Y_train = X_norm[~val_mask], Y_norm[~val_mask]
X_val,   Y_val   = X_norm[val_mask],  Y_norm[val_mask]
```

### 8.2 `dataset_v2_all_regions.h5` and `dataset_v2_all_regions_clean.h5`

```
attrs:
  n_cases, regions (JSON list)
  filter_min_timesteps        ← only in *_clean.h5
  filter_min_final_time       ← only in *_clean.h5
  filter_max_t_kelvin         ← only in *_clean.h5

case_XXX/
  attrs: name, T_set
         original_index       ← only in *_clean.h5
  times                                     (n_times,)
  <region>/
    coords                                  (n_cells, 3)
    T                                       (n_times, n_cells)
    attrs: outliers_replaced_with_nan       ← only in *_clean.h5
```

Loading example:

```python
import h5py, json

with h5py.File("dataset_v2_all_regions_clean.h5", "r") as f:
    n_cases = int(f.attrs["n_cases"])
    regions = json.loads(f.attrs["regions"])
    for ci in range(n_cases):
        grp   = f[f"case_{ci:03d}"]
        T_set = float(grp.attrs["T_set"])
        times = grp["times"][:]
        for region in regions:
            if region in grp:
                coords = grp[region]["coords"][:]
                T      = grp[region]["T"][:]       # NaN where T > 1773 K
```

---

## 9. Output Tree

After a successful run, `$OUTPUT_DIR` contains:

```
Dataset_k8/
├── case001_Tset1173.15K_cy150mm_cz195mm_r50mm_h100mm_k80.../
│   ├── 0/  constant/  system/
│   ├── VTK/                                       ← time-series output
│   ├── steel_cylinder_T_timeseries.h5             ← per-case cache
│   ├── cylinder_params.json                       ← feature values
│   └── log.chtMultiRegionFoam
├── case002_.../
├── ... 
├── case_manifest.json
├── dataset_cylinder_features.h5                   ← Step 4A output
├── dataset_v2_all_regions.h5                      ← Step 4B output (raw)
├── dataset_v2_all_regions_clean.h5                ← Step 5 output (GNN / FNO / DeepONet)
├── clean_dataset.py
├── run_all_openfoam.sh
└── logs/
    ├── pipeline_<jobid>.log
    └── case001_....log
```

---

## 10. Integration with Downstream Models

The datasets produced here are consumed by three neural surrogate packages in
the thesis repository, all built on the NVIDIA PhysicsNeMo framework:

| Package                  | Dataset consumed                        | Model family                                           |
| ------------------------ | --------------------------------------- | ------------------------------------------------------ |
| `GNN_PhysicsNeMo_*/`     | `dataset_v2_all_regions_clean.h5`       | Graph Neural Networks — NVIDIA PhysicsNeMo             |
| `FNO_*/`                 | `dataset_v2_all_regions_clean.h5`       | Fourier Neural Operators — NVIDIA PhysicsNeMo          |
| `DeepONet_*/`            | `dataset_v2_all_regions_clean.h5`       | Deep Operator Networks — NVIDIA PhysicsNeMo            |

---

## 11. Reproducibility

The pipeline is fully deterministic when `.env` and `parameters.py` are
unchanged:

- The LHS seed is pinned at 42
- `parameters.py` is the single source of truth for all sampling ranges
- `case_manifest.json` records every parameter value per case
- `cylinder_params.json` (per case) allows feature-matrix reconstruction
  independently of future code changes

Re-running the pipeline produces bit-identical case parameters in the
same order.

---

## 12. Troubleshooting

| Symptom                                                       | Cause                                        | Resolution                                                   |
| ------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| `python: command not found`                                   | Makefile assumes `python` alias              | `sed -i 's/python -m/python3 -m/g' Makefile`                 |
| `ModuleNotFoundError: dotenv`                                 | `python-dotenv` missing in container         | `pip install python-dotenv` inside the running container     |
| `externally-managed-environment` on `pip install`             | System Python protected                      | Activate `.venv` first: `source .venv/bin/activate`          |
| Cases saved to wrong folder                                   | `.env` not loaded                            | `python3 -c "from configs.defaults import PipelineConfig; print(PipelineConfig().output_dir)"` |
| `Permission denied` during `apt-get`                          | Not running as root                          | `docker run -it --user root -v ... openfoam-python bash`     |
| `.venv` missing after container restart                       | `.venv` outside the mounted volume           | Place `.venv` inside the mounted project directory           |
| `viewFactorsGen` fails in `inner_box`                         | Boundary not marked as `viewFactorWall`      | Check `constant/inner_box/polyMesh/boundary` → `inGroups 2(wall viewFactorWall)` |
| `chtMultiRegionFoam` diverges                                 | Bad initial `viewFactorField` at high `T_set`| Lower the upper bound of `T_set` in `parameters.py`          |
| VTK reader reports no `steel_cylinder`                        | Simulation crashed before writing output     | Inspect `log.chtMultiRegionFoam` for the failing case        |
| Step 5 drops all cases                                        | Write interval too coarse                    | Reduce `writeInterval` in base case `controlDict`            |
