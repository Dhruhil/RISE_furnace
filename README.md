# Simulating Heat Treatment of Cast Metal Products using OpenFOAM
### Master Thesis — RISE Research Institutes of Sweden

Automated pipeline for generating parametric OpenFOAM simulation cases and
building a normalised HDF5 dataset for machine learning.

---

## Project Structure

```
Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/
└── Dataset_creation/
    ├── configs/
    │   ├── defaults.py        # PipelineConfig — reads from .env
    │   ├── furnace.py         # Furnace geometry constants
    │   └── parameters.py      # LHS parameter ranges + feature columns
    ├── scripts/
    │   ├── create_cases.py    # Step 1: generate OpenFOAM cases
    │   ├── validate_cases.py  # Step 2: validate before running
    │   └── create_dataset.py  # Step 3: build ML dataset from results
    ├── src/
    │   ├── core/              # Case builder, dataset builder, manifest
    │   ├── geometry/          # .geo patcher, geometry validator
    │   ├── openfoam/          # Allmesh writer, heater patcher, thermo writer
    │   ├── sampling/          # Latin Hypercube Sampling
    │   ├── dataset/           # Feature matrix, normaliser, HDF5 writer
    │   └── vtk_io/            # VTK reader, HDF5 cache
    ├── tests/                 # Unit tests
    ├── .env.example           # Template — copy to .env and edit
    ├── Makefile
    └── pyproject.toml
```

> **Note:** The OpenFOAM base case and simulation output folders are stored
> on the server only — too large for GitHub.
> Server path: `/home/openfoam/rise_furnace/base_case_that_runs_chnage`

---

## Pipeline Overview

```
1. Generate cases   →   make create-cases
                        Creates parametric OpenFOAM case folders using
                        Latin Hypercube Sampling over cylinder geometry
                        and material properties.

2. Run simulations  →   bash run_all_openfoam.sh
                        Runs all cases in parallel inside the OpenFOAM
                        Docker container.

3. Build dataset    →   make create-dataset
                        Reads VTK output, extracts temperature fields,
                        builds a normalised HDF5 dataset for ML training.
```

Output: `dataset_cylinder_features.h5` — normalised feature matrix ready for ML.

---

## Quick Start

### Requirements

- Docker
- OpenFOAM container (`openfoam-python` — build instructions below)
- NVIDIA PhysicsNeMo container for dataset step (`physicsnemo:25.06`)

### Build the OpenFOAM + Python image (once only)

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

### Start the OpenFOAM container

```bash
docker run -it --user root \
  -v /home/openfoam/rise_furnace:/home/openfoam/rise_furnace \
  openfoam-python bash
```

### Setup inside container (once only)

```bash
cd /home/openfoam/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/Dataset_creation

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Every time you restart the container

```bash
cd /home/openfoam/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/Dataset_creation
source .venv/bin/activate
```

---

## Configuration

```bash
cp .env.example .env
nano .env
```

| Variable | Description | Example |
|---|---|---|
| `BASE_CASE` | Path to OpenFOAM base case | `/home/openfoam/rise_furnace/base_case_that_runs_chnage` |
| `OUTPUT_DIR` | Where generated cases are saved | `/home/openfoam/rise_furnace/Testing_Create_Dataset` |
| `CONTAINER_BASE_DIR` | Same as OUTPUT_DIR (used in run script) | `/home/openfoam/rise_furnace/Testing_Create_Dataset` |
| `N_LHS_SAMPLES` | Number of parameter study cases | `22` |
| `LHS_SEED` | Random seed for reproducibility | `42` |
| `MAX_PARALLEL_JOBS` | Parallel simulation jobs | `3` |

---

## Step-by-Step

### Step 1 — Generate cases

```bash
make create-cases
```

Creates case folders inside `OUTPUT_DIR`:

```
Testing_Create_Dataset/
├── case001_Tset1000_cy180mm_.../
├── case002_Tset950_cy150mm_.../
├── ...
├── case_manifest.json
└── run_all_openfoam.sh
```

### Step 2 — Validate (optional but recommended)

```bash
make validate
```

### Step 3 — Run simulations

```bash
cd /home/openfoam/rise_furnace/Testing_Create_Dataset
bash run_all_openfoam.sh
```

Monitor progress:

```bash
tail -f parallel_logs/*.log
```

### Step 4 — Build ML dataset (PhysicsNeMo container)

```bash
docker run --rm -it --gpus all -u 0:0 \
  -v /home/openfoam/rise_furnace:/workspace/rise_furnace \
  nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash

cd /workspace/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/Dataset_creation
pip install python-dotenv
make create-dataset
```

Output: `Testing_Create_Dataset/dataset_cylinder_features.h5`

---

## Makefile Commands

| Command | Description |
|---|---|
| `make create-cases` | Generate OpenFOAM parameter study cases |
| `make validate` | Validate generated cases |
| `make create-dataset` | Build normalised HDF5 dataset |
| `make test` | Run unit tests |
| `make lint` | Run ruff linter |
| `make clean` | Remove Python cache files |

---

## Dataset Format

The output HDF5 file contains:

| Key | Shape | Description |
|---|---|---|
| `X_norm` | `(N, 15)` | Normalised input features |
| `Y_norm` | `(N, 1)` | Normalised temperature |
| `X_mean`, `X_std` | `(15,)` | Normalisation parameters |
| `Y_mean`, `Y_std` | scalar | Normalisation parameters |
| `sim_start_indices` | `(n_sims,)` | Row boundaries per simulation |

Feature columns (order is fixed — do not change after first training run):

```
x, y, z, t, T_set, cx, cy, cz, radius, height, volume, mass, kappa, Cp, rho
```

---

## Troubleshooting

**`python: command not found`**
```bash
sed -i 's/python -m/python3 -m/g' Makefile
```

**`externally-managed-environment` pip error**
```bash
source .venv/bin/activate
pip install -e .
```

**Cases saved to wrong folder**
```bash
python3 -c "from configs.defaults import PipelineConfig; print(PipelineConfig().output_dir)"
```

**`.venv` missing after container restart**
```bash
ls /home/openfoam/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/Dataset_creation/.venv
# if missing, re-run the setup steps above
```