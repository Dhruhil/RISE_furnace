# OpenFOAM Dataset Pipeline

Automated OpenFOAM case generation and ML dataset creation for
cylinder-in-furnace heat transfer simulations.

---

## Overview

This pipeline uses **two Docker containers**:

| Container | Purpose |
|---|---|
| `openfoam-python` (custom build) | Generate cases, run simulations |
| `physicsnemo:25.06` (NVIDIA) | Build ML dataset from results |

Flow:
1. **Generate** parameterised OpenFOAM cases using Latin Hypercube Sampling
2. **Run** simulations inside the OpenFOAM container
3. **Extract** temperature fields and build a normalised HDF5 dataset inside the PhysicsNeMo container

---

## Prerequisites

- Docker installed on your host machine
- NVIDIA GPU + drivers (for PhysicsNeMo container)
- This repo cloned somewhere on your host:

```bash
git clone https://github.com/Dhruhil/RISE_furnace.git
cd RISE_furnace/Dataset_creation
```

---

## Step 0 — Build the OpenFOAM + Python image (once only)

The base OpenFOAM image has no Python. Build a custom image that includes it:

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

> You only do this **once**. The image is saved locally as `openfoam-python`.

---

## Container 1 — OpenFOAM (Case Generation + Simulation)

### Start the container

Replace `<your_host_path>` with the folder where your `rise_furnace` directory lives:

```bash
docker run -it \
  --user root \
  -v <your_host_path>/rise_furnace:/home/openfoam/rise_furnace \
  openfoam-python bash
```

Example:
```bash
docker run -it --user root \
  -v ~/OpenFOAM/rise_furnace:/home/openfoam/rise_furnace \
  openfoam-python bash
```

### First time setup (once only)

The `.venv` lives inside your mounted volume so it **persists** across container restarts:

```bash
cd /home/openfoam/rise_furnace/Dataset_creation

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
sed -i 's/python -m/python3 -m/g' Makefile
```

### Every time you restart the container

```bash
cd /home/openfoam/rise_furnace/Dataset_creation
source .venv/bin/activate
```

---

## Configure Paths

```bash
cp .env.example .env
nano .env
```

Edit `.env` with your paths (all paths are **inside the container**):

```properties
# Path to your base OpenFOAM case (inside container)
BASE_CASE=/home/openfoam/rise_furnace/base_case_that_runs_chnage

# Where generated cases will be saved (inside container)
OUTPUT_DIR=/home/openfoam/rise_furnace/Testing_Create_Dataset

# Same as OUTPUT_DIR — written into the generated run script
CONTAINER_BASE_DIR=/home/openfoam/rise_furnace/Testing_Create_Dataset

# Number of LHS samples to generate
N_LHS_SAMPLES=22

# Random seed for reproducibility
LHS_SEED=42

# Max parallel simulation jobs
MAX_PARALLEL_JOBS=3
```

---

## Step-by-Step Workflow

### Step 1 — Generate cases

```bash
make create-cases
```

Creates case folders inside `OUTPUT_DIR`:
```
Testing_Create_Dataset/
├── case001_Tset1000_.../
├── case002_Tset950_.../
├── ...
├── case_manifest.json
└── run_all_openfoam.sh
```

### Step 2 — Validate cases (optional but recommended)

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
tail -f /home/openfoam/rise_furnace/Testing_Create_Dataset/parallel_logs/*.log
```

---

## Container 2 — PhysicsNeMo (Dataset Creation)

### Start the container

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -u 0:0 \
  -v <your_host_path>/rise_furnace:/workspace/rise_furnace \
  nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
```

Example:
```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -u 0:0 \
  -v ~/OpenFOAM/rise_furnace:/workspace/rise_furnace \
  nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
```

### First time setup (once only)

```bash
cd /workspace/rise_furnace/Dataset_creation
pip install python-dotenv   # everything else is pre-installed in PhysicsNeMo
```

### Step 4 — Build ML dataset

```bash
cd /workspace/rise_furnace/Dataset_creation
make create-dataset
```

Output:
```
Testing_Create_Dataset/
└── dataset_cylinder_features.h5    ← normalised HDF5 dataset for ML training
```

---

## Clean and Re-run

```bash
rm -rf /home/openfoam/rise_furnace/Testing_Create_Dataset/*
make create-cases
```

---

## Project Structure

```
Dataset_creation/
├── .env                    # Your local config (git ignored)
├── .env.example            # Template — copy to .env and edit
├── Makefile
├── pyproject.toml
├── configs/
│   ├── defaults.py         # PipelineConfig — reads from .env
│   └── parameters.py       # LHS parameter ranges
├── scripts/
│   ├── create_cases.py     # Step 1: generate cases
│   ├── validate_cases.py   # Step 2: validate
│   └── create_dataset.py   # Step 4: build dataset
├── src/
│   ├── core/               # Case builder, manifest
│   ├── geometry/           # Geometry patcher
│   ├── openfoam/           # OpenFOAM file writers
│   ├── sampling/           # Latin Hypercube Sampling
│   └── utils/              # Logging, naming, scripts
└── tests/
```

---

## Makefile Commands

| Command | Description |
|---|---|
| `make create-cases` | Generate OpenFOAM parameter study cases |
| `make validate` | Validate generated cases before running |
| `make create-dataset` | Build ML training HDF5 dataset |
| `make clean` | Remove Python cache files |
| `make test` | Run test suite |
| `make lint` | Run ruff linter |

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

**`Permission denied` on apt-get**
```bash
docker run -it --user root -v ... openfoam-python bash
```

**`.venv` missing after container restart**
```bash
# Confirm .venv is inside the mounted volume, not container-only storage
ls /home/openfoam/rise_furnace/Dataset_creation/.venv
```