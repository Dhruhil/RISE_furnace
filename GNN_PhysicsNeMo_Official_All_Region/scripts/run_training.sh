#!/bin/bash
# ============================================================
# Training launcher
# Run this INSIDE the PhysicsNeMo container from your folder:
#
#   root@c1c025623cd5:~# cd /workspace/rise_furnace/
#     Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/
#     GNN_PhysicsNeMo_Official
#   root@...:/workspace/.../GNN_PhysicsNeMo_Official# bash scripts/run_training.sh
# ============================================================

set -euo pipefail

# Exact path in your container
REPO_DIR="/workspace/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/GNN_PhysicsNeMo_Official"

echo "Working directory: $REPO_DIR"
cd "$REPO_DIR"

# Verify dataset exists
DATASET="/workspace/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/Dataset_creation/dataset_cylinder_features.h5"
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Run: find /workspace -name 'dataset_cylinder_features.h5'"
    exit 1
fi
echo "Dataset found: $DATASET"

# Install extra packages if needed
pip install -q torch-geometric h5py matplotlib scipy 2>/dev/null || true

# Start training
python3 train.py --epochs 200 --lr 1e-3 --batch 4 --device cuda