#!/bin/bash
#SBATCH --job-name=gnn_test
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/sanity_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/sanity_err_%j.log
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official
module purge 2>/dev/null
unset PYTHONPATH

echo "=== SANITY TEST ==="
apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_unified.py --test --device cuda

echo "=== EXIT CODE: $? ==="
