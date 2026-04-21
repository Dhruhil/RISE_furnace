#!/bin/bash
#SBATCH --job-name=heat_gnn_unified
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/unified_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/unified_err_%j.log
#SBATCH --time=15:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

mkdir -p /mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/checkpoints_unified
cd /mimer/NOBACKUP/groups/revar/GNN_Unified
module purge 2>/dev/null
unset PYTHONPATH

echo "=== UNIFIED GNN — T_next, KNN=12, Layers=4 ==="
echo "=== Start: $(date) ==="

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_unified.py --epochs 15 --lr 1e-4 --batch 4

echo "=== DONE: $(date) ==="
