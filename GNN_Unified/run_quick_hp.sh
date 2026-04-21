#!/bin/bash
#SBATCH --job-name=gnn_hp
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/hp_test_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/hp_test_err_%j.log
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

cd /mimer/NOBACKUP/groups/revar/GNN_Unified

echo "=== GNN Quick HP — 1 epoch × 5 LRs ==="

for LR in 1e-3 7e-4 5e-4 3e-4 1e-4; do
  echo ""
  echo "========== LR=$LR =========="
  apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
    python -u train_unified.py --epochs 1 --lr $LR --batch 4
done

echo "=== DONE ==="
