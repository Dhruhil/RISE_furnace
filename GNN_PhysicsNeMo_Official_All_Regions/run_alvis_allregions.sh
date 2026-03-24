#!/bin/bash
#SBATCH --job-name=heat_gnn_allregions
#SBATCH --account=naiss2026-4-525
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/allregions_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/allregions_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/checkpoints_allregions
cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official

echo "=== TRAINING ALL REGIONS ==="
apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python train_all_regions.py --epochs 65 --lr 1e-3 --batch 4

echo ""
echo "=== EVALUATING ALL REGIONS ==="
apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python evaluation/evaluate.py --device cuda

echo "=== DONE ==="
