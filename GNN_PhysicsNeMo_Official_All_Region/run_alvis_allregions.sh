#!/bin/bash
#SBATCH --job-name=heat_gnn_allregions
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/allregions_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/allregions_err_%j.log
#SBATCH --time=96:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs
cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official

apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python train_all_regions.py \
  --epochs 50 --lr 1e-3 --batch 4 \
  2>&1 | tee outputs/logs/allregions_$(date +%Y%m%d_%H%M%S).log
