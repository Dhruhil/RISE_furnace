#!/bin/bash
#SBATCH --job-name=heat_gnn_full
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/training_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/error_%j.log
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs
cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official

apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python train.py --batch 8 --hidden 128 --layers 15 --epochs 200 --lr 1e-3 \
  2>&1 | tee outputs/logs/training_$(date +%Y%m%d_%H%M%S).log
