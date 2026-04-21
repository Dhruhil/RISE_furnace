#!/bin/bash
#SBATCH --job-name=gnn_1e4
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/test_1e4_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/test_1e4_err_%j.log
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

cd /mimer/NOBACKUP/groups/revar/GNN_Unified
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_unified.py --epochs 1 --lr 1e-4 --batch 4
