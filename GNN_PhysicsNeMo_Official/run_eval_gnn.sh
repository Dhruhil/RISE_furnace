#!/bin/bash
#SBATCH --job-name=gnn_eval
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/logs/eval_%j.log
#SBATCH --error=outputs/logs/eval_err_%j.log
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u evaluation/evaluate.py --device cuda
