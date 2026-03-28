#!/bin/bash
#SBATCH --job-name=gnn_eval
#SBATCH --account=NAISS2026-4-525
#SBATCH --partition=alvis
#SBATCH --output=outputs/logs/eval_%j.log
#SBATCH --error=outputs/logs/eval_err_%j.log
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

cd /mimer/NOBACKUP/groups/revar/GNN_Unified
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u evaluation/evaluate_unified.py --device cuda --n_sims 5
