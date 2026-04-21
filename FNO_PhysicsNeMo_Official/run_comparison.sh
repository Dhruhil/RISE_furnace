#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/comparison_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/comparison_err_%j.log
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u plots_comparison.py
