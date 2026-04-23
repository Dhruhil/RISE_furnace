#!/bin/bash
#SBATCH --job-name=don_diag
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/logs/diag_%j.log
#SBATCH --error=outputs/logs/diag_err_%j.log
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

module purge 2>/dev/null
unset PYTHONPATH
cd /mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official
apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u diagnose_rollout.py
