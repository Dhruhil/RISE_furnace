#!/bin/bash
#SBATCH --job-name=fno_eval
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/logs/eval_%j.log
#SBATCH --error=outputs/logs/eval_err_%j.log
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  bash -c "PYTHONDONTWRITEBYTECODE=1 python -u evaluation/evaluate_fno3d.py --device cuda --n_sims 5"
