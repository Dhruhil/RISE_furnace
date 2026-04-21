#!/bin/bash
#SBATCH --job-name=hp_finder
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/hp_finder_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/hp_finder_err_%j.log
#SBATCH --time=09:00:00
#SBATCH --gpus-per-node=T4:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== HYPERPARAMETER FINDER ==="
echo "=== Start: $(date) ==="

apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u hp_finder.py

echo "=== DONE: $(date) ==="
