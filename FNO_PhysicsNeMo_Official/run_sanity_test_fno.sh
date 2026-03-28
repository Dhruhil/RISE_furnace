#!/bin/bash
#SBATCH --job-name=fno_test
#SBATCH --account=NAISS2026-4-525
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/test_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/test_err_%j.log
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== 3D FNO SANITY TEST ==="
apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --test --device cuda

echo "=== EXIT CODE: $? ==="
