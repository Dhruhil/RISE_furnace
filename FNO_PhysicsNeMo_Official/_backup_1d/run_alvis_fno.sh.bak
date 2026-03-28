#!/bin/bash
#SBATCH --job-name=heat_fno
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/checkpoints
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== FNO TRAINING — ALL REGIONS + PHYSICS ==="
echo "=== Start: $(date) ==="

apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python train.py --epochs 300 --lr 1e-3 --batch 8 --modes 24 --layers 6 --latent 128

echo "=== DONE: $(date) ==="
