#!/bin/bash
#SBATCH --job-name=heat_fno
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/checkpoints
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== TRAINING FNO — ALL REGIONS ==="
apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python train.py --epochs 200 --lr 1e-3 --batch 16 --modes 16 --layers 4 --latent 64

echo "=== DONE ==="
