#!/bin/bash
#SBATCH --job-name=heat_fno_v3
#SBATCH --account=NAISS2026-4-525
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno3d_v3_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno3d_v3_err_%j.log
#SBATCH --time=40:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/checkpoints
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== 3D FNO v3 — grid 30x36x54, lr=7e-4, wd=0, pushforward→1.0, 200ep ==="
echo "=== Start: $(date) ==="

apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 200 --lr 7e-4 --batch 4

echo "=== DONE: $(date) ==="
