#!/bin/bash
#SBATCH --job-name=heat_pinn
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/PINN_PhysicsNeMo_Official/outputs/logs/pinn_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/PINN_PhysicsNeMo_Official/outputs/logs/pinn_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p /mimer/NOBACKUP/groups/revar/PINN_PhysicsNeMo_Official/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/PINN_PhysicsNeMo_Official/outputs/checkpoints
cd /mimer/NOBACKUP/groups/revar/PINN_PhysicsNeMo_Official

echo "=== PINN TRAINING — GPU-Optimized ==="
echo "=== Start: $(date) ==="

apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python train.py --pretrain_epochs 3000 --physics_epochs 5000 --batch 32768

echo "=== DONE: $(date) ==="
