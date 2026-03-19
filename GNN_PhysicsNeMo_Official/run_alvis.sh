#!/bin/bash
#SBATCH --job-name=heat_gnn_full
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=outputs/logs/training_%j.log
#SBATCH --error=outputs/logs/error_%j.log
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p outputs/logs
cd /cephyr/users/dhruhil/Alvis/GNN_PhysicsNeMo_Official

apptainer exec --nv \
  /apps/containers/physicsnemo_25.06.sif \
  python train.py --batch 8 --hidden 128 --layers 15 --epochs 200 \
  2>&1 | tee outputs/logs/training_$(date +%Y%m%d_%H%M%S).log
