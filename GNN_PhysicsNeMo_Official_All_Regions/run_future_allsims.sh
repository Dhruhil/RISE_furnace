#!/bin/bash
#SBATCH --job-name=future_all
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/future_all_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/future_all_err_%j.log
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official

echo "=== Sim 0 to 8000s ==="
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python inference/infer_allregions.py --target_time 8000 --sim_idx 0 --device cuda

echo "=== Sim 1 to 8000s ==="
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python inference/infer_allregions.py --target_time 8000 --sim_idx 1 --device cuda

echo "=== Sim 2 to 8000s ==="
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python inference/infer_allregions.py --target_time 8000 --sim_idx 2 --device cuda

echo "=== DONE ==="
