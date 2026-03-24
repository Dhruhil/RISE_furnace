#!/bin/bash
#SBATCH --job-name=r2_both
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/r2_both_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/r2_both_err_%j.log
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official
echo "=== WITH CLAMPING ==="
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif python check_r2.py
echo ""
echo "=== WITHOUT CLAMPING ==="
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif python check_r2_no_clamp.py
