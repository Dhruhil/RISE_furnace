#!/bin/bash
#SBATCH --job-name=heat_gnn_v4F
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/FINAL_RUN_GNN_v4_200ep_20260420_1043/logs/gnn_v4_%j.log
#SBATCH --error=outputs/FINAL_RUN_GNN_v4_200ep_20260420_1043/logs/gnn_v4_err_%j.log
#SBATCH --time=74:00:00
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/GNN_Unified

echo "============================================================"
echo "  GNN v4 FINAL - 200 EPOCHS on A100fat"
echo "  Config: static lam=0.003, AdamW, noise ch3, WD=1e-4"
echo "  Matches FNO v4 FINAL for fair comparison"
echo "  GPU: A100fat (80GB, no OOM risk)"
echo "  Time budget: 74 hours"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_unified.py --epochs 200 --lr 5e-5 --batch 4 --lam 0.003

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
