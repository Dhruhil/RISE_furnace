#!/bin/bash
#SBATCH --job-name=heat_deeponet_v4F
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official/outputs/logs/deeponet_v4_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official/outputs/logs/deeponet_v4_err_%j.log
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official

echo "============================================================"
echo "  DeepONet v4 FINAL - 200 EPOCHS (matches GNN/FNO protocol)"
echo "  LR:         1e-4 (matches FNO)"
echo "  LR warmup:  20 epochs (10% of total)"
echo "  Pushforward: 10% warmup, ramps 0 -> 1.0"
echo "  Config:     static lam=0.003, AdamW, WD=1e-4, noise=0.01"
echo "  Physics:    0.4 cond + 0.3 conv + 0.2 rad + 0.1 eng"
echo "  GPU:        A40, Time budget: 72 hours"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 200 --lr 1e-4 --batch 4 --lam 0.003

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
