#!/bin/bash
#SBATCH --job-name=heat_fno_v4F
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/FINAL_RUN_v4_10pct_20260420_1157/logs/fno_v4_%j.log
#SBATCH --error=outputs/FINAL_RUN_v4_10pct_20260420_1157/logs/fno_v4_err_%j.log
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "============================================================"
echo "  FNO v4 FINAL - 200 EPOCHS (10% LR warmup)"
echo "  LR warmup: 20 epochs (10% of total)"
echo "  Pushforward: 20 epochs (10% - ALIGNED!)"
echo "  Config: static lam=0.003, AdamW, noise ch0, WD=1e-4"
echo "  GPU: A40, Time budget: 72 hours"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 200 --lr 1e-4 --batch 4 --lam 0.003

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
