#!/bin/bash
#SBATCH --job-name=don_v5_eval
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/logs/eval_v5_%j.log
#SBATCH --error=outputs/logs/eval_v5_err_%j.log
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

cd /mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official

CKPT="outputs/DeepONet_v5_FIX_150ep_20260425_1114/checkpoints/best.pt"

echo "============================================================"
echo "  DeepONet v5 EVALUATION"
echo "  Checkpoint: $CKPT"
echo "  Phase 2: 2760-3600s (840s window, matches FNO)"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  bash -c "PYTHONDONTWRITEBYTECODE=1 python -u evaluation/evaluate_deeponet.py \
    --device cuda \
    --ckpt $CKPT \
    --n_sims 7"

echo "Done: $(date)"
