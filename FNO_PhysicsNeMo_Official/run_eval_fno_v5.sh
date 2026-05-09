#!/bin/bash
#SBATCH --job-name=fno_v5_eval
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/logs/eval_v5_%j.log
#SBATCH --error=outputs/logs/eval_v5_err_%j.log
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

CKPT="outputs/FNO_v5_FIX_150ep_20260425_1040/checkpoints/best_model.pt"
OUT="outputs/FNO_v5_FIX_150ep_20260425_1040/evaluation"

echo "============================================================"
echo "  FNO v5 EVALUATION"
echo "  Checkpoint: $CKPT"
echo "  Output:     $OUT"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  bash -c "PYTHONDONTWRITEBYTECODE=1 python -u evaluation/evaluate.py \
    --device cuda \
    --checkpoint $CKPT \
    --output_dir $OUT"

echo "Done: $(date)"
