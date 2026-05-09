#!/bin/bash
#SBATCH --job-name=gnn_v5_eval
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/GNN_v5_FIX_150ep_20260425_0948/logs/eval_%j.log
#SBATCH --error=outputs/GNN_v5_FIX_150ep_20260425_0948/logs/eval_err_%j.log
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

set -e

cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official

CHECKPOINT="outputs/GNN_v5_FIX_150ep_20260425_0948/checkpoints/best_model_eval_snapshot.pt"
OUTPUT_DIR="outputs/GNN_v5_FIX_150ep_20260425_0948/evaluation"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  GNN v5 ROLLOUT EVALUATION"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output:     $OUTPUT_DIR"
echo "  Started:    $(date)"
echo "============================================================"

apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u evaluation/evaluate.py \
    --device cuda \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
