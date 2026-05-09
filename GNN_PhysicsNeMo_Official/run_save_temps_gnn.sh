#!/bin/bash
#SBATCH --job-name=gnn_save_temps
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/GNN_v5_FIX_150ep_20260425_0948/logs/save_temps_%j.log
#SBATCH --error=outputs/GNN_v5_FIX_150ep_20260425_0948/logs/save_temps_err_%j.log
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official

CKPT="outputs/GNN_v5_FIX_150ep_20260425_0948/checkpoints/best_model_eval_snapshot.pt"
OUT="outputs/GNN_v5_FIX_150ep_20260425_0948/evaluation"

echo "============================================================"
echo "  GNN SAVE ROLLOUT TEMPERATURES"
echo "  Checkpoint: $CKPT"
echo "  Output:     $OUT"
echo "  Started:    $(date)"
echo "============================================================"

apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  bash -c "PYTHONDONTWRITEBYTECODE=1 python -u evaluation/save_rollout_temps.py \
    --device cuda \
    --checkpoint $CKPT \
    --output_dir $OUT"

echo "Done: $(date)"
