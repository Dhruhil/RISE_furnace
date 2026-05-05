#!/bin/bash
#SBATCH --job-name=heat_gnn_v5F
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/GNN_v5_FIX_150ep_20260425_0948/logs/gnn_v5_%j.log
#SBATCH --error=outputs/GNN_v5_FIX_150ep_20260425_0948/logs/gnn_v5_err_%j.log
#SBATCH --time=88:00:00
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/GNN_Unified

echo "============================================================"
echo "  GNN v5 RETRAIN with bug fix — 150 EPOCHS on A100fat"
echo "  Output dir: outputs/GNN_v5_FIX_150ep_20260425_0948/"
echo "  Checkpoints: outputs/GNN_v5_FIX_150ep_20260425_0948/checkpoints/"
echo "  Bug fix: cx/cy/cz/r/h now parsed from case names"
echo "============================================================"
echo "Start: $(date)"

if ! grep -q "_parse_mm" data/dataset_unified.py; then
    echo "ERROR: dataset_unified.py patch missing — aborting!"
    exit 1
fi
if ! grep -q "args.checkpoint_dir" train_unified.py; then
    echo "ERROR: train_unified.py patch missing — aborting!"
    exit 1
fi
echo "✓ Both patches verified"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_unified.py \
    --epochs 150 \
    --lr 5e-5 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/GNN_v5_FIX_150ep_20260425_0948/checkpoints

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
