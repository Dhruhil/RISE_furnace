#!/bin/bash
#SBATCH --job-name=heat_fno_v5F
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/FNO_v5_FIX_150ep_20260425_1040/logs/fno_v5_%j.log
#SBATCH --error=outputs/FNO_v5_FIX_150ep_20260425_1040/logs/fno_v5_err_%j.log
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "============================================================"
echo "  FNO v5 RETRAIN with kappa fix — 150 EPOCHS on A100"
echo "  Output dir: outputs/FNO_v5_FIX_150ep_20260425_1040/"
echo "  Checkpoints: outputs/FNO_v5_FIX_150ep_20260425_1040/checkpoints/"
echo "  Bug fix: kappa now correctly 80 from updated HDF5 attrs"
echo "  Time budget: 72 hours"
echo "============================================================"
echo "Start: $(date)"

if ! grep -q "args.checkpoint_dir" train.py; then
    echo "ERROR: train.py patch missing — aborting!"
    exit 1
fi
echo "✓ Patch verified"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py \
    --epochs 150 \
    --lr 1e-4 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/FNO_v5_FIX_150ep_20260425_1040/checkpoints

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
