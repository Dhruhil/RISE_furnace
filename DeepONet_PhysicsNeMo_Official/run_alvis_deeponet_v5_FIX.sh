#!/bin/bash
#SBATCH --job-name=heat_dpo_v5F
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=outputs/DeepONet_v5_FIX_150ep_20260425_1114/logs/dpo_v5_%j.log
#SBATCH --error=outputs/DeepONet_v5_FIX_150ep_20260425_1114/logs/dpo_v5_err_%j.log
#SBATCH --time=66:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official

echo "============================================================"
echo "  DeepONet v5 RETRAIN with bug fixes — 150 EPOCHS on A100"
echo "  Output dir: outputs/DeepONet_v5_FIX_150ep_20260425_1114/"
echo "  Checkpoints: outputs/DeepONet_v5_FIX_150ep_20260425_1114/checkpoints/"
echo "  Bug fixes:"
echo "    - kappa = 80 (was 60)"
echo "    - branch_scalars expanded: 2 → 7 (added cx/cy/cz/r/h)"
echo "  Time budget: 66 hours"
echo "============================================================"
echo "Start: $(date)"

if ! grep -q "_parse_mm" data/dataset.py; then
    echo "ERROR: dataset.py patch missing — aborting!"
    exit 1
fi
if ! grep -q "branch_scalar_inputs: int = 7" configs/deeponet_config.py; then
    echo "ERROR: config patch missing — aborting!"
    exit 1
fi
if ! grep -q "args.checkpoint_dir" training/train.py; then
    echo "ERROR: train.py patch missing — aborting!"
    exit 1
fi
echo "✓ All patches verified"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py \
    --epochs 150 \
    --lr 1e-4 \
    --batch 4 \
    --lam 0.003 \
    --checkpoint_dir outputs/DeepONet_v5_FIX_150ep_20260425_1114/checkpoints

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
