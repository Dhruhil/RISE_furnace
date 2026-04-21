#!/bin/bash
#SBATCH --job-name=heat_fno_FINAL
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/FINAL_RUN_20260419_1414/logs/fno_final_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/FINAL_RUN_20260419_1414/logs/fno_final_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

FINAL_DIR="outputs/FINAL_RUN_20260419_1414"

echo "============================================================"
echo "  FNO v4 — FINAL 100-EPOCH RUN"
echo "  Config: L_conv /100, L_rad /1000, lam=0.003"
echo "  Dataset: v2 (78 cases), LR=1e-4, batch=4"
echo "  GPU: A40"
echo "  Output: ${FINAL_DIR}"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 100 --lr 1e-4 --batch 4 --lam 0.003

# Copy important outputs to final folder after training
echo ""
echo "=== Copying results to final folder ==="

mkdir -p ${FINAL_DIR}/checkpoints
mkdir -p ${FINAL_DIR}/rollout_results

# Copy checkpoints
cp -r outputs/checkpoints/* ${FINAL_DIR}/checkpoints/ 2>/dev/null

# Copy any other outputs
cp -r outputs/*.png ${FINAL_DIR}/ 2>/dev/null
cp -r outputs/*.json ${FINAL_DIR}/ 2>/dev/null
cp -r outputs/*.csv ${FINAL_DIR}/ 2>/dev/null

echo "Final outputs saved to: ${FINAL_DIR}"
ls -la ${FINAL_DIR}/

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
