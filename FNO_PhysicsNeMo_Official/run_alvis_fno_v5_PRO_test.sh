#!/bin/bash
#SBATCH --job-name=v5PRO_test
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/v5_PRO_test_5ep/logs/v5pro_test_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/v5_PRO_test_5ep/logs/v5pro_test_err_%j.log
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/v5_PRO_test_5ep/logs

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "============================================================"
echo "  FNO v5 PRO - TEST RUN (5 EPOCHS)"
echo "  Purpose: Verify adaptive lambda + AdamW + noise work"
echo "  If this works, submit full 100-epoch run"
echo "============================================================"
echo "  Expected behavior:"
echo "    Ep 1: lam=0.003 (init), no adaptive yet"
echo "    Ep 2: lam computed from ep 1 losses (adaptive!)"
echo "    Ep 3-5: lam continues adapting"
echo "    All epochs: noise injection active"
echo "    Optimizer: AdamW"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_v5_pro.py --epochs 3 --lr 1e-4 --batch 4

echo "============================================================"
echo "  TEST DONE: $(date)"
echo "  Check if lam column shows DIFFERENT values each epoch"
echo "  If yes -> adaptive working, submit full run"
echo "============================================================"
