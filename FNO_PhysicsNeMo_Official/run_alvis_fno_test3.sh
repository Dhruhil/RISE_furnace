#!/bin/bash
#SBATCH --job-name=fno_test3
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno_test3_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno_test3_err_%j.log
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/checkpoints

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "============================================================"
echo "  FNO 3-EPOCH TEST"
echo "  Purpose: Verify physics loss magnitudes, check training"
echo "============================================================"
echo "Start: $(date)"
echo ""
echo "Config: epochs=3, lr=1e-4, batch=4, lam=0.001"
echo "GPU: A40 (cheaper than A100fat)"
echo ""

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 3 --lr 1e-4 --batch 4 --lam 0.003

echo ""
echo "============================================================"
echo "  TEST DONE: $(date)"
echo "============================================================"
