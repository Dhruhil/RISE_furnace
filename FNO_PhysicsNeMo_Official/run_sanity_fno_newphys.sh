#!/bin/bash
#SBATCH --job-name=fno_sanity_newphys
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/sanity_newphys_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/sanity_newphys_err_%j.log
#SBATCH --time=01:00:00
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

echo "=== FNO SANITY TEST — NEW PROFESSIONAL PHYSICS LOSS ==="
echo "Start: $(date)"
echo "Testing: Fourier + Newton + Stefan-Boltzmann + Energy + Spectral smoothness"
echo "Config:  1 epoch, batch=4, lr=5e-4, lam=0.0001"
echo ""
echo "NOTE: NOT using --test mode (that skips physics_loss_3d)."
echo "      Using full 1-epoch training to actually exercise the new physics loss."
echo ""

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 1 --lr 5e-4 --batch 4 --lam 0.0001

echo ""
echo "=== DONE: $(date) ==="
