#!/bin/bash
#SBATCH --job-name=sanity_newphys
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/sanity_newphys_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/sanity_newphys_err_%j.log
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs
cd /mimer/NOBACKUP/groups/revar/GNN_Unified

echo "=== SANITY TEST — NEW PHYSICS LOSS ==="
echo "Start: $(date)"
echo "Testing: new physics_loss_unified with Fourier + Newton + Stefan-Boltzmann"
echo "Config:  1 epoch, batch=2, LR=3e-4, lam=0.0001 (safe)"
echo ""

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_unified.py \
    --epochs 1 --lr 3e-4 --lam 0.0001 --batch 2

echo ""
echo "=== DONE: $(date) ==="
