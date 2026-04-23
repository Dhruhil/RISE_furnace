#!/bin/bash
#SBATCH --job-name=heat_deeponet
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official/outputs/logs/train_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official/outputs/logs/train_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official

echo "============================================================"
echo "  DeepONet — 100-epoch run"
echo "  Dataset: dataset_v2_all_regions_clean.h5 (78 cases)"
echo "  LR=1e-3  batch=4  lambda=0.003"
echo "============================================================"
echo "Start: $(date)"

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 100 --lr 1e-3 --batch 4 --lam 0.003

echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
