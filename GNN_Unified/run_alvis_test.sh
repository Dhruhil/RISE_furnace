#!/bin/bash
#SBATCH --job-name=gnn_test
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/test_lam%x_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/test_lam%x_err_%j.log
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

module purge 2>/dev/null
unset PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs

cd /mimer/NOBACKUP/groups/revar/GNN_Unified

LAM=${1:-0.01}

echo "============================================"
echo "GNN 1-EPOCH TEST"
echo "============================================"
echo "Start:   $(date)"
echo "Lambda:  $LAM"
echo "Batch:   2"
echo "Epochs:  1"
echo "GPU:     A40"
echo "============================================"
echo ""

apptainer exec --nv --cleanenv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train_unified.py \
    --epochs 1 --lr 3e-4 --lam $LAM --batch 2

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "============================================"
