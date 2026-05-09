#!/bin/bash
#SBATCH --job-name=deeponet_sanity
#SBATCH --account=NAISS2026-4-712
#SBATCH --output=/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified/outputs/logs/%j_sanity.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified/outputs/logs/%j_sanity.err
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

BASE=/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified
CONTAINER=/mimer/NOBACKUP/groups/revar/jinis/deeponet_project/containers/physicsnemo.sif

mkdir -p $BASE/outputs/logs

echo "=== SANITY TEST ==="

apptainer exec --nv \
    --bind /mimer:/mimer \
    $CONTAINER \
    python3 $BASE/train_unified.py \
        --epochs 2 \
        --lr     5e-4 \
        --batch  2048 \
        --lam    0.0 \
        --ckpt_dir $BASE/outputs/checkpoints_unified \
        --log_dir  $BASE/outputs/logs

echo "=== SANITY TEST DONE ==="
