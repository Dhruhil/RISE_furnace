#!/bin/bash
#SBATCH --job-name=deeponet_v4_FINAL
#SBATCH --account=NAISS2026-4-712
#SBATCH --output=/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified/outputs/logs/%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified/outputs/logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

BASE=/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified
CONTAINER=/mimer/NOBACKUP/groups/revar/jinis/deeponet_project/containers/physicsnemo.sif

mkdir -p $BASE/outputs/logs
mkdir -p $BASE/outputs/checkpoints_unified

echo "========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Start    : $(date)"
echo "========================================="

apptainer exec --nv \
    --bind /mimer:/mimer \
    $CONTAINER \
    python3 $BASE/train_unified.py \
        --epochs 200 \
        --lr     5e-5 \
        --batch  4096 \
        --lam    0.003 \
        --ckpt_dir $BASE/outputs/checkpoints_unified \
        --log_dir  $BASE/outputs/logs

echo "========================================="
echo "End      : $(date)"
echo "========================================="
