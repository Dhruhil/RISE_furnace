#!/bin/bash
#SBATCH --job-name=gnn_sweep
#SBATCH --account=NAISS2026-4-712
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/sweep_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs/sweep_err_%j.log
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

SWEEP_DIR=/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/hp_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p $SWEEP_DIR
mkdir -p /mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/logs

cd /mimer/NOBACKUP/groups/revar/GNN_Unified
module purge 2>/dev/null
unset PYTHONPATH

SUMMARY=$SWEEP_DIR/summary.txt
echo "=== HP Sweep: 3 LR x 3 lam, 30 epochs each ===" | tee $SUMMARY
echo "Start: $(date)" | tee -a $SUMMARY

LRS=(1e-3 3e-4 1e-4)
LAMS=(0.0001 0.0005 0.002)

RUN=0
for LR in "${LRS[@]}"; do
  for LAM in "${LAMS[@]}"; do
    RUN=$((RUN+1))
    TAG="run${RUN}_lr${LR}_lam${LAM}"
    RUN_LOG=$SWEEP_DIR/${TAG}.log

    echo "" | tee -a $SUMMARY
    echo "=== [$RUN/9] LR=$LR lam=$LAM ===" | tee -a $SUMMARY
    echo "Start: $(date)" | tee -a $SUMMARY

    apptainer exec --nv --cleanenv \
      /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
      python -u train_unified.py \
        --epochs 30 --lr $LR --lam $LAM --batch 4 \
        2>&1 | tee $RUN_LOG

    LAST=$(grep -E "^Epoch" $RUN_LOG | tail -1)
    echo "Final: $LAST" | tee -a $SUMMARY
    echo "End: $(date)" | tee -a $SUMMARY
  done
done

echo "" | tee -a $SUMMARY
echo "=== All 9 runs complete ===" | tee -a $SUMMARY
echo "Logs in: $SWEEP_DIR" | tee -a $SUMMARY
