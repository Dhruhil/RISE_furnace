#!/bin/bash
# ============================================================
# Inference launcher
# ============================================================
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SIM_IDX=${1:-0}
TARGET_TIME=${2:-3000}

echo "Running inference: sim_idx=$SIM_IDX  target_time=$TARGET_TIME s"
echo "Note: predict at any time including t=3000 s from 0-4000 s trained model"

python3 infer.py \
  --sim_idx "$SIM_IDX" \
  --target_time "$TARGET_TIME" \
  --device cuda

echo ""
echo "--- Multi-time rollout (500, 1000, 2000, 3000, 4000 s) ---"
python3 infer.py \
  --rollout \
  --sim_idx "$SIM_IDX" \
  --target_times 500 1000 2000 3000 4000 \
  --device cuda