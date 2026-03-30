#!/bin/bash
# ============================================================
# GNN_Unified — DEEP DIAGNOSTIC + FIX
# Run from: /mimer/NOBACKUP/groups/revar/GNN_Unified
# ============================================================
set -u
cd /mimer/NOBACKUP/groups/revar/GNN_Unified

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DEEP DIAGNOSTIC: Target mismatch + Dataset path            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ──────────────────────────────────────────────────────────────
echo "═══ A. What does the dataset return as y? ═══"
echo ""
grep "y=torch" data/dataset_unified.py | head -1
echo ""
grep "dT1 = " data/dataset_unified.py | head -1
echo ""
echo "  → If y = dT1 = (T_tp1 - T_t - dT_mean) / dT_std"
echo "    then y is NORMALISED DELTA_T"
echo ""

# ──────────────────────────────────────────────────────────────
echo "═══ B. How does train_one_epoch denorm predictions? ═══"
echo ""
echo "  In training loss:"
grep -n "loss1 = " train_unified.py | head -2
echo ""
echo "  In pushforward (getting T for next step):"
grep -n "T_pred1 = " train_unified.py | head -2
echo ""

# ──────────────────────────────────────────────────────────────
echo "═══ C. How does evaluate() denorm predictions? ═══"
echo ""
grep -n "T_pred = pred" train_unified.py | head -2
grep -n "T_true = batch" train_unified.py | head -2
echo ""

# ──────────────────────────────────────────────────────────────
echo "═══ D. How does physics_loss_unified denorm? ═══"
echo ""
grep -n "T_next = pred" train_unified.py | head -2
echo ""

# ──────────────────────────────────────────────────────────────
echo "═══ E. What does evaluate_unified.py use for rollout? ═══"
echo ""
grep -n "T_next = pred\|dT_pred\|T_cur =" evaluation/evaluate_unified.py 2>/dev/null | head -5
echo ""

# ──────────────────────────────────────────────────────────────
echo "═══ F. Dataset path resolution ═══"
echo ""
python3 << 'PYEOF'
import sys, os
sys.path.insert(0, ".")
from configs.base_config import CONFIG

print(f"  all_regions_dataset_path = {CONFIG.all_regions_dataset_path}")
print(f"  Exists: {os.path.exists(CONFIG.all_regions_dataset_path)}")
print()

# Also check common locations
candidates = [
    CONFIG.all_regions_dataset_path,
    "/mimer/NOBACKUP/groups/revar/GNN_Unified/dataset_all_regions_66cases.h5",
    "/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/dataset_all_regions.h5",
    "/mimer/NOBACKUP/groups/revar/dataset_all_regions_66cases.h5",
    "/mimer/NOBACKUP/groups/revar/dataset_all_regions.h5",
]
print("  Scanning for dataset files:")
for c in candidates:
    exists = os.path.exists(c)
    size = ""
    if exists:
        mb = os.path.getsize(c) / 1024 / 1024
        size = f" ({mb:.1f} MB)"
    print(f"    {'✅' if exists else '❌'} {c}{size}")

# Also search for any .h5 files nearby
import glob
h5s = glob.glob("/mimer/NOBACKUP/groups/revar/**/*all_regions*.h5", recursive=True)
if h5s:
    print()
    print("  All matching .h5 files found:")
    for h in sorted(h5s):
        mb = os.path.getsize(h) / 1024 / 1024
        print(f"    {h} ({mb:.1f} MB)")
PYEOF

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ G. Checkpoint model_cfg (what was it trained with?) ═══"
echo ""
python3 << 'PYEOF'
import torch, sys
try:
    ckpt = torch.load("outputs/checkpoints_unified/best_model.pt",
                       map_location="cpu", weights_only=False)
    print(f"  Epoch:     {ckpt.get('epoch', '?')}")
    print(f"  Metrics:   {ckpt.get('metrics', {})}")
    m = ckpt.get("model_cfg", {})
    print(f"  node_in:   {m.get('node_in_features', '?')}")
    print(f"  edge_in:   {m.get('edge_in_features', '?')}")
    print(f"  hidden:    {m.get('hidden_features', '?')}")
    print(f"  layers:    {m.get('n_message_passing_layers', '?')}")
    print(f"  output:    {m.get('output_features', '?')}")
    print(f"  Backend:   {ckpt.get('backend', '?')}")

    # Check first layer weight shape to confirm actual input dim
    state = ckpt.get("model_state", {})
    for k, v in state.items():
        if "encoder" in k and "weight" in k:
            print(f"  First encoder weight: {k} → shape {v.shape}")
            break
except Exception as e:
    print(f"  Error: {e}")
PYEOF

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ H. The full picture: what interpretation is the model using? ═══"
echo ""
python3 << 'PYEOF'
# Read the actual training code to determine the truth
with open("train_unified.py") as f:
    code = f.read()

# Check 1: What does the training loss compare?
# If it does F.mse_loss(pred, batch.y) directly, and batch.y = dT_norm,
# then the model learns to predict dT_norm.
# But if pushforward does T_pred1 = pred * T_std + T_mean,
# that's treating pred as normalised T_next.

issues = []

# Detect: dataset target
with open("data/dataset_unified.py") as f:
    ds_code = f.read()

if "dT1 = ((T_tp1 - T_t" in ds_code:
    target_type = "delta_T_normalised"
    print("  Dataset target (y): normalised delta_T")
    print("    y = (T(t+1) - T(t) - dT_mean) / dT_std")
else:
    target_type = "unknown"
    print("  Dataset target: UNKNOWN — check manually")

# Detect: training loss
if "F.mse_loss(pred" in code or "weighted_mse(pred" in code:
    print("  Training loss: MSE between pred and batch.y (= normalised dT)")
    print("    → Model output = normalised delta_T")

# Detect: pushforward denorm
if "pred1.squeeze(-1).detach() * T_std_ds + T_mean" in code:
    print()
    print("  ⚠️  PUSHFORWARD BUG:")
    print("    T_pred1 = pred * T_std + T_mean")
    print("    But pred is normalised delta_T, NOT normalised T_next!")
    print("    Correct: T_pred1 = T_current + (pred * dT_std + dT_mean)")
    issues.append("pushforward_denorm")
elif "pred1.squeeze(-1).detach() * dT_std" in code:
    print("  Pushforward: uses dT_std (consistent with delta_T target)")

# Detect: evaluate denorm
if "pred.squeeze(-1) * T_std_ds + T_mean" in code:
    # Check if this is in evaluate()
    lines = code.split("\n")
    in_eval = False
    for line in lines:
        if "def evaluate(" in line:
            in_eval = True
        if in_eval and "pred.squeeze(-1) * T_std_ds + T_mean" in line:
            print()
            print("  ⚠️  EVALUATE BUG:")
            print("    T_pred = pred * T_std + T_mean")
            print("    But pred is normalised delta_T!")
            print("    Correct: T_pred = T_current + (pred * dT_std + dT_mean)")
            issues.append("evaluate_denorm")
            break
        if in_eval and "def " in line and "evaluate" not in line:
            break

# Detect: physics loss denorm
if "T_next = pred.squeeze(-1) * dT_std + dT_mean" in code:
    print()
    print("  ⚠️  PHYSICS LOSS:")
    print("    T_next = pred * dT_std + dT_mean")
    print("    This gives delta_T (not T_next)! Variable name is misleading.")
    print("    Should be: dT_pred = pred * dT_std + dT_mean")
    print("               T_next = T_current + dT_pred")
    issues.append("physics_denorm")

print()
print("  ════════════════════════════════════════")
if not issues:
    print("  ✅ No denormalization issues found")
else:
    print(f"  ❌ {len(issues)} denormalization issue(s):")
    for iss in issues:
        print(f"     - {iss}")
    print()
    print("  ROOT CAUSE: The dataset returns normalised delta_T,")
    print("  but multiple places treat the model output as")
    print("  normalised T_next (denorming with T_std/T_mean).")
    print()
    print("  The model LEARNS normalised delta_T (because that's")
    print("  what batch.y is). So everywhere that converts model")
    print("  output to Kelvin must do:")
    print("    dT = pred * dT_std + dT_mean")
    print("    T_next = T_current + dT")
    print("  NOT:")
    print("    T_next = pred * T_std + T_mean")
PYEOF

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Paste the full output back to Claude and I'll write the fix."
echo "══════════════════════════════════════════════════════════════"
echo ""
