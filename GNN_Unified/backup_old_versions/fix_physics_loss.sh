#!/bin/bash
# ============================================================
# FIX: physics_loss_unified() uses wrong denorm stats
#
# The dataset returns y = (T_tp1 - T_mean) / T_std
#   → normalised T_next (NOT delta_T, despite variable name dT1)
#
# evaluate() and pushforward correctly use:
#   T_pred = pred * T_std + T_mean  ✓
#
# But physics_loss_unified() incorrectly uses:
#   T_next = pred * dT_std + dT_mean  ✗
#   (dT_std ≈ 1K, dT_mean ≈ 0K → gives ~0K instead of ~600-1100K)
#
# Fix: change physics_loss to use T_std/T_mean
#
# Run with:  bash fix_physics_loss.sh
# From:      /mimer/NOBACKUP/groups/revar/GNN_Unified
# ============================================================
set -euo pipefail
cd /mimer/NOBACKUP/groups/revar/GNN_Unified

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  FIX: physics_loss_unified() denormalization"
echo "════════════════════════════════════════════════════════════"
echo ""

# Safety backup
cp train_unified.py train_unified.py.bak_before_physics_fix
echo "  Backup: train_unified.py.bak_before_physics_fix"

python3 << 'PYEOF'
with open("train_unified.py", "r") as f:
    code = f.read()

fixes = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 1: physics_loss_unified() — denorm with T_std/T_mean
#
#   The function signature passes dT_std, dT_mean but these are
#   the wrong stats. The pred is normalised T_next (not delta_T).
#
#   Change the denormalization line inside the function.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Pattern: the bad line inside physics_loss_unified
old_line = "    T_next = pred.squeeze(-1) * dT_std + dT_mean  # dT_std/mean used as T_std/mean here"
new_line = "    T_next = pred.squeeze(-1) * dT_std + dT_mean  # NOTE: dT_std/dT_mean are actually T_std/T_mean (see caller)"

# Actually, the real fix is at the CALLER — pass T_std/T_mean instead of dT_std/dT_mean.
# But let's check both the function and the caller.

# Check what the caller passes:
if "physics_loss_unified(pred1, batch, dT_std, dT_mean)" in code:
    print("  Caller passes: dT_std, dT_mean")
    print("  But pred is normalised T_next → needs T_std, T_mean")
    print()

    # APPROACH: Fix the caller to pass T_std, T_mean
    # The function uses the params as generic denorm stats.
    # The variable names inside the function (dT_std, dT_mean) are
    # just parameter names — we can pass T_std and T_mean to them.

    old_call_train = "physics_loss_unified(pred1, batch, dT_std, dT_mean)"
    new_call_train = "physics_loss_unified(pred1, batch, T_std_ds, T_mean)"

    code = code.replace(old_call_train, new_call_train)
    fixes += 1
    print("  ✓ FIX train_one_epoch: pass T_std_ds, T_mean to physics_loss")

if "physics_loss_unified(pred, batch, dT_std, dT_mean)" in code:
    old_call_eval = "physics_loss_unified(pred, batch, dT_std, dT_mean)"
    new_call_eval = "physics_loss_unified(pred, batch, T_std_ds, T_mean)"

    code = code.replace(old_call_eval, new_call_eval)
    fixes += 1
    print("  ✓ FIX evaluate: pass T_std_ds, T_mean to physics_loss")

# Update the comment inside physics_loss_unified to be accurate
old_comment = "    T_next = pred.squeeze(-1) * dT_std + dT_mean  # dT_std/mean used as T_std/mean here"
new_comment = "    T_next = pred.squeeze(-1) * dT_std + dT_mean  # caller now passes T_std, T_mean correctly"
if old_comment in code:
    code = code.replace(old_comment, new_comment)
    print("  ✓ Updated misleading comment in physics_loss_unified")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 2: Also fix evaluate_unified.py (rollout evaluation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# evaluate_unified.py:
#   T_next = pred.squeeze(-1).cpu().numpy() * T_std + T_mean
# This is CORRECT — pred is normalised T_next, T_std/T_mean are right.
# But we should verify T_std and T_mean come from the right source.

import os
eval_path = "evaluation/evaluate_unified.py"
if os.path.exists(eval_path):
    with open(eval_path) as f:
        eval_code = f.read()
    if "pred.squeeze(-1).cpu().numpy() * T_std + T_mean" in eval_code:
        print("  ✓ evaluate_unified.py: denorm is correct (pred * T_std + T_mean)")
    else:
        print("  ⚠ evaluate_unified.py: check denorm manually")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Write back
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with open("train_unified.py", "w") as f:
    f.write(code)

print()
print(f"  Total fixes applied: {fixes}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VERIFY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("  ════ VERIFICATION ════")
with open("train_unified.py") as f:
    final = f.read()

# Check: no more dT_std, dT_mean passed to physics_loss
if "physics_loss_unified(pred1, batch, T_std_ds, T_mean)" in final:
    print("  ✅ train_one_epoch: physics_loss gets T_std, T_mean")
else:
    print("  ❌ train_one_epoch: physics_loss call NOT fixed")

if "physics_loss_unified(pred, batch, T_std_ds, T_mean)" in final:
    print("  ✅ evaluate: physics_loss gets T_std, T_mean")
elif "physics_loss_unified(pred, batch, dT_std, dT_mean)" in final:
    print("  ❌ evaluate: physics_loss still gets dT_std, dT_mean")
else:
    print("  ⚠ evaluate: physics_loss call not found (may not use physics in eval)")

# Check pushforward is still correct (should NOT have been touched)
if "pred1.squeeze(-1).detach() * T_std_ds + T_mean" in final:
    print("  ✅ Pushforward: still uses T_std (correct, untouched)")
else:
    print("  ⚠ Pushforward: pattern changed — verify manually")

# Check evaluate denorm is still correct
if "pred.squeeze(-1) * T_std_ds + T_mean" in final:
    print("  ✅ evaluate() denorm: still uses T_std (correct, untouched)")
else:
    print("  ⚠ evaluate() denorm: pattern changed — verify manually")

# Show the actual physics_loss function header + first denorm line
print()
print("  ════ Physics loss function (first 5 lines of body) ════")
lines = final.split("\n")
in_func = False
count = 0
for line in lines:
    if "def physics_loss_unified" in line:
        in_func = True
        print(f"  {line}")
        continue
    if in_func:
        print(f"  {line}")
        count += 1
        if count >= 8:
            break

PYEOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cleanup stale .bak files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "  ════ Cleaning stale .bak files ════"
mkdir -p backup_old_versions
for f in configs/base_config.py.bak_before_fix \
         evaluation/evaluate_unified.py.bak_before_fix \
         train_unified.py.bak_before_fix; do
    if [ -f "$f" ]; then
        mv "$f" backup_old_versions/
        echo "  Moved: $f → backup_old_versions/"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DONE."
echo ""
echo "  What was fixed:"
echo "    physics_loss_unified() was receiving dT_std/dT_mean"
echo "    but pred is normalised T_next → needs T_std/T_mean."
echo "    The callers now pass T_std_ds, T_mean instead."
echo ""
echo "  What was NOT touched (already correct):"
echo "    - evaluate(): T_pred = pred * T_std + T_mean  ✓"
echo "    - pushforward: T_pred1 = pred * T_std + T_mean  ✓"
echo "    - evaluate_unified.py rollout  ✓"
echo ""
echo "  Next steps:"
echo "    1. diff train_unified.py train_unified.py.bak_before_physics_fix"
echo "    2. sbatch run_sanity_test.sh"
echo "    3. sbatch run_alvis_unified.sh"
echo "════════════════════════════════════════════════════════════"
echo ""
