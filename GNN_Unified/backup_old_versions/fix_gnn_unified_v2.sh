#!/bin/bash
# ============================================================
# GNN_Unified — CORRECTED FIX SCRIPT
# Paste these commands directly into your Alvis terminal
#
# Only fixes CONFIRMED bugs:
#   1. base_config.py: node_in_features 11 → 16
#   2. train_unified.py: remove duplicated region block
#   3. train_unified.py: hardcoded dataset path → config
#   4. train_unified.py: hardcoded pushforward time → config
#   5. evaluate_unified.py: default cx/kappa/Cp values
#   6. evaluate_unified.py: is_heater int vs string bug
#   7. Cleanup: stale .bak files, stray "0.5" file
#
# Region thresholds are CORRECT as-is (outer_box=11, >0.95 works)
# ============================================================

cd /mimer/NOBACKUP/groups/revar/GNN_Unified

# ──────────────────────────────────────────────────────────────
# STEP 1: Fix base_config.py — node_in_features 11 → 16
# ──────────────────────────────────────────────────────────────
echo "=== STEP 1: Fix base_config.py ==="
cp configs/base_config.py configs/base_config.py.bak_before_fix
sed -i 's/node_in_features:         int = 11/node_in_features:         int = 16/' configs/base_config.py
echo "Done. Verify:"
grep "node_in_features" configs/base_config.py

# ──────────────────────────────────────────────────────────────
# STEP 2: Fix train_unified.py — remove duplicated block
#   The evaluate() function has the region-masking block
#   (rid = batch.x[:, 5] ... region_trues) pasted TWICE.
#   We use Python to surgically remove only the second copy.
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 2: Remove duplicated region block in train_unified.py ==="
cp train_unified.py train_unified.py.bak_before_fix

python3 << 'PYEOF'
with open("train_unified.py", "r") as f:
    lines = f.readlines()

# Find all lines that start with "        rid = batch.x[:, 5]" inside evaluate()
# The duplicated block is 8 lines starting from the second occurrence of this line
occurrences = []
for i, line in enumerate(lines):
    if "rid = batch.x[:, 5]" in line:
        occurrences.append(i)

if len(occurrences) >= 2:
    # The second occurrence is the duplicate — remove it and the next 7 lines (8 total)
    dup_start = occurrences[1]
    # The block is:
    #   rid = batch.x[:, 5]
    #   is_steel = ...
    #   is_air   = ...
    #   is_outer = ...
    #   for name, mask in ...:
    #       if mask.any():
    #           region_preds[name]...
    #           region_trues[name]...
    # Plus the blank line before it = 9 lines to check
    
    # Count how many lines to remove: from "rid = " to the last "region_trues" line
    end = dup_start
    for j in range(dup_start, min(dup_start + 12, len(lines))):
        if "region_trues" in lines[j] or "region_preds" in lines[j]:
            end = j
    
    # Also remove the blank line before if present
    if dup_start > 0 and lines[dup_start - 1].strip() == "":
        dup_start -= 1
    
    removed = lines[dup_start:end+1]
    print(f"  Removing lines {dup_start+1} to {end+1} ({end-dup_start+1} lines):")
    for r in removed:
        print(f"    {r.rstrip()}")
    
    new_lines = lines[:dup_start] + lines[end+1:]
    with open("train_unified.py", "w") as f:
        f.writelines(new_lines)
    print(f"  Done: removed duplicate block")
else:
    print(f"  Found {len(occurrences)} occurrence(s) of rid = batch.x[:, 5]")
    if len(occurrences) == 1:
        print("  Already clean — no duplicate to remove")
    else:
        print("  WARNING: unexpected count, check manually")
PYEOF

echo "Verify (should show exactly 1 occurrence):"
grep -c "rid = batch.x\[:, 5\]" train_unified.py

# ──────────────────────────────────────────────────────────────
# STEP 3: Fix hardcoded dataset path in train_unified.py
#   "dataset_all_regions.h5" → cfg.all_regions_dataset_path
#   This picks up the 66-case dataset from base_config.py
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 3: Fix dataset path in train_unified.py ==="
sed -i 's|    h5 = "dataset_all_regions.h5"|    h5 = cfg.all_regions_dataset_path|' train_unified.py
echo "Verify:"
grep "h5 = " train_unified.py

# ──────────────────────────────────────────────────────────────
# STEP 4: Remove redundant node_in_features override
#   Now that config says 16, no need to override in main()
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 4: Remove redundant node_in override ==="
sed -i 's/    cfg.node_in_features = 16/    # node_in_features = 16 (now default in base_config.py)/' train_unified.py
echo "Verify:"
grep "node_in_features" train_unified.py

# ──────────────────────────────────────────────────────────────
# STEP 5: Fix hardcoded pushforward time increment
#   10.0 / 4000.0 → cfg.dt / cfg.t_total
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 5: Fix pushforward time constant ==="
sed -i 's|batch2.x\[:, 6\] = batch.x\[:, 6\] + 10.0 / 4000.0|batch2.x[:, 6] = batch.x[:, 6] + cfg.dt / cfg.t_total|' train_unified.py
echo "Verify:"
grep "cfg.dt" train_unified.py

# ──────────────────────────────────────────────────────────────
# STEP 6: Fix evaluate_unified.py — wrong default values
#   cx: 0.103 → 0.0   (matches grp.attrs.get("cx", 0.0))
#   kappa: 55.0 → 60.0 (matches grp.attrs.get("kappa", 60.0))
#   Cp: 500.0 → 450.0  (matches grp.attrs.get("Cp", 450.0))
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 6: Fix evaluate_unified.py defaults ==="
cp evaluation/evaluate_unified.py evaluation/evaluate_unified.py.bak_before_fix

sed -i 's/sim.get("cx", 0.103)/sim.get("cx", 0.0)/g' evaluation/evaluate_unified.py
sed -i 's/sim.get("kappa", 55.0)/sim.get("kappa", 60.0)/g' evaluation/evaluate_unified.py
sed -i 's/sim.get("Cp", 500.0)/sim.get("Cp", 450.0)/g' evaluation/evaluate_unified.py
echo "Verify defaults:"
grep 'sim.get("cx"' evaluation/evaluate_unified.py
grep 'sim.get("kappa"' evaluation/evaluate_unified.py
grep 'sim.get("Cp"' evaluation/evaluate_unified.py

# ──────────────────────────────────────────────────────────────
# STEP 7: Fix is_heater bug in evaluate_unified.py
#   rid is an integer (0-11), HEATER_REGIONS has strings
#   "rid in HEATER_REGIONS" always returns False!
#   Fix: use rname (the string name) instead
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 7: Fix is_heater check in evaluate_unified.py ==="
sed -i 's/is_h = rid in HEATER_REGIONS/is_h = rname in HEATER_REGIONS/g' evaluation/evaluate_unified.py
echo "Verify:"
grep "is_h = " evaluation/evaluate_unified.py

# ──────────────────────────────────────────────────────────────
# STEP 8: Clean up stale .bak files and stray "0.5"
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 8: Cleanup ==="
mkdir -p backup_old_versions

for f in \
    data/dataset_unified.py.bak \
    data/dataset_unified.py.bak_rid \
    data/dataset_unified.py.bak_cyl \
    models/meshgraphnet.py.bak_final \
    evaluation/evaluate_unified.py.bak \
    configs/base_config.py.bak; do
    if [ -f "$f" ]; then
        mv "$f" backup_old_versions/
        echo "  Moved: $f"
    fi
done

if [ -e "0.5" ]; then
    rm -rf "0.5"
    echo "  Removed stray '0.5'"
fi

# ──────────────────────────────────────────────────────────────
# VERIFICATION
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  VERIFICATION"
echo "============================================================"

python3 << 'PYEOF'
ok = 0
fail = 0

# 1. base_config node_in
with open("configs/base_config.py") as f:
    c = f.read()
if "node_in_features:         int = 16" in c:
    print("  ✓ base_config: node_in_features = 16")
    ok += 1
else:
    print("  ✗ base_config: node_in_features NOT 16")
    fail += 1

# 2. No duplicate region block
with open("train_unified.py") as f:
    t = f.read()
cnt = t.count("rid = batch.x[:, 5]")
if cnt == 1:
    print("  ✓ train_unified: 1 region block (no duplicate)")
    ok += 1
else:
    print(f"  ✗ train_unified: {cnt} region blocks (expected 1)")
    fail += 1

# 3. Config dataset path
if "cfg.all_regions_dataset_path" in t:
    print("  ✓ train_unified: uses config dataset path")
    ok += 1
else:
    print("  ✗ train_unified: still hardcoded dataset path")
    fail += 1

# 4. Pushforward time
if "cfg.dt / cfg.t_total" in t:
    print("  ✓ train_unified: pushforward uses cfg.dt/cfg.t_total")
    ok += 1
else:
    print("  ✗ train_unified: pushforward still hardcoded")
    fail += 1

# 5. Evaluate defaults
with open("evaluation/evaluate_unified.py") as f:
    e = f.read()
if 'sim.get("cx", 0.0)' in e:
    print("  ✓ evaluate: cx default = 0.0")
    ok += 1
else:
    print("  ✗ evaluate: cx default wrong")
    fail += 1

if 'sim.get("kappa", 60.0)' in e:
    print("  ✓ evaluate: kappa default = 60.0")
    ok += 1
else:
    print("  ✗ evaluate: kappa default wrong")
    fail += 1

if 'sim.get("Cp", 450.0)' in e:
    print("  ✓ evaluate: Cp default = 450.0")
    ok += 1
else:
    print("  ✗ evaluate: Cp default wrong")
    fail += 1

# 6. is_heater fix
if "rname in HEATER_REGIONS" in e:
    print("  ✓ evaluate: is_heater uses rname (string)")
    ok += 1
else:
    print("  ✗ evaluate: is_heater still uses rid (int)")
    fail += 1

print(f"\n  Result: {ok} passed, {fail} failed")
if fail == 0:
    print("  ✓ ALL FIXES VERIFIED")
else:
    print("  ✗ Some fixes failed — check above")
PYEOF

echo ""
echo "============================================================"
echo "  DONE — Next steps:"
echo "    1. sbatch run_sanity_test.sh"
echo "    2. sbatch run_alvis_unified.sh"
echo "    3. sbatch run_eval_gnn.sh"
echo "============================================================"
