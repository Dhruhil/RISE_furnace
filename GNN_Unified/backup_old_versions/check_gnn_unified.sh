#!/bin/bash
# ============================================================
# GNN_Unified — FULL DIAGNOSTIC CHECK
# Paste this entire script into your Alvis terminal:
#   cd /mimer/NOBACKUP/groups/revar/GNN_Unified
#   bash check_gnn_unified.sh
#
# Or paste the commands one section at a time.
# ============================================================

set -u
cd /mimer/NOBACKUP/groups/revar/GNN_Unified 2>/dev/null || { echo "ERROR: GNN_Unified directory not found"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         GNN_Unified — FULL DIAGNOSTIC CHECK                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Directory: $(pwd)"
echo "  Date:      $(date)"
echo ""

OK=0
WARN=0
FAIL=0

pass() { OK=$((OK+1));   echo "  ✅ PASS: $1"; }
warn() { WARN=$((WARN+1)); echo "  ⚠️  WARN: $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ FAIL: $1"; }

# ──────────────────────────────────────────────────────────────
echo "═══ 1. base_config.py: node_in_features ═══"
# ──────────────────────────────────────────────────────────────
VAL=$(grep "node_in_features:" configs/base_config.py 2>/dev/null | head -1 | grep -oP '\d+' | tail -1)
if [ "$VAL" = "16" ]; then
    pass "node_in_features = 16"
elif [ "$VAL" = "11" ]; then
    fail "node_in_features = 11 (should be 16 — fix_gnn_unified_v2.sh NOT run)"
else
    warn "node_in_features = '$VAL' (unexpected value)"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 2. train_unified.py: duplicated region block ═══"
# ──────────────────────────────────────────────────────────────
COUNT=$(grep -c "rid = batch.x\[:, 5\]" train_unified.py 2>/dev/null)
if [ "$COUNT" = "1" ]; then
    pass "1 region block (no duplicate)"
elif [ "$COUNT" = "2" ]; then
    fail "2 region blocks — duplicate still present"
else
    warn "$COUNT occurrences of rid = batch.x[:, 5] (expected 1)"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 3. train_unified.py: dataset path ═══"
# ──────────────────────────────────────────────────────────────
if grep -q 'h5 = cfg.all_regions_dataset_path' train_unified.py 2>/dev/null; then
    pass "Uses cfg.all_regions_dataset_path"
elif grep -q 'h5 = "dataset_all_regions.h5"' train_unified.py 2>/dev/null; then
    fail "Hardcoded h5 = \"dataset_all_regions.h5\" (won't find 66-case dataset)"
else
    warn "Could not determine dataset path pattern"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 4. train_unified.py: pushforward time ═══"
# ──────────────────────────────────────────────────────────────
if grep -q 'cfg.dt / cfg.t_total' train_unified.py 2>/dev/null; then
    pass "Pushforward uses cfg.dt / cfg.t_total"
elif grep -q '10.0 / 4000.0' train_unified.py 2>/dev/null; then
    fail "Pushforward hardcoded 10.0/4000.0 (should use cfg.dt/cfg.t_total)"
else
    warn "Could not find pushforward time pattern"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 5. CRITICAL: physics_loss_unified() denormalization ═══"
# ──────────────────────────────────────────────────────────────
echo "  Checking what physics_loss_unified uses to denorm predictions..."
echo "  --- Function signature ---"
grep "def physics_loss_unified" train_unified.py 2>/dev/null
echo "  --- Denormalization line ---"
grep "T_next = pred" train_unified.py 2>/dev/null | head -3
echo ""
if grep -q "T_next = pred.squeeze(-1) \* dT_std + dT_mean" train_unified.py 2>/dev/null; then
    fail "physics_loss uses dT_std/dT_mean to denorm T_next (WRONG — should use T_std/T_mean)"
    echo "       dT_std ≈ 0.01-1K, dT_mean ≈ 0.0001K"
    echo "       T_std ≈ 300K, T_mean ≈ 600K"
    echo "       → Physics loss computed on ~0K temps instead of ~600-1100K"
elif grep -q "T_next = pred.squeeze(-1) \* T_std + T_mean" train_unified.py 2>/dev/null; then
    pass "physics_loss uses T_std/T_mean to denorm (correct)"
else
    warn "Could not determine denormalization pattern — check manually"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 6. physics_loss_unified(): hardcoded dt ═══"
# ──────────────────────────────────────────────────────────────
if grep -A2 "def physics_loss_unified" train_unified.py 2>/dev/null | grep -q "dt = 10.0"; then
    warn "physics_loss has hardcoded dt=10.0 (should use cfg.dt)"
else
    pass "dt not hardcoded in physics_loss (or function not found)"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 7. evaluate_unified.py: is_heater type bug ═══"
# ──────────────────────────────────────────────────────────────
if grep -q "rname in HEATER_REGIONS" evaluation/evaluate_unified.py 2>/dev/null; then
    pass "is_heater uses rname (string comparison) ✓"
elif grep -q "rid in HEATER_REGIONS" evaluation/evaluate_unified.py 2>/dev/null; then
    fail "is_heater compares int rid to string set — always False!"
else
    warn "Could not find is_heater pattern in evaluate_unified.py"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 8. evaluate_unified.py: default material values ═══"
# ──────────────────────────────────────────────────────────────
echo "  Current defaults:"
grep 'sim.get("cx"' evaluation/evaluate_unified.py 2>/dev/null | head -1
grep 'sim.get("kappa"' evaluation/evaluate_unified.py 2>/dev/null | head -1
grep 'sim.get("Cp"' evaluation/evaluate_unified.py 2>/dev/null | head -1

CX_OK=$(grep -c 'sim.get("cx", 0.0)' evaluation/evaluate_unified.py 2>/dev/null)
KA_OK=$(grep -c 'sim.get("kappa", 60.0)' evaluation/evaluate_unified.py 2>/dev/null)
CP_OK=$(grep -c 'sim.get("Cp", 450.0)' evaluation/evaluate_unified.py 2>/dev/null)

if [ "$CX_OK" -ge 1 ] && [ "$KA_OK" -ge 1 ] && [ "$CP_OK" -ge 1 ]; then
    pass "Defaults: cx=0.0, kappa=60.0, Cp=450.0 (correct)"
else
    fail "Wrong defaults — cx should be 0.0, kappa 60.0, Cp 450.0"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 9. REGION_IDS mapping consistency ═══"
# ──────────────────────────────────────────────────────────────
echo "  Live dataset_unified.py:"
grep -A15 "^REGION_IDS" data/dataset_unified.py 2>/dev/null | head -15
echo ""
OUTER_ID=$(python3 -c "
import sys; sys.path.insert(0,'.')
from data.dataset_unified import REGION_IDS
print(REGION_IDS.get('outer_box', 'NOT FOUND'))
" 2>/dev/null)
echo "  outer_box ID = $OUTER_ID"
if [ "$OUTER_ID" = "11" ]; then
    pass "outer_box = 11 (matches region weight code: rids == 11 → 0.1x)"
elif [ "$OUTER_ID" = "2" ]; then
    fail "outer_box = 2 (OLD mapping — region weights will be WRONG)"
else
    warn "outer_box ID = '$OUTER_ID' (unexpected)"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 10. Dataset file exists ═══"
# ──────────────────────────────────────────────────────────────
DS_PATH=$(grep "all_regions_dataset_path" configs/base_config.py 2>/dev/null | grep -oP '"[^"]+"' | tr -d '"')
echo "  Config points to: $DS_PATH"
if [ -f "$DS_PATH" ]; then
    SIZE=$(ls -lh "$DS_PATH" 2>/dev/null | awk '{print $5}')
    pass "Dataset exists ($SIZE)"
else
    fail "Dataset file NOT found at $DS_PATH"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 11. Stale backup files ═══"
# ──────────────────────────────────────────────────────────────
BAKS=$(find . -name "*.bak*" -not -path "./backup_old_versions/*" 2>/dev/null | wc -l)
STRAY_05=$(test -e "0.5" && echo "yes" || echo "no")
if [ "$BAKS" -gt 0 ]; then
    warn "$BAKS .bak files outside backup_old_versions/"
    find . -name "*.bak*" -not -path "./backup_old_versions/*" 2>/dev/null | head -10
else
    pass "No stale .bak files"
fi
if [ "$STRAY_05" = "yes" ]; then
    warn "Stray '0.5' file/directory exists"
else
    pass "No stray '0.5'"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 12. Model checkpoint exists ═══"
# ──────────────────────────────────────────────────────────────
CKPT="outputs/checkpoints_unified/best_model.pt"
if [ -f "$CKPT" ]; then
    SIZE=$(ls -lh "$CKPT" 2>/dev/null | awk '{print $5}')
    pass "Checkpoint exists: $CKPT ($SIZE)"
    echo "  Checking what REGION_IDS it was trained with..."
    python3 -c "
import torch
ckpt = torch.load('$CKPT', map_location='cpu', weights_only=False)
print(f'  Epoch: {ckpt.get(\"epoch\", \"?\")}'  )
print(f'  Metrics: {ckpt.get(\"metrics\", {})}')
m = ckpt.get('model_cfg', {})
print(f'  Model cfg: node_in={m.get(\"node_in_features\",\"?\")}, hidden={m.get(\"hidden_features\",\"?\")}, layers={m.get(\"n_message_passing_layers\",\"?\")}')
" 2>/dev/null || warn "Could not read checkpoint (may need GPU or different torch version)"
else
    warn "No checkpoint at $CKPT"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 13. Edge features dimension ═══"
# ──────────────────────────────────────────────────────────────
EDGE_DIM=$(grep "edge_in_features" configs/base_config.py 2>/dev/null | head -1 | grep -oP '\d+' | tail -1)
echo "  Config edge_in_features = $EDGE_DIM"
# Check if dataset builds 5-dim edge attrs (dx, dy, dz, dist, edge_type)
EDGE_COLS=$(grep -c "0.0\]" data/dataset_unified.py 2>/dev/null)  # edge_type column
if [ "$EDGE_DIM" = "5" ]; then
    pass "edge_in_features = 5 (dx, dy, dz, dist, edge_type)"
elif [ "$EDGE_DIM" = "4" ]; then
    fail "edge_in_features = 4 but dataset builds 5-dim edges (includes edge_type)"
else
    warn "edge_in_features = $EDGE_DIM (expected 5)"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ 14. train_unified.py: target is T_next or delta_T? ═══"
# ──────────────────────────────────────────────────────────────
echo "  What does the model predict?"
grep "T_pred = pred" train_unified.py 2>/dev/null | head -2
echo "  What is batch.y?"
grep "y=torch" data/dataset_unified.py 2>/dev/null | head -1
echo ""
# Check if y is dT (delta) or T_next
if grep -q "dT1 = ((T_tp1 - T_t" data/dataset_unified.py 2>/dev/null; then
    echo "  dataset: y = normalised delta_T (dT)"
    if grep -q "pred.squeeze(-1) \* T_std_ds + T_mean" train_unified.py 2>/dev/null; then
        fail "MISMATCH: dataset returns delta_T but training denorms as T_next!"
    elif grep -q "pred.squeeze(-1) \* dT_std + dT_mean" train_unified.py 2>/dev/null; then
        warn "dataset returns delta_T, training denorms with dT_std — check if this is intentional"
    fi
elif grep -q "T_tp1_norm" data/dataset_unified.py 2>/dev/null; then
    echo "  dataset: y = normalised T_next"
fi

# ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     SUMMARY                                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  ✅ PASS:  %-3d                                             ║\n" $OK
printf "║  ⚠️  WARN:  %-3d                                             ║\n" $WARN
printf "║  ❌ FAIL:  %-3d                                             ║\n" $FAIL
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "  🔴 $FAIL critical issues found — run fix_gnn_unified_v2.sh"
    echo "     and fix the physics_loss denormalization manually."
fi
if [ $WARN -gt 0 ]; then
    echo "  🟡 $WARN warnings — review above."
fi
if [ $FAIL -eq 0 ] && [ $WARN -eq 0 ]; then
    echo "  🟢 All checks passed!"
fi
echo ""
