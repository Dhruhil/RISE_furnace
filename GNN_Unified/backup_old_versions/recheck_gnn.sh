#!/bin/bash
# ============================================================
# GNN_Unified — COMPLETE RECHECK (post-fix)
# Run: cd /mimer/NOBACKUP/groups/revar/GNN_Unified
#      bash recheck_gnn.sh
# ============================================================
set -u
cd /mimer/NOBACKUP/groups/revar/GNN_Unified 2>/dev/null || { echo "ERROR: directory not found"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         GNN_Unified — COMPLETE RECHECK                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Directory: $(pwd)"
echo "  Date:      $(date)"
echo ""

OK=0; WARN=0; FAIL=0
pass() { OK=$((OK+1));   echo "  ✅ $1"; }
warn() { WARN=$((WARN+1)); echo "  ⚠️  $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ $1"; }

# ══════════════════════════════════════════════════════════════
echo "━━━ CONFIG ━━━"
# ══════════════════════════════════════════════════════════════

# 1. node_in_features
VAL=$(grep "node_in_features:" configs/base_config.py | head -1 | grep -oP '\d+' | tail -1)
[ "$VAL" = "16" ] && pass "node_in_features = 16" || fail "node_in_features = $VAL (need 16)"

# 2. edge_in_features
VAL=$(grep "edge_in_features:" configs/base_config.py | head -1 | grep -oP '\d+' | tail -1)
[ "$VAL" = "5" ] && pass "edge_in_features = 5" || fail "edge_in_features = $VAL (need 5)"

# 3. dataset path resolves
python3 -c "
import sys,os; sys.path.insert(0,'.')
try:
    from configs.base_config import CONFIG
    p = CONFIG.all_regions_dataset_path
    print(p)
    if os.path.exists(p): print('EXISTS')
    else: print('MISSING')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null > /tmp/_ds_check.txt
DS_PATH=$(head -1 /tmp/_ds_check.txt)
DS_STATUS=$(tail -1 /tmp/_ds_check.txt)
if [ "$DS_STATUS" = "EXISTS" ]; then
    SIZE=$(ls -lh "$DS_PATH" 2>/dev/null | awk '{print $5}')
    pass "Dataset exists: $DS_PATH ($SIZE)"
elif [ "$DS_STATUS" = "MISSING" ]; then
    fail "Dataset NOT found: $DS_PATH"
else
    # Fallback: check directly
    if [ -f "dataset_all_regions_66cases.h5" ]; then
        SIZE=$(ls -lh dataset_all_regions_66cases.h5 | awk '{print $5}')
        pass "Dataset in CWD: dataset_all_regions_66cases.h5 ($SIZE)"
    else
        fail "Cannot locate dataset"
    fi
fi

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ DATASET (data/dataset_unified.py) ━━━"
# ══════════════════════════════════════════════════════════════

# 4. REGION_IDS mapping
OUTER=$(grep "outer_box" data/dataset_unified.py | grep -oP '\d+' | head -1)
[ "$OUTER" = "11" ] && pass "REGION_IDS: outer_box = 11" || fail "outer_box = $OUTER (need 11)"

STEEL=$(grep "steel_cylinder" data/dataset_unified.py | grep -oP '\d+' | head -1)
[ "$STEEL" = "0" ] && pass "REGION_IDS: steel_cylinder = 0" || fail "steel = $STEEL (need 0)"

# 5. What is y (target)?
echo ""
echo "  Target (y) definition:"
grep "dT1 = " data/dataset_unified.py | head -1 | sed 's/^/    /'
echo ""
if grep -q "T_tp1 - self.T_mean" data/dataset_unified.py; then
    pass "Target y = normalised T_next: (T_tp1 - T_mean) / T_std"
    TARGET="T_next_norm"
elif grep -q "T_tp1 - T_t" data/dataset_unified.py; then
    warn "Target y = normalised delta_T (old version)"
    TARGET="dT_norm"
else
    warn "Cannot determine target format"
    TARGET="unknown"
fi

# 6. Node features count
FEAT_COUNT=$(grep -c "np.full(total," data/dataset_unified.py 2>/dev/null)
COORD_FEATS=3  # x, y, z
TOTAL_GUESS=$((FEAT_COUNT + COORD_FEATS + 2))  # coords + T_norm + is_heater_feat
echo "  Node features: ~$FEAT_COUNT scalar features + 3 coords + T_norm + is_heater"
grep "# \[" data/dataset_unified.py 2>/dev/null | head -20 | sed 's/^/    /'

# 7. Graph: edge types
if grep -q "edge_type\|0.0\]" data/dataset_unified.py; then
    pass "Edges have 5th feature (edge_type: 0=intra, 1=inter)"
fi

# 8. Boundary edges use VALID_ADJACENCY
if grep -q "VALID_ADJACENCY" data/dataset_unified.py; then
    pass "Inter-region edges filtered by VALID_ADJACENCY"
else
    warn "No VALID_ADJACENCY filter — all region pairs get boundary edges"
fi

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ TRAINING (train_unified.py) ━━━"
# ══════════════════════════════════════════════════════════════

# 9. Duplicate region block
COUNT=$(grep -c "rid = batch.x\[:, 5\]" train_unified.py)
[ "$COUNT" = "1" ] && pass "No duplicate region block" || fail "$COUNT region blocks (need 1)"

# 10. Dataset path
if grep -q "cfg.all_regions_dataset_path" train_unified.py; then
    pass "Uses cfg.all_regions_dataset_path"
else
    fail "Hardcoded dataset path"
fi

# 11. Pushforward time
if grep -q "cfg.dt / cfg.t_total" train_unified.py; then
    pass "Pushforward uses cfg.dt / cfg.t_total"
elif grep -q "10.0 / 4000.0" train_unified.py; then
    fail "Pushforward hardcoded 10.0/4000.0"
fi

# 12. Pushforward denorm (should match target type)
echo ""
echo "  Pushforward denorm:"
grep "T_pred1 = " train_unified.py | head -1 | sed 's/^/    /'
if [ "$TARGET" = "T_next_norm" ]; then
    if grep -q "pred1.squeeze(-1).detach() \* T_std_ds + T_mean" train_unified.py; then
        pass "Pushforward: pred * T_std + T_mean (correct for normalised T_next)"
    else
        fail "Pushforward denorm doesn't match target type"
    fi
fi

# 13. Evaluate denorm
echo ""
echo "  Evaluate denorm:"
grep "T_pred = pred" train_unified.py | head -1 | sed 's/^/    /'
if [ "$TARGET" = "T_next_norm" ]; then
    if grep -q "pred.squeeze(-1) \* T_std_ds + T_mean" train_unified.py; then
        pass "evaluate(): pred * T_std + T_mean (correct for normalised T_next)"
    else
        fail "evaluate() denorm doesn't match target type"
    fi
fi

# 14. CRITICAL: Physics loss — what gets passed?
echo ""
echo "  Physics loss calls:"
grep "physics_loss_unified(" train_unified.py | grep -v "def \|#" | sed 's/^/    /'
echo ""

if grep -q "physics_loss_unified(pred1, batch, T_std_ds, T_mean)" train_unified.py; then
    pass "PHYSICS LOSS (train): receives T_std, T_mean (correct)"
elif grep -q "physics_loss_unified(pred1, batch, dT_std, dT_mean)" train_unified.py; then
    fail "PHYSICS LOSS (train): receives dT_std, dT_mean (WRONG — need T_std, T_mean)"
fi

if grep -q "physics_loss_unified(pred, batch, T_std_ds, T_mean)" train_unified.py; then
    pass "PHYSICS LOSS (eval): receives T_std, T_mean (correct)"
elif grep -q "physics_loss_unified(pred, batch, dT_std, dT_mean)" train_unified.py; then
    fail "PHYSICS LOSS (eval): receives dT_std, dT_mean (WRONG — need T_std, T_mean)"
fi

# 15. Region weights
if grep -q "rids == 0.*10.0" train_unified.py; then
    pass "Region weights: steel=10x"
fi
if grep -q "rids == 11.*0.1" train_unified.py; then
    pass "Region weights: outer_box=0.1x"
fi

# 16. Heater masking in training
if grep -q "batch.is_heater.unsqueeze(-1).bool()" train_unified.py; then
    pass "Heaters masked in training loss"
fi

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ MODEL (models/meshgraphnet.py) ━━━"
# ══════════════════════════════════════════════════════════════

# 17. Aggregation direction
if grep -q "scatter_add_(0, dst" models/meshgraphnet.py; then
    pass "Fallback MGN: aggregation on dst (correct)"
elif grep -q "scatter_add_(0, src" models/meshgraphnet.py; then
    fail "Fallback MGN: aggregation on src (WRONG — should be dst)"
fi

# 18. PhysicsNeMo import
if grep -q "PHYSICSNEMO_AVAILABLE" models/meshgraphnet.py; then
    pass "PhysicsNeMo fallback logic present"
fi

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ EVALUATION (evaluation/evaluate_unified.py) ━━━"
# ══════════════════════════════════════════════════════════════

if [ -f "evaluation/evaluate_unified.py" ]; then
    # 19. is_heater type
    if grep -q "rname in HEATER_REGIONS" evaluation/evaluate_unified.py; then
        pass "is_heater: string comparison (correct)"
    elif grep -q "rid in HEATER_REGIONS" evaluation/evaluate_unified.py; then
        fail "is_heater: int vs string bug (always False)"
    fi

    # 20. Default values
    CX=$(grep 'sim.get("cx"' evaluation/evaluate_unified.py | head -1 | grep -oP '[\d.]+' | tail -1)
    KA=$(grep 'sim.get("kappa"' evaluation/evaluate_unified.py | head -1 | grep -oP '[\d.]+' | tail -1)
    CP=$(grep 'sim.get("Cp"' evaluation/evaluate_unified.py | head -1 | grep -oP '[\d.]+' | tail -1)
    [ "$CX" = "0.0" ] && pass "Default cx = 0.0" || fail "Default cx = $CX (need 0.0)"
    [ "$KA" = "60.0" ] && pass "Default kappa = 60.0" || fail "Default kappa = $KA (need 60.0)"
    [ "$CP" = "450.0" ] && pass "Default Cp = 450.0" || fail "Default Cp = $CP (need 450.0)"

    # 21. Rollout denorm
    echo ""
    echo "  Rollout denorm:"
    grep "T_next = pred" evaluation/evaluate_unified.py | head -1 | sed 's/^/    /'
    if grep -q "pred.squeeze(-1).cpu().numpy() \* T_std + T_mean" evaluation/evaluate_unified.py; then
        if [ "$TARGET" = "T_next_norm" ]; then
            pass "Rollout: pred * T_std + T_mean (correct for normalised T_next)"
        fi
    fi
else
    warn "evaluation/evaluate_unified.py not found"
fi

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ CHECKPOINT ━━━"
# ══════════════════════════════════════════════════════════════

CKPT="outputs/checkpoints_unified/best_model.pt"
if [ -f "$CKPT" ]; then
    SIZE=$(ls -lh "$CKPT" | awk '{print $5}')
    pass "Checkpoint: $CKPT ($SIZE)"
else
    warn "No checkpoint at $CKPT"
fi

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ HOUSEKEEPING ━━━"
# ══════════════════════════════════════════════════════════════

BAKS=$(find . -maxdepth 2 -name "*.bak*" -not -path "./backup_old_versions/*" 2>/dev/null | wc -l)
if [ "$BAKS" -eq 0 ]; then
    pass "No stale .bak files"
else
    warn "$BAKS .bak files outside backup_old_versions/:"
    find . -maxdepth 2 -name "*.bak*" -not -path "./backup_old_versions/*" 2>/dev/null | sed 's/^/    /'
fi

STRAY=$(test -e "0.5" && echo "yes" || echo "no")
[ "$STRAY" = "no" ] && pass "No stray '0.5'" || warn "Stray '0.5' exists"

# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     FINAL SUMMARY                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  ✅ PASS:  %-3d                                             ║\n" $OK
printf "║  ⚠️  WARN:  %-3d                                             ║\n" $WARN
printf "║  ❌ FAIL:  %-3d                                             ║\n" $FAIL
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "  🔴 $FAIL issue(s) need fixing."
elif [ $WARN -gt 0 ]; then
    echo "  🟡 All critical checks pass. $WARN minor warning(s)."
else
    echo "  🟢 Everything looks good!"
fi
echo ""
