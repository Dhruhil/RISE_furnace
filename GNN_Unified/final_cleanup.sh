#!/bin/bash
# ============================================================
# GNN_Unified — final cleanup: copy dataset + fix remaining issues
# ============================================================
set -u
cd /mimer/NOBACKUP/groups/revar/GNN_Unified || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
BK=backup_final_${STAMP}
mkdir -p "$BK"

echo "============================================================"
echo "  Final cleanup — backup dir: $BK"
echo "============================================================"

# ── BACKUP ──────────────────────────────────────────────────
for f in configs/base_config.py \
         data/dataset_unified.py \
         train_unified.py \
         evaluation/evaluate_unified.py; do
    cp "$f" "$BK/$(basename "$f").bak"
done
echo "[1/8] Backups created."

# ── STEP 2: Copy dataset into GNN_Unified folder ────────────
SRC=/mimer/NOBACKUP/groups/revar/Dataset_k8/dataset_all_regions_clean.h5
DST=/mimer/NOBACKUP/groups/revar/GNN_Unified/dataset_all_regions_clean.h5

if [ -f "$DST" ]; then
    echo "[2/8] Dataset already exists at $DST — skipping copy."
else
    echo "[2/8] Copying dataset (this may take 1–2 minutes)..."
    cp "$SRC" "$DST"
    echo "       Done. Size: $(ls -lh $DST | awk '{print $5}')"
fi

# Update config path to point at the local copy
sed -i 's|all_regions_dataset_path: str = "/mimer/NOBACKUP/groups/revar/Dataset_k8/dataset_all_regions_clean.h5"|all_regions_dataset_path: str = f"{_BASE}/GNN_Unified/dataset_all_regions_clean.h5"|' configs/base_config.py
echo "       Config path updated to local copy."

# ── STEP 3: Fix hardcoded dt = 10.0 in physics loss ─────────
sed -i 's|    dt = 10\.0$|    dt = 10.0  # TODO: pass cfg.dt from caller (currently same value)|' train_unified.py
# Keeping value but marking intent — changing it would break function signature
echo "[3/8] dt hardcode annotated."

# ── STEP 4: Remove dead-code weights from base_config ──────
# These are never read; leave them but add a comment so we don't confuse ourselves
python3 << 'PYEOF'
import pathlib
p = pathlib.Path("configs/base_config.py")
src = p.read_text()
old = "    w_conduction:   float = 0.3\n    w_convection:   float = 0.5\n    w_radiation:    float = 0.3"
new = ("    # NOTE: the loss weights below are currently unused.\n"
       "    # Actual weights are hardcoded in physics_loss_unified(): 0.5/0.3/0.15/0.05.\n"
       "    # Kept here for potential future use.\n"
       "    w_conduction:   float = 0.3\n"
       "    w_convection:   float = 0.5\n"
       "    w_radiation:    float = 0.3")
if old in src and new not in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("  Added NOTE above dead-code weights.")
else:
    print("  Skipped (already annotated or pattern changed).")
PYEOF
echo "[4/8] Dead-code weights annotated."

# ── STEP 5: Rename misleading dT1/dT2/dT3 variables ────────
python3 << 'PYEOF'
import pathlib, re
p = pathlib.Path("data/dataset_unified.py")
src = p.read_text()
# These three assignments have names dT* but hold normalised T_next values.
# Rename to T_tp*_norm for clarity.
changes = [
    ("dT1 = ((T_tp1 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)",
     "T_tp1_norm = ((T_tp1 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)"),
    ("dT2 = ((T_tp2 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)",
     "T_tp2_norm = ((T_tp2 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)"),
    ("dT3 = ((T_tp3 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)",
     "T_tp3_norm = ((T_tp3 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)"),
]
done = 0
for old, new in changes:
    if old in src:
        src = src.replace(old, new)
        done += 1

# Find downstream uses and rename them too (only the exact variable name, not substrings)
for old, new in [("dT1", "T_tp1_norm"), ("dT2", "T_tp2_norm"), ("dT3", "T_tp3_norm")]:
    # Only replace where dTN is a standalone word — in y=dT1 or Data(y=dT1, ...)
    src = re.sub(rf"\b{old}\b(?!_)", new, src)

p.write_text(src)
print(f"  Renamed {done} target variables + downstream references.")
PYEOF
echo "[5/8] Target variable names corrected."

# ── STEP 6: Add NaN guard in training loop ────────────────
python3 << 'PYEOF'
import pathlib, re
p = pathlib.Path("train_unified.py")
src = p.read_text()

# Insert right after the first 'pred1 = model(batch)' line in train_one_epoch
if "if not torch.isfinite(pred1).all():" not in src:
    m = re.search(r"(\s+)(pred1 = model\(batch\)\n)", src)
    if m:
        indent = m.group(1)
        guard = (f"{indent}if not torch.isfinite(pred1).all():\n"
                 f"{indent}    print(f'  WARN: non-finite pred1 at batch (skipping)')\n"
                 f"{indent}    optimizer.zero_grad(); continue\n")
        src = src[:m.end()] + guard + src[m.end():]
        p.write_text(src)
        print("  NaN guard added to training loop.")
    else:
        print("  Could not locate pred1 = model(batch) — skipped.")
else:
    print("  NaN guard already present.")
PYEOF
echo "[6/8] NaN guard added."

# ── STEP 7: Stratified split by T_set ─────────────────────
python3 << 'PYEOF'
import pathlib, re
p = pathlib.Path("data/dataset_unified.py")
src = p.read_text()

if "# stratified by T_set" in src:
    print("  Stratified split already installed.")
else:
    old_block = re.compile(
        r"(        n_test = max\(1, int\(n_sims \* cfg\.test_fraction\)\)\n"
        r"        n_val = max\(1, int\(n_sims \* cfg\.val_fraction\)\)\n"
        r"        n_train = n_sims - n_val - n_test\n"
        r"        import random\n"
        r"        shuffled = list\(range\(n_sims\)\)\n"
        r"        random\.Random\(42\)\.shuffle\(shuffled\)\n"
        r"        split_map = \{\n"
        r"            \"train\": shuffled\[:n_train\],\n"
        r"            \"val\": shuffled\[n_train:n_train \+ n_val\],\n"
        r"            \"test\": shuffled\[n_train \+ n_val:\],\n"
        r"        \}\n"
        r"        self\.sim_indices = split_map\[split\]\n)")

    m = old_block.search(src)
    if not m:
        print("  Could not locate old split block — skipped.")
    else:
        new_block = '''        # stratified by T_set — deterministic, reproducible
        import random as _rand
        _by_tset = {}
        for _i, _s in enumerate(self._simulations):
            _by_tset.setdefault(float(_s["T_set"]), []).append(_i)

        _train, _val, _test = [], [], []
        _rng = _rand.Random(42)
        for _tset in sorted(_by_tset):
            _idxs = _by_tset[_tset][:]
            _rng.shuffle(_idxs)
            _n = len(_idxs)
            _n_test = max(1, int(round(_n * cfg.test_fraction))) if _n >= 3 else 0
            _n_val  = max(1, int(round(_n * cfg.val_fraction)))  if _n >= 2 else 0
            if _n - _n_test - _n_val < 1:
                _n_val = max(0, _n - _n_test - 1)
            _test.extend(_idxs[:_n_test])
            _val.extend(_idxs[_n_test:_n_test + _n_val])
            _train.extend(_idxs[_n_test + _n_val:])

        split_map = {"train": _train, "val": _val, "test": _test}
        print(f"  [stratified split] train={len(_train)} val={len(_val)} test={len(_test)}")
        for _tset in sorted(_by_tset):
            _tr = sum(1 for i in _train if float(self._simulations[i]["T_set"]) == _tset)
            _vl = sum(1 for i in _val   if float(self._simulations[i]["T_set"]) == _tset)
            _te = sum(1 for i in _test  if float(self._simulations[i]["T_set"]) == _tset)
            print(f"    T_set={_tset:.0f}K  train={_tr} val={_vl} test={_te}")
        self.sim_indices = split_map[split]
'''
        src = src[:m.start()] + new_block + src[m.end():]
        p.write_text(src)
        print("  Stratified split installed.")
PYEOF
echo "[7/8] Split logic → stratified by T_set."

# ── STEP 8: Fix cosmetic print in evaluate_unified.py ─────
sed -i 's|0-3200s | Phase 2: 3200-4000s|0-2760s | Phase 2: 2760-3460s|' evaluation/evaluate_unified.py
echo "[8/8] Eval phase labels updated."

# ── SYNTAX CHECK ────────────────────────────────────────────
echo ""
echo "=== SYNTAX CHECK ==="
for f in configs/base_config.py data/dataset_unified.py train_unified.py evaluation/evaluate_unified.py; do
    if python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        echo "  OK   $f"
    else
        echo "  FAIL $f"
        python3 -c "import ast; ast.parse(open('$f').read())"
    fi
done

# ── VERIFICATION ────────────────────────────────────────────
echo ""
echo "=== VERIFICATION ==="
echo ""
echo "--- Dataset path ---"
grep "all_regions_dataset_path" configs/base_config.py | sed 's/^/  /'
echo ""
echo "--- Local dataset file ---"
ls -lh dataset_all_regions_clean.h5 2>/dev/null | sed 's/^/  /' || echo "  MISSING"
echo ""
echo "--- Time caps ---"
grep -E "t_total:|train_time_end:|predict_time_end:" configs/base_config.py | sed 's/^/  /'
echo ""
echo "--- Stratified split marker ---"
grep -n "stratified by T_set" data/dataset_unified.py | sed 's/^/  /' || echo "  not found"
echo ""
echo "--- NaN guard ---"
grep -n "torch.isfinite(pred1)" train_unified.py | sed 's/^/  /' || echo "  not found"
echo ""
echo "--- Renamed targets ---"
grep -n "T_tp1_norm\|T_tp2_norm\|T_tp3_norm" data/dataset_unified.py | head -6 | sed 's/^/  /'
echo ""
echo "============================================================"
echo "  DONE. Next: sbatch run_sanity_test.sh"
echo "============================================================"
