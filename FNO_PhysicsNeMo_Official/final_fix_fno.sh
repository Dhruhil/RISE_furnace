#!/bin/bash
# ============================================================================
#  FINAL FIX: Clean up duplicate physics functions in FNO train.py
#  Run from: /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# ============================================================================
set -e

echo ""
echo "================================================================"
echo "  FINAL FIX: Clean FNO train.py"
echo "================================================================"
echo ""

python3 << 'PYEOF'
with open("train.py", "r") as f:
    content = f.read()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Remove the OLD get_physics_lambda (line ~82) — replace calls
#    with get_lambda_fno (the correct GNN-matching one at line 36)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Remove the old get_physics_lambda function entirely
old_lambda = '''def get_physics_lambda(epoch, n_epochs):'''
if old_lambda in content:
    # Find and remove the entire old function (it's a few lines)
    lines = content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if 'def get_physics_lambda(epoch, n_epochs):' in line:
            skip = True
            continue
        if skip:
            # Skip until we hit the next non-indented line or empty line after return
            if line.strip() == '' or (not line.startswith(' ') and not line.startswith('\t') and line.strip() != ''):
                if line.strip() == '':
                    continue  # skip blank line after function
                skip = False
                new_lines.append(line)
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)
    print("  ✓ Removed old get_physics_lambda()")

# Replace all calls to get_physics_lambda → get_lambda_fno
content = content.replace('get_physics_lambda(', 'get_lambda_fno(')
print("  ✓ Replaced get_physics_lambda → get_lambda_fno in training loop")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Remove the OLD fno_physics_loss (the one with wrong signature)
#    Keep the NEW one (correct one with dataset arg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# The old one starts with: def fno_physics_loss(dT_pred_norm, T_current, T_set_val,
# The new one starts with: def fno_physics_loss(pred_norm, y_norm, x, cfg, dataset):
# Remove the OLD one

lines = content.split('\n')
new_lines = []
skip_old_physics = False
removed_old = False

for i, line in enumerate(lines):
    # Detect the OLD fno_physics_loss (has dT_pred_norm parameter)
    if 'def fno_physics_loss(dT_pred_norm' in line or \
       ('def fno_physics_loss(' in line and 'T_current, T_set_val' in line):
        skip_old_physics = True
        removed_old = True
        continue
    
    if skip_old_physics:
        # Skip lines belonging to this function
        stripped = line.strip()
        if stripped == '':
            continue  # skip blank lines in/after function
        # If we hit a non-indented line that's a new definition, stop skipping
        if not line.startswith(' ') and not line.startswith('\t') and stripped != '':
            skip_old_physics = False
            new_lines.append(line)
        # Stay in skip mode for indented lines
        continue
    
    new_lines.append(line)

if removed_old:
    content = '\n'.join(new_lines)
    print("  ✓ Removed old fno_physics_loss(dT_pred_norm, ...) duplicate")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Fix the training loop to call the NEW fno_physics_loss correctly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Replace the old physics call block in the training loop
old_call = """            if lam > 1e-10:
                T_set_vals = x[:, 1, 0] * ds.Tset_std + ds.Tset_mean
                T_set_mean = T_set_vals.mean().item()
                L_conv, L_cond, L_rad = fno_physics_loss(
                    pred, T_cur_dev, T_set_mean, ds.dT_std, ds.dT_mean, dt=cfg.dt)
                loss_phys = 0.5 * L_conv + 0.3 * L_cond + 0.2 * L_rad"""

new_call = """            if lam > 1e-10:
                loss_phys, _ = fno_physics_loss(
                    pred, y, x, cfg, ds)"""

if old_call in content:
    content = content.replace(old_call, new_call)
    print("  ✓ Fixed training loop → calls new fno_physics_loss(pred, y, x, cfg, ds)")
else:
    print("  ⚠ Old call pattern not found exactly — checking alternatives...")
    # Try a more flexible match
    if "T_set_vals = x[:, 1, 0]" in content:
        # Replace line by line
        lines = content.split('\n')
        new_lines = []
        skip_block = False
        for line in lines:
            if "T_set_vals = x[:, 1, 0]" in line:
                skip_block = True
                continue
            if skip_block:
                if "loss_phys = 0.5 * L_conv" in line:
                    # Replace this entire block with new call
                    new_lines.append("                loss_phys, _ = fno_physics_loss(\n")
                    new_lines.append("                    pred, y, x, cfg, ds)\n")
                    skip_block = False
                    continue
                if "L_conv, L_cond, L_rad = fno_physics_loss" in line:
                    continue
                if line.strip().startswith("pred, T_cur_dev"):
                    continue
                continue
            new_lines.append(line)
        content = '\n'.join(new_lines)
        print("  ✓ Fixed training loop (flexible match)")

# Also remove T_cur_dev if it's no longer needed
# (the new fno_physics_loss doesn't need it — it extracts T from x)
content = content.replace("            T_cur_dev = T_cur.to(device)\n", "")
if "T_cur_dev" not in content.split("fno_physics_loss")[0].split("for x, y")[-1]:
    print("  ✓ Removed unused T_cur_dev")

# Remove "ds = train_loader.dataset" if not already present at the right spot
# (the loop uses ds already, which should be defined before the epoch loop)
if "ds = train_loader.dataset" not in content:
    # Add it before the epoch loop
    content = content.replace(
        "    for epoch in range(1, cfg.n_epochs + 1):",
        "    ds = train_loader.dataset\n\n    for epoch in range(1, cfg.n_epochs + 1):"
    )
    print("  ✓ Added ds = train_loader.dataset")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Write back
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with open("train.py", "w") as f:
    f.write(content)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Final verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with open("train.py", "r") as f:
    final = f.read()

# Count how many fno_physics_loss definitions exist
n_defs = final.count("def fno_physics_loss(")
n_lambda = final.count("def get_lambda_fno(")
n_old_lambda = final.count("def get_physics_lambda(")
n_old_call = final.count("T_set_vals = x[:, 1, 0]")
has_correct_call = "fno_physics_loss(\n                    pred, y, x, cfg, ds)" in final or \
                   "fno_physics_loss(pred, y, x, cfg, ds)" in final or \
                   "fno_physics_loss(\n                    pred, y, x, cfg," in final

print()
print("  ━━━ Final state ━━━")
print(f"    fno_physics_loss definitions: {n_defs} {'✓' if n_defs == 1 else '✗ (should be 1)'}")
print(f"    get_lambda_fno definitions:   {n_lambda} {'✓' if n_lambda == 1 else '✗ (should be 1)'}")
print(f"    old get_physics_lambda:       {n_old_lambda} {'✓ removed' if n_old_lambda == 0 else '✗ still present'}")
print(f"    old T_set_vals call:          {n_old_call} {'✓ removed' if n_old_call == 0 else '✗ still present'}")
print(f"    correct physics call:         {'✓' if has_correct_call else '✗'}")
print(f"    lam = get_lambda_fno:         {'✓' if 'lam = get_lambda_fno' in final else '✗'}")

PYEOF

echo ""
echo "  Verify training loop:"
echo "  ────────────────────"
grep -n "lam = get_lambda\|fno_physics_loss(\|loss_data\|loss = loss_data\|loss_phys" train.py | head -15
echo ""

echo "================================================================"
echo "  Now run:  sbatch run_alvis_fno.sh"
echo "================================================================"
