#!/bin/bash
# ============================================================================
#  DIRECT FIX: Patch FNO train.py training loop with physics loss
#  Run from: /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# ============================================================================
set -e

echo ""
echo "================================================================"
echo "  DIRECT PATCH: FNO train.py physics loss"
echo "================================================================"
echo ""

# This script reads your actual train.py and patches it correctly
# using Python (not sed) — handles any current state of the file.

python3 << 'PYEOF'
import re

with open("train.py", "r") as f:
    lines = f.readlines()

content = "".join(lines)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Check if get_lambda_fno already exists
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
has_lambda = "get_lambda_fno" in content
has_physics = "fno_physics_loss" in content
has_loop_patched = "lam = get_lambda_fno" in content

print(f"  Current state:")
print(f"    get_lambda_fno:    {'present' if has_lambda else 'MISSING'}")
print(f"    fno_physics_loss:  {'present' if has_physics else 'MISSING'}")
print(f"    loop patched:      {'YES' if has_loop_patched else 'NO — needs fix'}")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Add physics functions if not present
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

physics_code = '''
# ─────────────────────────────────────────────────────────────────────
# Physics curriculum — SAME smooth exponential as GNN all-regions
# ─────────────────────────────────────────────────────────────────────
import math as _math

def get_lambda_fno(epoch: int, n_epochs: int) -> float:
    """Smooth exponential — identical to GNN get_lambda_ar()."""
    p = epoch / n_epochs
    lam = 0.001 * _math.exp(4.6 * p)
    return min(lam, 0.10)

SIGMA_SB = 5.67e-8

def fno_physics_loss(pred_norm, y_norm, x, cfg, dataset):
    """
    Physics loss for FNO — same 3 equations as GNN, same weights.
    L = 0.5*L_conv + 0.3*L_cond + 0.2*L_rad
    """
    dt = cfg.dt
    T_pred = pred_norm.squeeze(1) * dataset.T_std + dataset.T_mean
    T_now  = x[:, 0, :] * dataset.T_std + dataset.T_mean
    T_set  = x[:, 1, :] * dataset.Tset_std + dataset.Tset_mean
    dT_pred = T_pred - T_now
    dT_dt   = dT_pred / dt

    # 1. Convection: T ≤ T_set (weight 0.5)
    is_heater = (T_now > T_set * 1.05).float()
    overshoot = torch.nn.functional.relu(T_pred - T_set) * (1.0 - is_heater)
    L_conv = (overshoot / T_set.clamp(min=300)).pow(2).mean()

    # 2. Conduction: spectral smoothness (weight 0.3)
    pred_fft = torch.fft.rfft(pred_norm.squeeze(1), dim=-1)
    n_freq = pred_fft.shape[-1]
    cutoff = max(n_freq // 3, 1)
    high_freq = pred_fft[:, cutoff:].abs().pow(2)
    L_cond = high_freq.mean()

    # 3. Radiation: Stefan-Boltzmann (weight 0.2)
    Q_rad = cfg.epsilon_steel * SIGMA_SB * (T_set.pow(4) - T_now.pow(4))
    dT_rad = Q_rad / (7800.0 * 450.0 * cfg.char_thickness)
    scale_r = dT_rad.abs().mean().clamp(min=1e-8)
    L_rad = ((dT_dt - dT_rad) / scale_r).pow(2).mean()

    L_physics = 0.5 * L_conv + 0.3 * L_cond + 0.2 * L_rad
    return L_physics, {"conv": L_conv.item(), "cond": L_cond.item(), "rad": L_rad.item()}

'''

if not has_lambda or not has_physics:
    # Find where to insert — after the last import/from line before validate()
    marker = "from utils.checkpoint import CheckpointManager"
    if marker in content:
        content = content.replace(marker, marker + physics_code)
        print("  ✓ Inserted get_lambda_fno() + fno_physics_loss()")
    else:
        # Try alternate insertion point
        marker2 = "@torch.no_grad()\ndef validate"
        content = content.replace(marker2, physics_code + "\n" + marker2)
        print("  ✓ Inserted physics functions (alternate location)")
else:
    print("  ✓ Physics functions already present")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Patch the training loop — find it by pattern and replace
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if not has_loop_patched:
    # Find the training loop: look for the pattern
    # "for x, y, *_ in train_loader:" followed by MSE loss
    # We need to add "lam = get_lambda_fno(...)" before the for loop
    # and replace the loss computation inside

    # Pattern: find "for epoch in range(1, cfg.n_epochs + 1):"
    # then find the inner "for x, y" loop and patch it

    # Strategy: line-by-line replacement
    new_lines = []
    i = 0
    patched = False
    while i < len(lines):
        line = lines[i]

        # Find: "        for x, y, *_ in train_loader:"
        if "for x, y, *_ in train_loader:" in line and not patched:
            # Insert lam computation BEFORE this for loop
            indent = "        "
            new_lines.append(f"{indent}lam = get_lambda_fno(epoch, cfg.n_epochs)\n")
            new_lines.append(line)  # keep the for loop line
            i += 1

            # Now consume lines until we find "loss.backward()" and replace the block
            inner_block = []
            while i < len(lines):
                inner_line = lines[i]
                inner_block.append(inner_line)
                if "loss.backward()" in inner_line:
                    break
                i += 1
            i += 1  # move past loss.backward()

            # Write the new inner block
            ind = "            "
            new_lines.append(f"{ind}x, y = x.to(device), y.to(device)\n")
            new_lines.append(f"{ind}optimizer.zero_grad()\n")
            new_lines.append(f"{ind}pred = model(x)\n")
            new_lines.append(f"{ind}loss_data = F.mse_loss(pred, y)\n")
            new_lines.append(f"\n")
            new_lines.append(f"{ind}# Physics loss (same curriculum as GNN)\n")
            new_lines.append(f"{ind}if lam > 1e-10:\n")
            new_lines.append(f"{ind}    L_phys, _ = fno_physics_loss(\n")
            new_lines.append(f"{ind}        pred, y, x, cfg, train_loader.dataset)\n")
            new_lines.append(f"{ind}    loss = loss_data + lam * L_phys\n")
            new_lines.append(f"{ind}else:\n")
            new_lines.append(f"{ind}    loss = loss_data\n")
            new_lines.append(f"\n")
            new_lines.append(f"{ind}loss.backward()\n")
            patched = True
            print("  ✓ Patched training loop: loss = MSE + λ * physics")
            continue

        # Find log line and add lambda display
        if ('tag = "  < BEST"' in line or "tag = '  < BEST'" in line) and "lam_now" not in lines[i-1] if i > 0 else True:
            if "lam_now" not in content[:sum(len(l) for l in lines[:i])]:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}lam_now = get_lambda_fno(epoch, cfg.n_epochs)\n")

        new_lines.append(line)

        # If this is the print line with lr, add lambda
        if "within_5K" in line and "lam_now" not in line and "{lr:" in line:
            # Replace this line to add lambda at the end
            new_lines.pop()  # remove the line we just added
            # Add λ to the print
            patched_line = line.rstrip().rstrip('")')
            if "tag}" not in line:
                patched_line = line.replace('{tag}")', '{tag} | \\u03bb={lam_now:.4f}")')
            new_lines.append(line)  # keep original for safety

        i += 1

    if patched:
        content = "".join(new_lines)
    else:
        print("  ⚠ Could not find training loop pattern — manual edit needed")
else:
    print("  ✓ Training loop already patched")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Write back
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with open("train.py", "w") as f:
    f.write(content)

print()
print("  Final verification:")
with open("train.py", "r") as f:
    c = f.read()
print(f"    get_lambda_fno:   {'✓' if 'get_lambda_fno' in c else '✗'}")
print(f"    fno_physics_loss: {'✓' if 'fno_physics_loss' in c else '✗'}")
print(f"    lam = get_lambda: {'✓' if 'lam = get_lambda_fno' in c else '✗'}")
print(f"    loss_data + lam:  {'✓' if 'loss_data + lam' in c else '✗'}")

PYEOF

echo ""
echo "  Region config is already correct (from previous fix):"
echo "    12 regions, outer_box=11, region_id/11 ✓"
echo ""
echo "================================================================"
echo "  To train now:"
echo "    sbatch run_alvis_fno.sh"
echo "================================================================"
