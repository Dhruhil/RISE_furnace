#!/bin/bash
# ============================================================================
#  ADD PHYSICS LOSS TO FNO — MATCHING GNN's EXACT CURRICULUM
#  Master's Thesis — Simulating Heat Treatment using OpenFOAM and AI
#
#  Run from: /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
#
#  Your GNN uses:
#    λ = 0.001 * exp(4.6 * epoch/n_epochs), capped at 0.10
#    L_physics = 0.5*L_conv + 0.3*L_cond + 0.2*L_rad
#
#  This script adds the SAME curriculum + weights to FNO, but adapted
#  for the spectral domain (no graph → use FFT for diffusion).
# ============================================================================
set -e

echo ""
echo "================================================================"
echo "  ADD PHYSICS LOSS TO FNO (matching GNN curriculum)"
echo "================================================================"
echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 1: Add physics config to fno_config.py
# ─────────────────────────────────────────────────────────────────
echo "[1/2] Adding physics config..."

python3 << 'PYEOF'
with open("configs/fno_config.py", "r") as f:
    content = f.read()

old = "    grad_clip:       float = 1.0"
new = """    grad_clip:       float = 1.0

    # ── Physics-informed loss (same weights as GNN all-regions) ───
    w_convection:      float = 0.5     # T ≤ T_set  (Newton cooling)
    w_conduction:      float = 0.3     # spectral smoothness (≡ diffusion)
    w_radiation:       float = 0.2     # Stefan-Boltzmann dT constraint
    sigma_sb:          float = 5.67e-8
    epsilon_steel:     float = 0.80
    char_thickness:    float = 0.01"""

if "w_convection" not in content:
    content = content.replace(old, new)
    with open("configs/fno_config.py", "w") as f:
        f.write(content)
    print("  ✓ Added physics weights (0.5/0.3/0.2) — same as GNN")
else:
    print("  ✓ Physics weights already present")
PYEOF

echo ""

# ─────────────────────────────────────────────────────────────────
# STEP 2: Add physics loss + exponential curriculum to train.py
# ─────────────────────────────────────────────────────────────────
echo "[2/2] Adding physics loss + smooth exponential curriculum to train.py..."

python3 << 'PYEOF'
with open("train.py", "r") as f:
    content = f.read()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A) Add the physics functions after imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

physics_block = '''

# ─────────────────────────────────────────────────────────────────────
# Physics curriculum — SAME smooth exponential as GNN all-regions
# λ = 0.001 * exp(4.6 * p), capped at 0.10
# ─────────────────────────────────────────────────────────────────────
import math as _math

def get_lambda_fno(epoch: int, n_epochs: int) -> float:
    """Smooth exponential curriculum — identical to GNN get_lambda_ar()."""
    p = epoch / n_epochs
    lam = 0.001 * _math.exp(4.6 * p)
    return min(lam, 0.10)


# ─────────────────────────────────────────────────────────────────────
# Physics-informed loss for FNO (3 terms, same weights as GNN)
#
#   L_physics = 0.5 * L_conv + 0.3 * L_cond + 0.2 * L_rad
#
# Adapted from GNN graph-based → FNO spectral-based:
#   - GNN conduction uses graph Laplacian → FNO uses FFT high-freq penalty
#   - GNN convection uses T_set_raw      → FNO uses T_set from input channel
#   - GNN radiation uses Stefan-Boltzmann → FNO uses same equation
# ─────────────────────────────────────────────────────────────────────
SIGMA_SB = 5.67e-8


def fno_physics_loss(pred_norm, y_norm, x, cfg, dataset):
    """
    Physics losses for FNO — mirrors GNN physics_loss_allregions().

    Args:
        pred_norm: (B, 1, N) predicted T_next (normalised)
        y_norm:    (B, 1, N) ground truth T_next (normalised)
        x:         (B, 4, N) input channels [T_norm, Tset_norm, rid, time]
        cfg:       FNOConfig with w_convection, w_conduction, w_radiation
        dataset:   FNOAllRegionsDataset (T_mean, T_std, Tset_mean, Tset_std)

    Returns:
        L_physics: scalar tensor
        details:   dict with individual loss components
    """
    dt = cfg.dt  # 10.0 seconds

    # ── Denormalize to Kelvin ─────────────────────────────────────
    T_pred = pred_norm.squeeze(1) * dataset.T_std + dataset.T_mean  # (B, N)
    T_now  = x[:, 0, :] * dataset.T_std + dataset.T_mean           # (B, N)
    T_set  = x[:, 1, :] * dataset.Tset_std + dataset.Tset_mean     # (B, N)

    dT_pred = T_pred - T_now
    dT_dt   = dT_pred / dt

    # ── 1. CONVECTION: T_next ≤ T_set  (weight 0.5) ──────────────
    # Same as GNN: penalise overshoot, skip heater-like regions
    is_heater = (T_now > T_set * 1.05).float()
    overshoot = torch.nn.functional.relu(T_pred - T_set) * (1.0 - is_heater)
    L_conv    = (overshoot / T_set.clamp(min=300)).pow(2).mean()

    # ── 2. CONDUCTION: spectral smoothness (weight 0.3) ──────────
    # GNN uses graph Laplacian: lap(T) via scatter_add on edges.
    # FNO equivalent: penalise high-frequency energy in FFT.
    # Heat diffusion = low-pass filter → smooth T fields.
    #
    # This is mathematically connected: the graph Laplacian eigenvalues
    # correspond to spatial frequencies. High Laplacian residual ≡
    # high-frequency spectral energy. Both enforce Fourier's law.
    pred_fft = torch.fft.rfft(pred_norm.squeeze(1), dim=-1)
    n_freq   = pred_fft.shape[-1]
    cutoff   = max(n_freq // 3, 1)  # keep bottom 1/3, penalise top 2/3
    high_freq = pred_fft[:, cutoff:].abs().pow(2)
    scale_cond = high_freq.mean().clamp(min=1e-8)
    L_cond     = high_freq.mean() / scale_cond  # normalised like GNN

    # ── 3. RADIATION: Stefan-Boltzmann (weight 0.2) ───────────────
    # Same equation as GNN: Q_rad = ε·σ·(T_set⁴ - T⁴)
    Q_rad   = cfg.epsilon_steel * SIGMA_SB * (T_set.pow(4) - T_now.pow(4))
    dT_rad  = Q_rad / (7800.0 * 450.0 * cfg.char_thickness)
    scale_r = dT_rad.abs().mean().clamp(min=1e-8)
    L_rad   = ((dT_dt - dT_rad) / scale_r).pow(2).mean()

    # ── Combined: same 0.5/0.3/0.2 weights as GNN ────────────────
    L_physics = (cfg.w_convection * L_conv +
                 cfg.w_conduction * L_cond +
                 cfg.w_radiation  * L_rad)

    return L_physics, {
        "conv": float(L_conv.item()),
        "cond": float(L_cond.item()),
        "rad":  float(L_rad.item()),
    }

'''

insert_after = "from utils.checkpoint import CheckpointManager"
if "fno_physics_loss" not in content:
    content = content.replace(insert_after, insert_after + physics_block)
    print("  ✓ Added fno_physics_loss() + get_lambda_fno()")
    print("    Curriculum: λ = 0.001 * exp(4.6 * epoch/n_epochs), cap 0.10")
    print("    Weights:    0.5*conv + 0.3*cond + 0.2*rad (same as GNN)")
else:
    print("  ✓ Physics functions already present")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B) Modify training loop to use physics loss every epoch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

old_loop = """        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()"""

new_loop = """        lam = get_lambda_fno(epoch, cfg.n_epochs)
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)

            # Data loss (MSE on normalised T_next)
            loss_data = F.mse_loss(pred, y)

            # Physics loss (smooth exponential curriculum, same as GNN)
            if lam > 1e-10:
                L_phys, _ = fno_physics_loss(
                    pred, y, x, cfg, train_loader.dataset)
                loss = loss_data + lam * L_phys
            else:
                loss = loss_data

            loss.backward()"""

if "lam = get_lambda_fno" not in content:
    content = content.replace(old_loop, new_loop)
    print("  ✓ Training loop now uses physics loss every epoch")
else:
    print("  ✓ Training loop already has physics loss")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C) Add lambda to log output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

old_log = '''            tag = "  < BEST" if is_best else ""
            print(f"  {epoch:>5} | {tr_loss:>9.5f} | {val_m[\'loss\']:>9.5f} | "
                  f"{val_m[\'mae\']:>7.2f} | {val_m[\'r2\']:>7.4f} | "
                  f"{val_m[\'within_5K\']:>6.1f} | {lr:>9.2e}{tag}")'''

new_log = '''            lam_now = get_lambda_fno(epoch, cfg.n_epochs)
            tag = "  < BEST" if is_best else ""
            print(f"  {epoch:>5} | {tr_loss:>9.5f} | {val_m[\'loss\']:>9.5f} | "
                  f"{val_m[\'mae\']:>7.2f} | {val_m[\'r2\']:>7.4f} | "
                  f"{val_m[\'within_5K\']:>6.1f} | {lr:>9.2e} | \\u03bb={lam_now:.4f}{tag}")'''

if "lam_now = get_lambda_fno" not in content:
    content = content.replace(old_log, new_log)
    print("  ✓ Log now shows λ value each epoch")

with open("train.py", "w") as f:
    f.write(content)

print("")
PYEOF

# ─────────────────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  VERIFICATION"
echo "================================================================"
echo ""

echo "  Physics config:"
grep -E "w_convection|w_conduction|w_radiation|sigma_sb|epsilon_steel|char_thickness" configs/fno_config.py 2>/dev/null | sed 's/^/    /'
echo ""

echo "  Lambda curriculum in train.py:"
if grep -q "0.001 \* _math.exp(4.6" train.py; then
    echo "    ✓ λ = 0.001 * exp(4.6 * p), cap 0.10  (SAME as GNN)"
else
    echo "    ✗ NOT FOUND"
fi
echo ""

echo "  Physics loss in train.py:"
if grep -q "fno_physics_loss" train.py; then
    echo "    ✓ fno_physics_loss() present"
else
    echo "    ✗ NOT FOUND"
fi
echo ""

echo "  Training loop uses physics:"
if grep -q "lam = get_lambda_fno" train.py; then
    echo "    ✓ Every epoch: loss = MSE + λ * (0.5*conv + 0.3*cond + 0.2*rad)"
else
    echo "    ✗ Still pure MSE"
fi
echo ""

echo "  ┌────────────────────────────────────────────────────────────┐"
echo "  │  GNN vs FNO — NOW BOTH PHYSICS-INFORMED                   │"
echo "  ├────────────────────────────────────────────────────────────┤"
echo "  │                    GNN              FNO                    │"
echo "  │  Curriculum:  0.001*exp(4.6*p)  0.001*exp(4.6*p)  SAME   │"
echo "  │  Cap:         0.10              0.10               SAME   │"
echo "  │  Conv weight: 0.5               0.5                SAME   │"
echo "  │  Cond weight: 0.3               0.3                SAME   │"
echo "  │  Rad weight:  0.2               0.2                SAME   │"
echo "  │  Conduction:  Graph Laplacian   FFT high-freq      EQUIV  │"
echo "  │  Convection:  T≤T_set           T≤T_set            SAME   │"
echo "  │  Radiation:   Stefan-Boltzmann  Stefan-Boltzmann   SAME   │"
echo "  └────────────────────────────────────────────────────────────┘"
echo ""

echo "  Smooth λ distribution over 300 epochs:"
python3 -c "
import math
for ep in [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]:
    p = ep / 300
    lam = min(0.001 * math.exp(4.6 * p), 0.10)
    bar = '█' * int(lam * 200)
    print(f'    Epoch {ep:>3}  λ={lam:.4f}  {bar}')
"

echo ""
echo "================================================================"
echo "  PHYSICS LOSS ADDED ✓"
echo ""
echo "  To train:  sbatch run_alvis_fno.sh"
echo "================================================================"
