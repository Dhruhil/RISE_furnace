#!/bin/bash
# ============================================================
# Fix FNO: Predict T_next directly (not delta_T)
# Inner_box grid (no outer_box) already applied
#
# Before: target = (T_next - T_current - dT_mean) / dT_std  (~tiny)
# After:  target = (T_next - T_mean) / T_std  (~full range)
#
# cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# bash fix_fno_tnext.sh
# ============================================================

set -euo pipefail
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "============================================"
echo "  Fix: Predict T_next instead of delta_T"
echo "============================================"

# Fix dataset.py: change target from dT_norm to T_next_norm
python3 << 'XEOF'
with open("data/dataset.py", "r") as f:
    code = f.read()

# Replace target computation
old_target = '''        dT = T_grid_tp1 - T_grid_t
        dT_norm = (dT - self.dT_mean) / self.dT_std'''

new_target = '''        # Target: T_next normalised (not delta_T!)
        # Predicting full temperature field gives much stronger loss signal
        dT = T_grid_tp1 - T_grid_t  # keep for reference
        dT_norm = (dT - self.dT_mean) / self.dT_std  # kept for compatibility'''

if old_target in code:
    code = code.replace(old_target, new_target)
    print("  OK: Added comment to target section")

# Replace y construction: use T_next_norm instead of dT_norm
old_y = '''        # Target: (1, Gx, Gy, Gz)
        y = dT_norm[None, ...]'''

new_y = '''        # Target: T_next normalised (full field, not delta)
        T_next_norm = (T_grid_tp1 - self.T_mean) / self.T_std
        y = T_next_norm[None, ...]'''

if old_y in code:
    code = code.replace(old_y, new_y)
    print("  OK: Target changed to T_next_norm")

with open("data/dataset.py", "w") as f:
    f.write(code)
XEOF


echo ""
echo "  Rewriting train.py for T_next prediction..."

cat > train.py << 'PYEOF'
"""
3D FNO — Predicts T_next directly (not delta_T).
Inner_box grid, region-weighted loss, pushforward, physics.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from configs.fno_config import CONFIG
from data.dataset import get_fno3d_dataloaders
from models.fno_model import HeatTreatmentFNO3D
from utils.metrics import compute_metrics, within_tolerance
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logging


def get_physics_lambda(epoch, n_epochs):
    warmup_end = int(n_epochs * 0.2)
    if epoch <= warmup_end:
        return 0.0
    return 0.1 * (epoch - warmup_end) / (n_epochs - warmup_end)

def get_pushforward_weight(epoch, n_epochs):
    warmup_end = int(n_epochs * 0.15)
    if epoch <= warmup_end:
        return 0.0
    return 0.5 * (epoch - warmup_end) / (n_epochs - warmup_end)

def get_warmup_lr(epoch, base_lr, warmup_epochs=5):
    if epoch <= warmup_epochs:
        return base_lr * (0.1 + 0.9 * epoch / warmup_epochs)
    return base_lr


def weighted_mse(pred, target, weight):
    """MSE weighted by region importance."""
    diff2 = (pred - target).pow(2)
    w = weight.unsqueeze(1)
    return (diff2 * w).sum() / (w.sum() * pred.shape[1] + 1e-8)


def physics_loss_3d(pred, x, T_mean, T_std):
    """Physics on T_next prediction."""
    device = pred.device
    # pred is normalised T_next: (B, 1, Gx, Gy, Gz)
    # x[:, 0] is normalised T_current
    T_pred_norm = pred.squeeze(1)
    T_cur_norm = x[:, 0]
    Tset_norm = x[:, 1]
    is_heater = x[:, 4]
    non_heater = (1.0 - is_heater)

    # Convection: T_next should not exceed T_set
    overshoot = F.relu(T_pred_norm - Tset_norm) * non_heater
    L_conv = overshoot.pow(2).mean()

    # Spectral smoothness: T_next should be smooth
    pred_fft = torch.fft.rfftn(T_pred_norm, dim=[-3, -2, -1])
    Nx, Ny, Nz_half = pred_fft.shape[-3], pred_fft.shape[-2], pred_fft.shape[-1]
    cx, cy, cz = max(Nx // 3, 1), max(Ny // 3, 1), max(Nz_half // 3, 1)
    high_freq = pred_fft[:, cx:, cy:, cz:].abs().pow(2)
    L_smooth = high_freq.mean()

    # Equilibrium: when T ~ T_set, T_next ~ T_current (no change)
    gap = (Tset_norm - T_cur_norm).abs()
    near_eq = torch.exp(-gap * 5.0) * non_heater
    change = (T_pred_norm - T_cur_norm)
    L_eq = (change * near_eq).pow(2).mean()

    return 0.5 * L_conv + 0.3 * L_smooth + 0.2 * L_eq


@torch.no_grad()
def evaluate(model, loader, device, train_ds, lam=0.0):
    model.eval()
    total_data, total_wdata, total_phys, n = 0.0, 0.0, 0.0, 0
    all_pred, all_true = [], []
    steel_pred, steel_true = [], []
    T_mean, T_std = train_ds.T_mean, train_ds.T_std

    for batch in loader:
        x, y, T_cur, T_next_gt, weight = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        pred = model(x)

        loss = F.mse_loss(pred, y)
        wloss = weighted_mse(pred, y, weight)
        total_data += loss.item()
        total_wdata += wloss.item()
        if lam > 1e-6:
            lp = physics_loss_3d(pred, x, T_mean, T_std)
            total_phys += lp.item()
        n += 1

        # Denormalise: pred is normalised T_next
        T_pred_K = pred.squeeze(1).cpu().numpy() * T_std + T_mean
        T_true_K = T_next_gt.cpu().numpy()
        all_pred.append(T_pred_K.ravel())
        all_true.append(T_true_K.ravel())

        # Steel-only
        rid = x[:, 2].cpu().numpy()
        is_steel = rid < 0.05
        if is_steel.any():
            steel_pred.append(T_pred_K[is_steel])
            steel_true.append(T_true_K[is_steel])

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    m = compute_metrics(all_pred, all_true)

    if steel_pred:
        sp = np.concatenate(steel_pred)
        st = np.concatenate(steel_true)
        m["steel_mae"] = float(np.mean(np.abs(sp - st)))
    else:
        m["steel_mae"] = 0.0

    avg_data = total_data / max(n, 1)
    avg_wdata = total_wdata / max(n, 1)
    avg_phys = total_phys / max(n, 1)
    m["loss_data"] = avg_data
    m["loss_wdata"] = avg_wdata
    m["loss_phys"] = avg_phys
    m["loss"] = avg_wdata
    m["within_5K"] = within_tolerance(all_pred, all_true, 5.0)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    cfg = CONFIG
    if args.epochs: cfg.n_epochs = args.epochs
    if args.batch: cfg.batch_size = args.batch
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  3D FNO — Predict T_next (inner_box grid)")
    print(f"  Device: {device}  Batch: {cfg.batch_size}  Epochs: {cfg.n_epochs}")
    print(f"  Grid: {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}  "
          f"Modes: {cfg.fno_modes}  Layers: {cfg.fno_layers}  Latent: {cfg.fno_latent}")
    print(f"  Target: T_next (not delta_T)")
    print(f"  Loss: region-weighted (steel=10x, air=3x)")
    print(f"{sep}\n")

    train_loader, val_loader, train_ds, val_ds = get_fno3d_dataloaders(cfg)

    T_mean = train_ds.T_mean
    T_std = train_ds.T_std
    print(f"  Stats: T_mean={T_mean:.1f}K  T_std={T_std:.1f}K\n")

    if args.test:
        print("  === SANITY TEST ===")
        batch = next(iter(train_loader))
        x, y, T_cur, T_next_gt, weight = batch
        print(f"  x: {x.shape}, y: {y.shape}, weight: {weight.shape}")
        print(f"  y range: [{y.min():.3f}, {y.max():.3f}] (normalised T_next)")
        print(f"  T_next range: [{T_next_gt.min():.1f}, {T_next_gt.max():.1f}]K")
        model = HeatTreatmentFNO3D(cfg).to(device)
        out = model(x.to(device))
        loss = weighted_mse(out, y.to(device), weight.to(device))
        loss.backward()
        print(f"  Forward+backward OK: loss={loss.item():.4f}")
        print(f"  === ALL CHECKS PASSED ===")
        return

    model = HeatTreatmentFNO3D(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)
    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)

    print(f"  {'Ep':>4} | {'TrWLoss':>9} | {'TrData':>9} | {'TrPhys':>9} | "
          f"{'VaWLoss':>9} | {'VaData':>9} | "
          f"{'MAE':>6} | {'Steel':>6} | {'R2':>7} | "
          f"{'lam':>5} | {'w2':>4} | {'Time':>5}")
    print(f"  {'-'*110}")

    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        lam = get_physics_lambda(epoch, cfg.n_epochs)
        w2 = get_pushforward_weight(epoch, cfg.n_epochs)
        ep_start = time.time()

        lr = get_warmup_lr(epoch, args.lr, warmup_epochs=5)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss, total_data, total_phys, nb = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            x, y, T_cur, T_next_gt, weight = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            optimizer.zero_grad()

            # Step 1: predict T_next from ground truth T_current
            pred1 = model(x)
            loss1 = weighted_mse(pred1, y, weight)

            # Step 2: pushforward — use predicted T_next as new input
            loss_data = loss1
            if w2 > 1e-6:
                # pred1 is normalised T_next prediction
                x2 = x.clone()
                x2[:, 0] = pred1.squeeze(1).detach()  # update T channel
                x2[:, 3] = x[:, 3] + 10.0 / 4000.0   # advance time

                pred2 = model(x2)
                loss2 = weighted_mse(pred2, y, weight)  # approximate target
                loss_data = loss1 + w2 * loss2

            # Physics
            if lam > 1e-6:
                L_phys = physics_loss_3d(pred1, x, T_mean, T_std)
                loss = (1 - lam) * loss_data + lam * L_phys
                total_phys += L_phys.item()
            else:
                loss = loss_data

            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_data += loss1.item()
            nb += 1

        tr_loss = total_loss / max(nb, 1)
        tr_data = total_data / max(nb, 1)
        tr_phys = total_phys / max(nb, 1)

        val_m = evaluate(model, val_loader, device, train_ds, lam=lam)
        if epoch > 5:
            scheduler.step(val_m["loss"])

        is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_m)
        if epoch % 10 == 0:
            ckpt_mgr.save_periodic(model, optimizer, scheduler, epoch, val_m)

        ep_time = time.time() - ep_start
        tag = " *" if is_best else ""
        print(f"  {epoch:>4} | {tr_loss:>9.5f} | {tr_data:>9.5f} | "
              f"{tr_phys:>9.5f} | {val_m['loss_wdata']:>9.5f} | "
              f"{val_m['loss_data']:>9.5f} | "
              f"{val_m['mae']:>6.2f} | {val_m['steel_mae']:>6.2f} | "
              f"{val_m['r2']:>7.4f} | "
              f"{lam:>5.3f} | {w2:>4.2f} | {ep_time:>5.0f}s{tag}")

    total_time = time.time() - t0
    print(f"\n  Training done in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
    print(f"  Best steel MAE: {ckpt_mgr.best_mae:.3f}K (epoch {ckpt_mgr.best_epoch})")

    # Inference speed test
    print(f"\n  === INFERENCE SPEED TEST ===")
    model.eval()
    batch = next(iter(val_loader))
    x_test = batch[0].to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_test)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x_test)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = (time.time() - t_inf) / 100
    t_full = t_inf * int(cfg.t_total / cfg.dt)
    print(f"  Single step: {t_inf*1000:.2f} ms")
    print(f"  Full rollout (400 steps): {t_full:.2f}s")
    print(f"  Speedup vs OpenFOAM: ~{3600*3/t_full:.0f}x")


if __name__ == "__main__":
    main()
PYEOF

echo "  OK: train.py rewritten for T_next prediction"

echo ""
echo "============================================"
echo "  VERIFICATION"
echo "============================================"
python3 -c "import ast; ast.parse(open('train.py').read()); print('  OK: train.py')"
python3 -c "import ast; ast.parse(open('data/dataset.py').read()); print('  OK: dataset.py')"

grep "T_next_norm" data/dataset.py | head -2
grep "Predict T_next" train.py | head -1

echo ""
echo "============================================"
echo "  DONE"
echo "============================================"
echo ""
echo "  Target changed: delta_T -> T_next"
echo "  Before: y = (T_next - T_current - dT_mean) / dT_std  (tiny values)"
echo "  After:  y = (T_next - T_mean) / T_std  (full range 300-1100K)"
echo ""
echo "  Expected epoch 1:"
echo "    MAE ~50-100K  (model starts far from correct T field)"
echo "    Steel ~30-80K (must learn heating curve)"
echo "    R2 ~0.85-0.95 (not trivially 1.0)"
echo ""
echo "  Expected epoch 100:"
echo "    MAE ~3-8K     (smooth decrease)"
echo "    Steel ~2-5K   (good accuracy)"
echo "    R2 ~0.995+    (high but earned)"
echo ""
echo "  scancel <jobid>"
echo "  sbatch run_alvis_fno.sh"
