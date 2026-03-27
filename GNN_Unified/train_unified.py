"""
Unified Multi-Region GNN Training with Pushforward + Physics.
All 12 regions in ONE graph — heat flows between regions.
Master's Thesis: Digital Twin Modeling of Heat Treatment
"""
from __future__ import annotations
import sys, time, math, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, "/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official")
sys.path.insert(0, ".")

from configs.base_config import CONFIG, BaseConfig
from models.meshgraphnet import HeatTreatmentGNN
from utils.checkpoint import CheckpointManager
from utils.metrics import compute_metrics, within_tolerance

import importlib.util
spec = importlib.util.spec_from_file_location(
    "dataset_unified",
    "/mimer/NOBACKUP/groups/revar/GNN_Unified/data/dataset_unified.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
UnifiedDataset = mod.UnifiedDataset

SIGMA = 5.67e-8


def physics_loss_unified(pred, batch, dT_std, dT_mean, device):
    """Physics constraints for unified graph.
    1. Conduction: graph Laplacian (now includes cross-region edges!)
    2. Convection: T <= T_set
    3. Radiation: Stefan-Boltzmann
    """
    dt = 10.0
    dT_pred = pred.squeeze(-1) * dT_std + dT_mean
    T_now = batch.T_current.to(device)
    T_next = T_now + dT_pred
    T_set = batch.T_set_raw.to(device)

    # Convection: non-heater nodes must not exceed T_set
    non_heater = ~batch.is_heater.bool()
    overshoot = torch.nn.functional.relu(T_next - T_set) * non_heater.float()
    L_conv = (overshoot / T_set.clamp(min=300)).pow(2).mean()

    # Conduction via graph Laplacian (cross-region heat flow!)
    src, dst = batch.edge_index[0], batch.edge_index[1]
    N = T_now.shape[0]
    T_diff = T_now[dst] - T_now[src]
    lap_T = torch.zeros(N, device=device, dtype=T_now.dtype)
    degree = torch.zeros(N, device=device, dtype=T_now.dtype)
    lap_T.scatter_add_(0, src, T_diff)
    degree.scatter_add_(0, src, torch.ones_like(T_diff))
    lap_T = lap_T / degree.clamp(min=1.0)
    dT_dt = dT_pred / dt
    scale = dT_dt.abs().mean().clamp(min=1e-6)
    L_cond = ((dT_dt - lap_T * 0.001) / scale).pow(2).mean()

    # Radiation: Stefan-Boltzmann (non-heater nodes only)
    Q_rad = 0.8 * SIGMA * (T_set.pow(4) - T_now.pow(4))
    dT_rad = Q_rad / (7800.0 * 450.0 * 0.01)
    scale_r = dT_rad.abs().mean().clamp(min=1e-8)
    L_rad = ((dT_dt - dT_rad) / scale_r).pow(2).mean()

    return 0.5 * L_conv + 0.3 * L_cond + 0.2 * L_rad


def train_one_epoch(model, loader, optimizer, device, cfg, lam=0.0, phys_norm=1.0):
    """3-step pushforward + physics on unified graph."""
    model.train()
    total_loss, total_data, total_phys, n = 0.0, 0.0, 0.0, 0
    dT_std = loader.dataset.dT_std
    dT_mean = loader.dataset.dT_mean
    T_mean = loader.dataset.T_mean
    T_std_ds = loader.dataset.T_std

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        heater_mask = batch.is_heater.unsqueeze(-1).bool()

        # Step 1: predict from ground truth
        pred1 = model(batch)
        pred1 = pred1.masked_fill(heater_mask, 0.0)
        target1 = batch.y.masked_fill(heater_mask, 0.0)
        loss1 = F.mse_loss(pred1, target1)

        # Step 2: feed own prediction
        dT_pred1 = pred1.squeeze(-1) * dT_std + dT_mean
        T_pred1 = batch.T_current + dT_pred1
        T_pred1 = torch.where(batch.is_heater.bool(), batch.T_set_raw, T_pred1)

        batch2 = batch.clone()
        batch2.x = batch.x.clone()
        batch2.x[:, 3] = (T_pred1 - T_mean) / (T_std_ds + 1e-8)
        batch2.x[:, 6] = batch.x[:, 6] + 1.0 / 400.0
        batch2.T_current = T_pred1

        pred2 = model(batch2)
        pred2 = pred2.masked_fill(heater_mask, 0.0)
        target2 = batch.y2.masked_fill(heater_mask, 0.0)
        loss2 = F.mse_loss(pred2, target2)

        # Combined data loss (2-step pushforward)
        loss_data = 1.0 * loss1 + 0.5 * loss2

        # Physics loss (on step 1)
        if lam > 1e-10:
            L_phys = physics_loss_unified(
                pred1, batch, dT_std, dT_mean, str(device))
            total_phys += L_phys.item()
            loss = (1.0 - lam) * loss_data + lam * L_phys / (phys_norm + 1e-8)
        else:
            loss = loss_data

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_data += loss1.item()
        n += 1

    tr_data = total_data / max(n, 1)
    tr_phys = total_phys / max(n, 1)
    if tr_phys > 1e-8:
        phys_norm = 0.9 * phys_norm + 0.1 * (tr_phys / (tr_data + 1e-8))
    return total_loss / max(n, 1), phys_norm


@torch.no_grad()
def evaluate(model, loader, device, lam=0.0, phys_norm=1.0):
    """Single-step validation on unified graph."""
    model.eval()
    total_loss, total_total, n = 0.0, 0.0, 0
    all_pred_K, all_true_K = [], []
    dT_std = loader.dataset.dT_std
    dT_mean = loader.dataset.dT_mean
    import numpy as np

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        heater_mask = batch.is_heater.unsqueeze(-1).bool()
        pred_masked = pred.masked_fill(heater_mask, 0.0)
        target_masked = batch.y.masked_fill(heater_mask, 0.0)
        loss = F.mse_loss(pred_masked, target_masked)
        total_loss += loss.item()
        if lam > 1e-10:
            L_phys = physics_loss_unified(pred, batch, dT_std, dT_mean, str(device))
            total_total += ((1.0 - lam) * loss.item() + lam * L_phys.item() / (phys_norm + 1e-8))
        else:
            total_total += loss.item()
        n += 1

        non_heater = ~batch.is_heater.bool()
        dT_pred = pred.squeeze(-1) * dT_std + dT_mean
        T_pred = batch.T_current + dT_pred
        T_true = batch.T_next
        all_pred_K.append(T_pred[non_heater].cpu().numpy())
        all_true_K.append(T_true[non_heater].cpu().numpy())

    all_pred_K = np.concatenate(all_pred_K)
    all_true_K = np.concatenate(all_true_K)
    m = compute_metrics(all_pred_K, all_true_K)
    m["loss"] = total_loss / max(n, 1)
    m["total_loss"] = total_total / max(n, 1)
    m["within_5K"] = within_tolerance(all_pred_K, all_true_K, 5.0)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  UNIFIED MULTI-REGION GNN — Pushforward + Physics")
    print(f"  All 12 regions in ONE graph with cross-region edges")
    print(f"  Device: {device}  Batch: {args.batch}  Epochs: {args.epochs}")
    print(f"  Lambda: linear 0->1 | Physics: conv + cond + rad")
    print(f"{sep}\n")

    h5 = "dataset_all_regions.h5"
    train_ds = UnifiedDataset(h5, cfg, "train", "training")
    val_ds = UnifiedDataset(h5, cfg, "val", "training")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=2)

    cfg.node_in_features = 7
    model = HeatTreatmentGNN(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6)
    ckpt_mgr = CheckpointManager(
        "/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/checkpoints")

    print(f"  {'Ep':>5} | {'TrLoss':>9} | {'VaLoss':>9} | "
          f"{'MAE[K]':>7} | {'R2':>7} | {'W5K':>6} | {'lam':>6}")
    print(f"  {'-'*65}")

    phys_norm = 1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        lam = min(epoch / args.epochs, 0.3)  # max 30% physics

        tr_loss, phys_norm = train_one_epoch(
            model, train_loader, optimizer, device, cfg,
            lam=lam, phys_norm=phys_norm)

        val_m = evaluate(model, val_loader, device, lam=lam, phys_norm=phys_norm)
        scheduler.step(val_m["loss"])
        is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_m)

        if epoch % 10 == 0:
            ckpt_mgr.save_periodic(model, optimizer, scheduler, epoch, val_m)

        lr = optimizer.param_groups[0]["lr"]
        tag = "  < BEST" if is_best else ""
        print(f"  {epoch:>5} | {tr_loss:>9.5f} | "
              f"{val_m['total_loss']:>9.5f} | "
              f"{val_m['mae']:>7.2f} | {val_m['r2']:>7.4f} | "
              f"{val_m['within_5K']:>6.1f} | {lam:>6.3f}{tag}")

    total_time = time.time() - t0
    print(f"\n  Training done in {total_time/60:.1f} min")
    print(f"  Best MAE: {ckpt_mgr.best_mae:.3f}K (epoch {ckpt_mgr.best_epoch})")


if __name__ == "__main__":
    main()
