"""
Train GNN on ALL regions — steel, air, heaters, brick.
Usage:
    python train_all_regions.py --epochs 200 --lr 1e-3
"""

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")
from configs.base_config import BaseConfig, CONFIG
from data.dataset_all_regions import AllRegionsDataset, get_all_regions_dataloaders
from models.meshgraphnet import HeatTreatmentGNN
from training.scheduler import build_scheduler
from utils.metrics import compute_metrics, within_tolerance
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logging, log_metrics


def get_lambda_ar(epoch: int, n_epochs: int) -> float:
    """Smooth exponential physics curriculum."""
    import math
    p = epoch / n_epochs
    lam = p  # linear: 0.0 at epoch 1 → 1.0 at last epoch
    return min(lam, 1.0)
def physics_loss_allregions(
    pred:       torch.Tensor,
    batch,
    dT_std:     float,
    dT_mean:    float,
    device:     str,
) -> torch.Tensor:
    """
    Physics losses for all regions:
    1. Conduction: rho*Cp*dT/dt = kappa * laplacian(T)  [steel only]
    2. Convection: T_steel <= T_set                      [all regions]
/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/allregions_%j.log    3. Radiation:  Stefan-Boltzmann                      [steel only]
    """
    SIGMA = 5.67e-8
    dt    = 10.0

    # Denormalise predicted dT
    dT_pred = pred.squeeze(-1) * dT_std + dT_mean
    T_now   = batch.T_current.to(device)
    T_next  = T_now + dT_pred
    T_set   = batch.T_set_raw.to(device)

    # Convection: temperature must not exceed T_set
    is_heater = (T_now > T_set * 1.05).float()
    overshoot = torch.nn.functional.relu(T_next - T_set) * (1.0 - is_heater)
    L_conv    = (overshoot / T_set.clamp(min=300)).pow(2).mean()

    # Conduction via graph Laplacian
    src, dst = batch.edge_index[0], batch.edge_index[1]
    N        = T_now.shape[0]
    T_diff   = T_now[dst] - T_now[src]
    lap_T    = torch.zeros(N, device=device, dtype=T_now.dtype)
    degree   = torch.zeros(N, device=device, dtype=T_now.dtype)
    lap_T.scatter_add_(0, src, T_diff)
    degree.scatter_add_(0, src, torch.ones_like(T_diff))
    lap_T    = lap_T / degree.clamp(min=1.0)
    dT_dt    = dT_pred / dt
    scale    = dT_dt.abs().mean().clamp(min=1e-6)
    L_cond   = ((dT_dt - lap_T * 0.001) / scale).pow(2).mean()

    # Radiation: Stefan-Boltzmann
    Q_rad    = 0.8 * SIGMA * (T_set.pow(4) - T_now.pow(4))
    dT_rad   = Q_rad / (7800.0 * 450.0 * 0.01)
    scale_r  = dT_rad.abs().mean().clamp(min=1e-8)
    L_rad    = ((dT_dt - dT_rad) / scale_r).pow(2).mean()

    return 0.5 * L_conv + 0.3 * L_cond + 0.2 * L_rad


def train_one_epoch(model, loader, optimizer, device, cfg, Y_std, lam=0.001, phys_norm=1.0):
    """3-step pushforward training: model learns from its own predictions."""
    model.train()
    total_loss = 0.0
    total_data = 0.0
    total_phys = 0.0
    n = 0
    dT_std_val = loader.dataset.dT_std
    dT_mean_val = loader.dataset.dT_mean
    T_mean = loader.dataset.T_mean
    T_std_ds = loader.dataset.T_std

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Step 1: predict from ground truth input
        pred1 = model(batch)
        loss1 = F.mse_loss(pred1, batch.y)

        # Step 2: update features with OWN prediction, predict again
        dT_pred1 = pred1.squeeze(-1) * dT_std_val + dT_mean_val
        T_pred1 = batch.T_current + dT_pred1

        batch2 = batch.clone()
        batch2.x = batch.x.clone()
        batch2.x[:, 3] = (T_pred1 - T_mean) / (T_std_ds + 1e-8)
        batch2.x[:, 6] = batch.x[:, 6] + 1.0 / 400.0
        batch2.T_current = T_pred1

        pred2 = model(batch2)
        loss2 = F.mse_loss(pred2, batch.y2)

        # Step 3: update features with step 2 prediction
        dT_pred2 = pred2.squeeze(-1) * dT_std_val + dT_mean_val
        T_pred2 = T_pred1 + dT_pred2

        batch3 = batch.clone()
        batch3.x = batch2.x.clone()
        batch3.x[:, 3] = (T_pred2 - T_mean) / (T_std_ds + 1e-8)
        batch3.x[:, 6] = batch2.x[:, 6] + 1.0 / 400.0
        batch3.T_current = T_pred2

        pred3 = model(batch3)
        loss3 = F.mse_loss(pred3, batch.y3)

        # Combined loss: step1=1.0, step2=0.5, step3=0.25
        loss_data = 1.0 * loss1 + 0.5 * loss2 + 0.25 * loss3

        # Physics loss on step 1
        if lam > 1e-10:
            dT_std_f = (float(batch.dT_std[0]) if hasattr(batch.dT_std, "__len__") else float(batch.dT_std))
            dT_mean_f = (float(batch.dT_mean[0]) if hasattr(batch.dT_mean, "__len__") else float(batch.dT_mean))
            L_phys = physics_loss_allregions(pred1, batch, dT_std_f, dT_mean_f, str(device))
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
def evaluate(model, loader, device, cfg):
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    total_data = 0.0
    total_phys = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch)
        loss  = F.mse_loss(pred, batch.y)
        total_loss += loss.item()
        n += 1

        dT_std  = float(batch.dT_std[0]) if hasattr(batch.dT_std, "__len__") else float(batch.dT_std)
        dT_mean = float(batch.dT_mean[0]) if hasattr(batch.dT_mean, "__len__") else float(batch.dT_mean)

        T_pred_K = (batch.T_current.cpu() +
                    pred.squeeze(-1).cpu() * dT_std + dT_mean).numpy().ravel()
        T_true_K = batch.T_next.cpu().numpy().ravel()
        all_pred.append(T_pred_K)
        all_true.append(T_true_K)

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    m = compute_metrics(y_pred, y_true)
    m["loss"]       = total_loss / max(n, 1)
    m["within_5K"]  = within_tolerance(y_pred, y_true,  5.0)
    m["within_10K"] = within_tolerance(y_pred, y_true, 10.0)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=200)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--batch",  type=int,   default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = CONFIG
    cfg.n_epochs      = args.epochs
    cfg.learning_rate = args.lr
    cfg.batch_size    = args.batch

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"\n{'='*70}")
    print(f"  TRAINING GNN — ALL REGIONS")
    print(f"  Dataset: {cfg.all_regions_dataset_path}")
    print(f"  Device:  {device}")
    print(f"{'='*70}\n")

    # Override node features for all-regions model
    cfg_ar = BaseConfig()
    cfg_ar.node_in_features = 7   # x,y,z,T,T_set,region_id,time
    cfg_ar.n_epochs         = args.epochs
    cfg_ar.learning_rate    = args.lr
    cfg_ar.batch_size       = args.batch
    cfg_ar.checkpoint_dir   = str(
        Path(cfg.checkpoint_dir).parent / "checkpoints_allregions"
    )
    cfg_ar.log_dir = str(
        Path(cfg.log_dir).parent / "logs_allregions"
    )
    Path(cfg_ar.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    import os
    os.makedirs(cfg_ar.log_dir, exist_ok=True)
    os.makedirs(cfg_ar.checkpoint_dir, exist_ok=True)
    # Use separate logger to avoid conflict with steel cylinder logger
    import logging

    log_path = Path(cfg_ar.log_dir) / "train_allregions.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt     = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                 datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(fmt)

    logger = logging.getLogger("heat_gnn_allregions")
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info("Log file: %s", log_path)
    loaders   = get_all_regions_dataloaders(cfg_ar)
    train_loader, val_loader, test_loader = loaders

    model     = HeatTreatmentGNN(cfg_ar).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg_ar.learning_rate,
        weight_decay=cfg_ar.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg_ar)
    ckpt_mgr  = CheckpointManager(cfg_ar.checkpoint_dir)

    print(f"\n{'='*70}")
    print(f"  {'Ep':>5} | {'TrLoss':>9} | {'VaLoss':>9} | "
          f"{'MAE[K]':>7} | {'R2':>7} | {'W5K':>6}")
    print(f"  {'-'*60}")

    phys_norm = 1.0

    for epoch in range(1, cfg_ar.n_epochs + 1):
        lam = get_lambda_ar(epoch, cfg_ar.n_epochs)
        tr_loss, phys_norm = train_one_epoch(
            model, train_loader, optimizer, device, cfg_ar,
            Y_std=train_loader.dataset.T_std,
            lam=lam, phys_norm=phys_norm,
        )
        val_m = evaluate(model, val_loader, device, cfg_ar)
        scheduler.step(val_m["loss"])

        is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_m)

        if epoch % cfg_ar.save_every_n_epochs == 0:
            ckpt_mgr.save_periodic(model, optimizer, scheduler, epoch, val_m)

        if epoch % cfg_ar.log_every_n_epochs == 0 or epoch == 1 or is_best:
            print(
                f"  {epoch:>5} | {tr_loss:>9.5f} | "
                f"{val_m['loss']:>9.5f} | "
                f"{val_m['mae']:>7.2f} | {val_m['r2']:>7.4f} | "
                f"{val_m['within_5K']:>6.1f} | λ={lam:.4f}"
                + ("  ◄ BEST" if is_best else "")
            )
            log_metrics(logger, epoch, tr_loss, val_m, cfg_ar)

    print(f"\nBest checkpoint: {ckpt_mgr.best_path}")
    print(f"Best val MAE: {ckpt_mgr.best_mae:.3f} K")


if __name__ == "__main__":
    main()
