"""
PINN Training — All Regions, PhysicsNeMo framework.
Master's Thesis: Digital Twin Modeling of Heat Treatment in Cast Metals

Two-phase training:
  Phase A: Data-only pretraining (learn approximate T field)
  Phase B: Physics-informed fine-tuning (enforce PDE + boundary conditions)

GPU-optimized: ALL data preloaded to GPU. No DataLoader bottleneck.
Same evaluation as GNN/FNO: Phase 1 (0-3200s) + Phase 2 (3200-4000s).

Usage:
    python train.py
    python train.py --pretrain_epochs 2000 --physics_epochs 3000
"""
from __future__ import annotations

import argparse
import sys
import time
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from configs.pinn_config import PINNConfig, CONFIG
from data.dataset import PINNAllRegionsDataset
from models.pinn_model import HeatTreatmentPINN
from models.physics import compute_pde_residual
from utils.metrics import compute_metrics, within_tolerance
from utils.checkpoint import CheckpointManager


@torch.no_grad()
def validate(model, val_X, val_Y, ds, batch_size=65536):
    model.eval()
    N = val_X.shape[0]
    preds = []
    for i in range(0, N, batch_size):
        preds.append(model(val_X[i:i+batch_size]))
    pred = torch.cat(preds)
    val_loss = F.mse_loss(pred, val_Y).item()
    T_pred_K = (pred.cpu().numpy().ravel() * ds.T_std + ds.T_mean)
    T_true_K = (val_Y.cpu().numpy().ravel() * ds.T_std + ds.T_mean)
    m = compute_metrics(T_pred_K, T_true_K)
    m["loss"] = val_loss
    m["within_5K"] = within_tolerance(T_pred_K, T_true_K, 5.0)
    m["within_10K"] = within_tolerance(T_pred_K, T_true_K, 10.0)
    return m


def run_evaluation(model, cfg, device, save_dir):
    model.eval()
    train_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "train", "training")
    test_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "test", "evaluation")
    test_ds.set_norm_stats(train_ds)
    dev = torch.device(device)
    x_n = (test_ds.x_raw - test_ds.x_mean) / test_ds.x_std
    y_n = (test_ds.y_raw - test_ds.y_mean) / test_ds.y_std
    z_n = (test_ds.z_raw - test_ds.z_mean) / test_ds.z_std
    t_n = (test_ds.t_raw - test_ds.t_mean) / test_ds.t_std
    Tset_n = (test_ds.Tset_raw - test_ds.Tset_mean) / test_ds.Tset_std
    rid_n = test_ds.rid_raw / 11.0
    test_X = torch.tensor(np.stack([x_n, y_n, z_n, t_n, Tset_n, rid_n], axis=1),
                           dtype=torch.float32, device=dev)
    t_raw = test_ds.t_raw
    T_true = test_ds.T_raw
    preds = []
    bs = 65536
    with torch.no_grad():
        for i in range(0, test_X.shape[0], bs):
            preds.append(model(test_X[i:i+bs]))
    pred = torch.cat(preds)
    T_pred = pred.cpu().numpy().ravel() * test_ds.T_std + test_ds.T_mean
    p1_mask = t_raw <= cfg.train_time_end
    p2_mask = t_raw > cfg.train_time_end
    m1 = compute_metrics(T_pred[p1_mask], T_true[p1_mask]) if p1_mask.sum() > 0 else {}
    m2 = compute_metrics(T_pred[p2_mask], T_true[p2_mask]) if p2_mask.sum() > 0 else {}
    summary = {
        "phase1": {"mean_mae": m1.get("mae"), "mean_r2": m1.get("r2"), "n_points": int(p1_mask.sum())},
        "phase2": {"mean_mae": m2.get("mae"), "mean_r2": m2.get("r2"), "n_points": int(p2_mask.sum())},
    }
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/pinn_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  PINN EVALUATION")
    print(f"{sep}")
    p1 = summary["phase1"]
    p2 = summary["phase2"]
    print(f"  Phase 1 (0-{cfg.train_time_end:.0f}s):    MAE={p1['mean_mae']:.2f}K  R2={p1['mean_r2']:.4f}  ({p1['n_points']:,} pts)")
    print(f"  Phase 2 ({cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s): MAE={p2['mean_mae']:.2f}K  R2={p2['mean_r2']:.4f}  ({p2['n_points']:,} pts)")
    print(f"{sep}")
    return summary


def main(cfg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_epochs", type=int, default=None)
    parser.add_argument("--physics_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if cfg is None:
        cfg = CONFIG
    if args.pretrain_epochs: cfg.n_epochs_pretrain = args.pretrain_epochs
    if args.physics_epochs: cfg.n_epochs_physics = args.physics_epochs
    if args.lr: cfg.learning_rate = args.lr
    if args.batch: cfg.batch_size = args.batch
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = cfg.batch_size
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  PINN TRAINING — GPU-Optimized (All Regions)")
    print(f"  Dataset : {cfg.dataset_path}")
    print(f"  Device  : {device}  Batch: {batch_size}")
    print(f"  Phase A : {cfg.n_epochs_pretrain} epochs (data-only)")
    print(f"  Phase B : {cfg.n_epochs_physics} epochs (physics-informed)")
    print(f"  PDE     : rho*Cp*dT/dt = kappa*Laplacian(T)")
    print(f"{sep}\n")
    print("  Loading datasets...")
    train_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "val", "training")
    val_ds.set_norm_stats(train_ds)
    print("  Preloading ALL data to GPU...")
    def build_gpu_tensors(ds):
        x_n = (ds.x_raw - ds.x_mean) / ds.x_std
        y_n = (ds.y_raw - ds.y_mean) / ds.y_std
        z_n = (ds.z_raw - ds.z_mean) / ds.z_std
        t_n = (ds.t_raw - ds.t_mean) / ds.t_std
        Tset_n = (ds.Tset_raw - ds.Tset_mean) / ds.Tset_std
        rid_n = ds.rid_raw / 11.0
        T_n = (ds.T_raw - ds.T_mean) / ds.T_std
        inputs = torch.tensor(np.stack([x_n, y_n, z_n, t_n, Tset_n, rid_n], axis=1), dtype=torch.float32, device=device)
        targets = torch.tensor(T_n[:, None], dtype=torch.float32, device=device)
        return inputs, targets
    train_X, train_Y = build_gpu_tensors(train_ds)
    val_X, val_Y = build_gpu_tensors(val_ds)
    N_train = train_X.shape[0]
    n_batches = (N_train + batch_size - 1) // batch_size
    print(f"  Train: {N_train:,} on GPU | Val: {val_X.shape[0]:,} on GPU")
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB | Batches/ep: {n_batches}")
    print(f"  T: mean={train_ds.T_mean:.1f}K std={train_ds.T_std:.1f}K")
    model = HeatTreatmentPINN(cfg).to(device)
    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)
    print(f"\n{'='*70}")
    print(f"  PHASE A: DATA-ONLY ({cfg.n_epochs_pretrain} epochs)")
    print(f"{'='*70}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs_pretrain, eta_min=1e-6)
    print(f"  {'Ep':>6} | {'TrLoss':>9} | {'VaLoss':>9} | {'MAE[K]':>7} | {'R2':>7} | {'W5K':>6}")
    t0 = time.time()
    for epoch in range(1, cfg.n_epochs_pretrain + 1):
        model.train()
        perm = torch.randperm(N_train, device=device)
        total_loss = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            optimizer.zero_grad()
            pred = model(train_X[idx])
            loss = F.mse_loss(pred, train_Y[idx])
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        tr_loss = total_loss / n_batches
        if epoch % cfg.log_every == 0 or epoch == 1:
            val_m = validate(model, val_X, val_Y, train_ds)
            is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_m)
            tag = "  < BEST" if is_best else ""
            print(f"  {epoch:>6} | {tr_loss:>9.5f} | {val_m['loss']:>9.5f} | {val_m['mae']:>7.2f} | {val_m['r2']:>7.4f} | {val_m['within_5K']:>6.1f}{tag}")
    print(f"\n  Phase A done in {(time.time()-t0)/60:.1f} min | Best MAE: {ckpt_mgr.best_mae:.3f}K")
    print(f"\n{'='*70}")
    print(f"  PHASE B: PHYSICS-INFORMED ({cfg.n_epochs_physics} epochs)")
    print(f"  Lambda: linear 0->1 | PDE: heat equation via autograd")
    print(f"{'='*70}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_physics, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs_physics, eta_min=1e-6)
    print(f"  {'Ep':>6} | {'TrLoss':>9} | {'VaLoss':>9} | {'MAE[K]':>7} | {'R2':>7} | {'Phys':>9} | {'lam':>7}")
    phys_norm = 1.0
    n_phys = min(1024, batch_size)
    t1 = time.time()
    for epoch in range(1, cfg.n_epochs_physics + 1):
        lam = epoch / cfg.n_epochs_physics
        model.train()
        perm = torch.randperm(N_train, device=device)
        total_loss, total_phys, total_data = 0.0, 0.0, 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            optimizer.zero_grad()
            pred = model(train_X[idx])
            loss_data = F.mse_loss(pred, train_Y[idx])
            loss_phys = torch.tensor(0.0, device=device)
            if lam > 1e-10:
                phys_idx = idx[:n_phys]
                _, residual, _ = compute_pde_residual(model, train_X[phys_idx], cfg, train_ds)
                loss_phys = residual.pow(2).mean()
            loss = (1.0 - lam) * loss_data + lam * loss_phys / (phys_norm + 1e-8)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_data += loss_data.item()
            total_phys += loss_phys.item()
        scheduler.step()
        tr_loss = total_loss / n_batches
        tr_data = total_data / n_batches
        tr_phys = total_phys / n_batches
        if tr_phys > 1e-8:
            phys_norm = 0.9 * phys_norm + 0.1 * (tr_phys / (tr_data + 1e-8))
        if epoch % cfg.save_every == 0:
            ckpt_mgr.save_periodic(model, optimizer, scheduler, cfg.n_epochs_pretrain + epoch, {})
        if epoch % cfg.log_every == 0 or epoch == 1:
            val_m = validate(model, val_X, val_Y, train_ds)
            is_best = ckpt_mgr.update(model, optimizer, scheduler, cfg.n_epochs_pretrain + epoch, val_m)
            tag = "  < BEST" if is_best else ""
            print(f"  {epoch:>6} | {tr_loss:>9.5f} | {val_m['loss']:>9.5f} | {val_m['mae']:>7.2f} | {val_m['r2']:>7.4f} | {tr_phys:>9.3e} | {lam:>7.4f}{tag}")
    print(f"\n  Phase B done in {(time.time()-t1)/60:.1f} min")
    print(f"  Total: {(time.time()-t0)/60:.1f} min | Best MAE: {ckpt_mgr.best_mae:.3f}K")
    print(f"\n  Loading best model for evaluation...")
    model = HeatTreatmentPINN.load(ckpt_mgr.best_path, cfg, str(device))
    run_evaluation(model, cfg, str(device), f"{cfg.output_dir}/evaluation")


if __name__ == "__main__":
    main()
