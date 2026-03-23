"""
Physics-Informed FNO Training — All Regions.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Trains on t=0-3200s (80%) per simulation.
After training, verifies on t=3200-4000s (20%, never seen during training).

Usage:
    python train.py --epochs 200 --lr 1e-3
    python train.py --epochs 60 --batch 16
"""
from __future__ import annotations

import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from configs.fno_config import FNOConfig, CONFIG
from data.dataset import get_fno_dataloaders, get_fno_eval_dataset
from models.fno_model import HeatTreatmentFNO
from models.rollout import rollout_fno_all_regions
from training.scheduler import build_scheduler
from utils.metrics import compute_metrics, within_tolerance
from utils.logging import setup_logging, log_metrics
from utils.checkpoint import CheckpointManager


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    """One-step validation — compute loss and metrics in Kelvin."""
    model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    ds = loader.dataset

    for x, y, T_cur, T_next, *_ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        total_loss += loss.item()
        n += 1

        T_pred_K = (pred.squeeze(1).cpu().numpy() * ds.T_std + ds.T_mean).ravel()
        T_true_K = T_next.numpy().ravel()
        all_pred.append(T_pred_K)
        all_true.append(T_true_K)

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    m = compute_metrics(y_pred, y_true)
    m["loss"]       = total_loss / max(n, 1)
    m["within_5K"]  = within_tolerance(y_pred, y_true, 5.0)
    m["within_10K"] = within_tolerance(y_pred, y_true, 10.0)
    return m


# ─────────────────────────────────────────────────────────────────────
# Verification rollout
# ─────────────────────────────────────────────────────────────────────

def run_verification(model, cfg, device, save_dir):
    """Full rollout on test sims — Phase 1 + Phase 2 per region."""
    dataset = get_fno_eval_dataset(cfg)
    n_train = cfg.n_train_steps
    start_t = 40
    p1_end  = n_train - start_t

    all_p1_mae, all_p2_mae = [], []
    all_p1_r2, all_p2_r2   = [], []
    per_region_mae = {}

    sep  = "=" * 70
    dash = "-" * 72

    print(f"\n{sep}")
    print(f"  FNO ROLLOUT EVALUATION — ALL REGIONS (t={start_t*cfg.dt:.0f} -> {cfg.predict_time_end:.0f}s)")
    print(f"{sep}")
    print(f"  {'Sim':>4}  {'Region':>16}  {'P1 MAE':>10}  {'P2 MAE':>10}")
    print(f"  {dash}")

    for sim_i in dataset.sim_indices:
        results = rollout_fno_all_regions(
            model, dataset, sim_i, start_t=start_t, device=device
        )
        sim_p1_pred, sim_p1_true = [], []
        sim_p2_pred, sim_p2_true = [], []

        for region, (T_pred, T_true) in results.items():
            ns = T_pred.shape[0]
            p1s = min(p1_end + 1, ns)
            m1 = compute_metrics(T_pred[:p1s].ravel(), T_true[:p1s].ravel())
            sim_p1_pred.append(T_pred[:p1s].ravel())
            sim_p1_true.append(T_true[:p1s].ravel())

            if p1_end < ns and p1_end < T_true.shape[0]:
                gt_end = min(ns, T_true.shape[0])
                m2 = compute_metrics(T_pred[p1_end:gt_end].ravel(),
                                     T_true[p1_end:gt_end].ravel())
                sim_p2_pred.append(T_pred[p1_end:gt_end].ravel())
                sim_p2_true.append(T_true[p1_end:gt_end].ravel())
            else:
                m2 = {"mae": float("nan")}

            print(f"  {sim_i:>4}  {region:>16}  "
                  f"P1={m1['mae']:.2f}K  P2={m2['mae']:.2f}K")

            if region not in per_region_mae:
                per_region_mae[region] = {"p1": [], "p2": []}
            per_region_mae[region]["p1"].append(m1["mae"])
            if not np.isnan(m2["mae"]):
                per_region_mae[region]["p2"].append(m2["mae"])

        if sim_p1_pred:
            agg1 = compute_metrics(np.concatenate(sim_p1_pred),
                                   np.concatenate(sim_p1_true))
            all_p1_mae.append(agg1["mae"])
            all_p1_r2.append(agg1["r2"])
        if sim_p2_pred:
            agg2 = compute_metrics(np.concatenate(sim_p2_pred),
                                   np.concatenate(sim_p2_true))
            all_p2_mae.append(agg2["mae"])
            all_p2_r2.append(agg2["r2"])
        print()

    summary = {
        "phase1": {
            "mean_mae": float(np.mean(all_p1_mae)) if all_p1_mae else None,
            "mean_r2":  float(np.mean(all_p1_r2))  if all_p1_r2  else None,
        },
        "phase2": {
            "mean_mae": float(np.mean(all_p2_mae)) if all_p2_mae else None,
            "mean_r2":  float(np.mean(all_p2_r2))  if all_p2_r2  else None,
        },
        "per_region": {
            r: {"p1": float(np.mean(v["p1"])),
                "p2": float(np.mean(v["p2"])) if v["p2"] else None}
            for r, v in per_region_mae.items()
        },
    }

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/fno_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{sep}")
    print(f"  FNO SUMMARY — ALL REGIONS")
    print(f"{sep}")
    p1 = summary["phase1"]
    p2 = summary["phase2"]
    print(f"  Phase 1 (training):     MAE={p1['mean_mae']:.2f}K  R2={p1['mean_r2']:.4f}")
    print(f"  Phase 2 (verification): MAE={p2['mean_mae']:.2f}K  R2={p2['mean_r2']:.4f}")
    print(f"\n  Per-region:")
    for r, v in summary["per_region"].items():
        p2_str = f"{v['p2']:.2f}K" if v["p2"] else "N/A"
        print(f"    {r:>16}: P1={v['p1']:.2f}K  P2={p2_str}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────

def main(cfg=None):
    parser = argparse.ArgumentParser(description="Train PhysicsNeMo FNO — All Regions")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--device",  default=None)
    parser.add_argument("--modes",   type=int,   default=None)
    parser.add_argument("--layers",  type=int,   default=None)
    parser.add_argument("--latent",  type=int,   default=None)
    args = parser.parse_args()

    if cfg is None:
        cfg = CONFIG
    if args.epochs: cfg.n_epochs      = args.epochs
    if args.lr:     cfg.learning_rate = args.lr
    if args.batch:  cfg.batch_size    = args.batch
    if args.modes:  cfg.fno_modes     = args.modes
    if args.layers: cfg.fno_layers    = args.layers
    if args.latent: cfg.fno_latent    = args.latent

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger = setup_logging(cfg)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  FNO TRAINING — All Regions (NVIDIA PhysicsNeMo)")
    print(f"  Dataset : {cfg.dataset_path}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {cfg.n_epochs}  LR: {cfg.learning_rate}  Batch: {cfg.batch_size}")
    print(f"  FNO     : modes={cfg.fno_modes} layers={cfg.fno_layers} latent={cfg.fno_latent}")
    print(f"  Train   : t=0-{cfg.train_time_end:.0f}s | Verify: t={cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s")
    print(f"{sep}\n")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_fno_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────
    model     = HeatTreatmentFNO(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg)
    ckpt_mgr  = CheckpointManager(cfg.checkpoint_dir)

    # ── Training loop ─────────────────────────────────────────────────
    print(f"  {'Ep':>5} | {'TrLoss':>9} | {'VaLoss':>9} | "
          f"{'MAE[K]':>7} | {'R2':>7} | {'W5K':>6} | {'LR':>9}")
    print(f"  {'-'*70}")

    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        # Train
        model.train()
        total_loss, nb = 0.0, 0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        tr_loss = total_loss / max(nb, 1)

        # Validate
        val_m = validate(model, val_loader, device)
        scheduler.step(val_m["loss"])
        lr = optimizer.param_groups[0]["lr"]

        # Checkpoint
        is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_m)

        if epoch % cfg.save_every_n_epochs == 0:
            ckpt_mgr.save_periodic(model, optimizer, scheduler, epoch, val_m)

        # Log
        if epoch % cfg.log_every_n_epochs == 0 or epoch == 1 or is_best:
            tag = "  < BEST" if is_best else ""
            print(f"  {epoch:>5} | {tr_loss:>9.5f} | {val_m['loss']:>9.5f} | "
                  f"{val_m['mae']:>7.2f} | {val_m['r2']:>7.4f} | "
                  f"{val_m['within_5K']:>6.1f} | {lr:>9.2e}{tag}")
            log_metrics(logger, epoch, tr_loss, val_m, cfg)

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed/60:.1f} min")
    print(f"  Best MAE: {ckpt_mgr.best_mae:.3f}K  (epoch {ckpt_mgr.best_epoch})")
    print(f"  Checkpoint: {ckpt_mgr.best_path}")

    # ── Verification rollout ──────────────────────────────────────────
    print(f"\n  Loading best model for rollout evaluation...")
    model = HeatTreatmentFNO.load(ckpt_mgr.best_path, cfg, str(device))
    run_verification(model, cfg, str(device), f"{cfg.output_dir}/evaluation")


if __name__ == "__main__":
    main()
