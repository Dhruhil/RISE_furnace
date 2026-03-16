"""
Physics-Informed Training Loop — Heat Treatment GNN.

This is a direct upgrade of the original train.py.
The only change: MaskedMSELoss is replaced by PhysicsInformedLoss
which adds three physics residual terms from your thesis:

  Conduction  — Fourier's Law:        rho * Cp * dT/dt = kappa * laplacian(T)
  Convection  — Newton's Law:         T_surface must not exceed T_set
  Radiation   — Stefan-Boltzmann:     dT/dt proportional to (T_set^4 - T^4)

Lambda schedule (physics weight grows over training):
  Epoch   1– 50 :  0.001   model learns basic patterns from data first
  Epoch  51–100 :  0.01    light physics guidance added
  Epoch 101–150 :  0.05    balanced data + physics
  Epoch 151–200 :  0.10    full physics enforcement

What is printed every epoch:
  Epoch  10/200 | TrLoss:0.00821 | VaLoss:0.00743 | MAE:8.3K | R2:0.9712 |
  W5K:71.2% | W10K:88.4% | Cond:0.0012 | Conv:0.0008 | Rad:0.0003 | lam:0.001

Usage:
    python train.py
"""

from __future__ import annotations

import time
import json

import numpy as np
import torch

from configs.base_config import BaseConfig, CONFIG
from data.dataset import get_dataloaders
from models.meshgraphnet import HeatTreatmentGNN
from training.loss import PhysicsInformedLoss
from training.scheduler import build_scheduler
from utils.metrics import compute_metrics
from utils.logging import setup_logging, log_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Lambda schedule
# Physics weight starts near-zero so the model learns temperature patterns
# from data first, then is gradually pushed to obey physics equations.
# ─────────────────────────────────────────────────────────────────────────────
def get_lambda(epoch: int, n_epochs: int) -> float:
    progress = epoch / n_epochs
    if progress < 0.25:
        return 0.001    # epochs  1– 50: mostly data
    elif progress < 0.50:
        return 0.01     # epochs 51–100: light physics
    elif progress < 0.75:
        return 0.05     # epochs 101–150: balanced
    else:
        return 0.10     # epochs 151–200: full physics


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helper
# ─────────────────────────────────────────────────────────────────────────────
def within_tolerance(y_pred: np.ndarray, y_true: np.ndarray,
                     tolerance_K: float) -> float:
    return 100.0 * np.sum(np.abs(y_pred - y_true) <= tolerance_K) / len(y_true)


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, cfg, lam):
    model.train()
    criterion.lambda_physics = lam    # update physics weight for this epoch

    total_loss                      = 0.0
    total_cond = total_conv = total_rad = 0.0
    n_batches                       = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        delta_T_pred = model(batch)

        Y_std = (float(batch.Y_std[0])
                 if hasattr(batch.Y_std, "__len__")
                 else float(batch.Y_std))

        loss, breakdown = criterion(
            delta_T_pred = delta_T_pred,
            target       = batch.y,
            batch        = batch,
            Y_std        = Y_std,
            dt           = cfg.dt,
        )

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cond += breakdown["cond"]
        total_conv += breakdown["conv"]
        total_rad  += breakdown["rad"]
        n_batches  += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "cond": total_cond / n,
        "conv": total_conv / n,
        "rad":  total_rad  / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — returns full metrics dict in Kelvin
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        delta_T_pred = model(batch)

        Y_std = (float(batch.Y_std[0])
                 if hasattr(batch.Y_std, "__len__")
                 else float(batch.Y_std))

        loss, _ = criterion(
            delta_T_pred = delta_T_pred,
            target       = batch.y,
            batch        = batch,
            Y_std        = Y_std,
            dt           = cfg.dt,
        )
        total_loss += loss.item()
        n_batches  += 1

        T_pred = (batch.T_current
                  + delta_T_pred.squeeze(-1).cpu() * Y_std).numpy().ravel()
        T_true = batch.T_next.cpu().numpy().ravel()
        all_pred.append(T_pred)
        all_true.append(T_true)

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    m = compute_metrics(y_pred, y_true)
    m["loss"]       = total_loss / max(n_batches, 1)
    m["within_5K"]  = within_tolerance(y_pred, y_true,  5.0)
    m["within_10K"] = within_tolerance(y_pred, y_true, 10.0)
    m["within_20K"] = within_tolerance(y_pred, y_true, 20.0)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Dataset split summary printed at startup
# ─────────────────────────────────────────────────────────────────────────────
def print_split_summary(cfg, train_loader, val_loader, test_loader):
    n_tr = len(train_loader.dataset)
    n_va = len(val_loader.dataset)
    n_te = len(test_loader.dataset)
    total = n_tr + n_va + n_te

    n_sims_total = 50
    n_test_sims  = max(1, int(n_sims_total * cfg.test_fraction))
    n_val_sims   = max(1, int(n_sims_total * cfg.val_fraction))
    n_train_sims = n_sims_total - n_val_sims - n_test_sims

    print(f"\n{'='*68}")
    print(f"  DATASET SPLIT  (50 LHS simulations, split by simulation)")
    print(f"{'='*68}")
    print(f"  {'Split':<8} {'Sims':>5}  {'%':>4}  {'Pairs':>10}  {'Purpose'}")
    print(f"  {'-'*60}")
    print(f"  {'TRAIN':<8} {n_train_sims:>5}  {100*n_tr//total:>3}%  {n_tr:>10,}  "
          f"Model learns weights from these")
    print(f"  {'VAL':<8} {n_val_sims:>5}  {100*n_va//total:>3}%  {n_va:>10,}  "
          f"Pick best checkpoint each epoch")
    print(f"  {'TEST':<8} {n_test_sims:>5}  {100*n_te//total:>3}%  {n_te:>10,}  "
          f"Final accuracy — never seen in training")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<8} {n_sims_total:>5}  100%  {total:>10,}  "
          f"pairs = (sim, timestep) graphs")
    print(f"{'='*68}")
    print(f"\n  1 pair = temperature graph at time t → predict ΔT to t+{cfg.dt:.0f}s")
    print(f"  Each sim ≈ 399 time steps × ~450 cells = ~179,550 node predictions")
    print(f"{'='*68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(cfg: BaseConfig = CONFIG):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*68}")
    print(f"  PHYSICS-INFORMED HEAT TREATMENT GNN")
    print(f"{'='*68}")
    print(f"  Dataset  : {cfg.dataset_path}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {cfg.n_epochs}")
    print(f"  LR       : {cfg.learning_rate}")
    print(f"  Batch    : {cfg.batch_size} graphs/batch")
    print(f"\n  Physics equations (from thesis):")
    print(f"    Conduction  Fourier      : rho*Cp*dT/dt = kappa*laplacian(T)")
    print(f"    Convection  Newton       : T_surface <= T_set")
    print(f"    Radiation   Stefan-Boltz : dT/dt ~ epsilon*sigma*(T_set^4 - T^4)")
    print(f"\n  Lambda schedule (physics weight over epochs):")
    print(f"    Epoch   1– 50 : 0.001  (learn data patterns first)")
    print(f"    Epoch  51–100 : 0.01   (add light physics)")
    print(f"    Epoch 101–150 : 0.05   (balanced)")
    print(f"    Epoch 151–200 : 0.10   (full physics enforcement)")

    logger = setup_logging(cfg)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(cfg, rollout_steps=1)
    print_split_summary(cfg, train_loader, val_loader, test_loader)

    # ── Model ─────────────────────────────────────────────────────────
    model     = HeatTreatmentGNN(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)

    # Physics-informed loss with all three heat transfer terms
    criterion = PhysicsInformedLoss(
        lambda_physics = 0.001,   # starts small, updated each epoch by schedule
        w_cond         = 1.0,     # conduction is the dominant term
        w_conv         = 0.5,     # convection boundary condition
        w_rad          = 0.3,     # radiation from heaters
        epsilon_steel  = 0.8,     # typical emissivity for steel surfaces
    )

    # ── History ───────────────────────────────────────────────────────
    history = {
        "train_loss":     [],
        "val_loss":       [],
        "val_mae":        [],
        "val_rmse":       [],
        "val_r2":         [],
        "val_within_5K":  [],
        "val_within_10K": [],
        # physics loss breakdown per epoch
        "loss_cond":      [],
        "loss_conv":      [],
        "loss_rad":       [],
        "lambda":         [],
    }
    best_val_mae = float("inf")

    # ── Column header ─────────────────────────────────────────────────
    print(f"{'='*90}")
    print(f"  TRAINING PROGRESS  (all temperatures in Kelvin)")
    print(f"{'='*90}")
    print(
        f"  {'Ep':>5} | {'TrLoss':>8} | {'VaLoss':>8} | "
        f"{'MAE':>6} | {'R2':>6} | "
        f"{'W5K':>6} | {'W10K':>6} | "
        f"{'Cond':>8} | {'Conv':>8} | {'Rad':>8} | {'lam':>6}"
    )
    print(f"  {'-'*90}")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, cfg.n_epochs + 1):
        t0  = time.time()
        lam = get_lambda(epoch, cfg.n_epochs)

        train_out   = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, lam
        )
        val_metrics = evaluate(model, val_loader, criterion, device, cfg)
        scheduler.step(val_metrics["loss"])

        # Record history
        history["train_loss"].append(train_out["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_within_5K"].append(val_metrics["within_5K"])
        history["val_within_10K"].append(val_metrics["within_10K"])
        history["loss_cond"].append(train_out["cond"])
        history["loss_conv"].append(train_out["conv"])
        history["loss_rad"].append(train_out["rad"])
        history["lambda"].append(lam)

        is_best = val_metrics["mae"] < best_val_mae

        # Print every epoch
        if epoch % cfg.log_every_n_epochs == 0 or epoch == 1 or is_best:
            print(
                f"  {epoch:5d} | {train_out['loss']:8.5f} | "
                f"{val_metrics['loss']:8.5f} | "
                f"{val_metrics['mae']:5.2f}K | {val_metrics['r2']:6.4f} | "
                f"{val_metrics['within_5K']:5.1f}% | "
                f"{val_metrics['within_10K']:5.1f}% | "
                f"{train_out['cond']:8.5f} | "
                f"{train_out['conv']:8.5f} | "
                f"{train_out['rad']:8.5f} | "
                f"{lam:6.4f}"
                + ("  ◄ BEST" if is_best else "")
            )
            log_metrics(logger, epoch, train_out["loss"], val_metrics, cfg)

        # Periodic checkpoint
        if epoch % cfg.save_every_n_epochs == 0:
            ckpt = f"{cfg.checkpoint_dir}/checkpoint_epoch{epoch:04d}.pt"
            model.save(ckpt, epoch, optimizer.state_dict(), val_metrics)

        # Best model checkpoint
        if is_best:
            best_val_mae = val_metrics["mae"]
            model.save(
                f"{cfg.checkpoint_dir}/best_model.pt",
                epoch, optimizer.state_dict(), val_metrics,
            )

    # ── Final test evaluation ─────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  FINAL TEST ACCURACY")
    print(f"  (5 held-out simulations — NEVER seen during training)")
    print(f"{'='*68}")

    model = HeatTreatmentGNN.load(
        f"{cfg.checkpoint_dir}/best_model.pt", cfg, str(device)
    )
    test = evaluate(model, test_loader, criterion, device, cfg)

    print(f"\n  MAE          : {test['mae']:.2f} K")
    print(f"  RMSE         : {test['rmse']:.2f} K")
    print(f"  Max Error    : {test['max_err']:.2f} K")
    print(f"  R²           : {test['r2']:.4f}")
    print(f"\n  Within  5 K  : {test['within_5K']:.1f}%")
    print(f"  Within 10 K  : {test['within_10K']:.1f}%")
    print(f"  Within 20 K  : {test['within_20K']:.1f}%")
    print(f"\n  Physics equations enforced during training:")
    print(f"    Conduction  (Fourier)        rho*Cp*dT/dt = kappa*laplacian(T)")
    print(f"    Convection  (Newton)         T_surface <= T_set")
    print(f"    Radiation   (Stefan-Boltzm.) dT/dt ~ epsilon*sigma*(T_set^4 - T^4)")
    print(f"\n  R²={test['r2']:.4f}  → "
          + ("Excellent" if test['r2'] >= 0.99 else
             "Good"      if test['r2'] >= 0.95 else "Needs more training"))
    print(f"  Within 10K={test['within_10K']:.1f}%  → "
          + ("Very accurate" if test['within_10K'] >= 90 else
             "Acceptable"    if test['within_10K'] >= 75 else "Train longer"))

    # Save training history
    hist_path = f"{cfg.output_dir}/logs/training_history.json"
    with open(hist_path, "w") as f:
        json.dump({
            "physics_equations": {
                "conduction": "rho * Cp * dT/dt = kappa * laplacian(T)  [Fourier]",
                "convection": "T_surface <= T_set                        [Newton]",
                "radiation":  "dT/dt ~ epsilon * sigma * (T_set^4 - T^4) [Stefan-Boltzmann]",
            },
            "physics_weights": {
                "w_conduction": 1.0,
                "w_convection": 0.5,
                "w_radiation":  0.3,
                "epsilon_steel": 0.8,
            },
            "lambda_schedule": {
                "epoch_1_50":    0.001,
                "epoch_51_100":  0.01,
                "epoch_101_150": 0.05,
                "epoch_151_200": 0.10,
            },
            "config": {
                "n_simulations": 50,
                "n_train_sims":  int(50 * (1 - cfg.val_fraction - cfg.test_fraction)),
                "n_val_sims":    int(50 * cfg.val_fraction),
                "n_test_sims":   int(50 * cfg.test_fraction),
                "n_epochs":      cfg.n_epochs,
                "learning_rate": cfg.learning_rate,
                "batch_size":    cfg.batch_size,
            },
            "history":      history,
            "best_val_mae": best_val_mae,
            "test_metrics": test,
        }, f, indent=2)

    print(f"\n  History   → {hist_path}")
    print(f"  Best model → {cfg.checkpoint_dir}/best_model.pt")
    print(f"\n  Best Val MAE = {best_val_mae:.2f} K")
    print(f"  Test MAE     = {test['mae']:.2f} K\n")


if __name__ == "__main__":
    main()