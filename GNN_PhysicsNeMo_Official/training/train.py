"""
Physics-Informed Training Loop — Option A Temporal Split.

Trains on t=0–3200s (80%) per simulation.
After training, verifies on t=3200–4000s (20%, never seen during training).
Both windows have OpenFOAM ground truth → full verification.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from configs.base_config import BaseConfig, CONFIG
from data.dataset import get_dataloaders, get_evaluation_dataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout import rollout_from_dataset
from training.loss import PhysicsInformedLoss
from training.scheduler import build_scheduler
from utils.metrics import compute_metrics, within_tolerance, metrics_per_timestep
from utils.logging import setup_logging, log_metrics
from utils.checkpoint import CheckpointManager


# ─────────────────────────────────────────────────────────────────────────────
# Lambda curriculum
# ─────────────────────────────────────────────────────────────────────────────
def _denorm_dT(delta_T_pred, batch):
    """Denormalise using dT_std, not Y_std."""
    if hasattr(batch, "dT_std"):
        s = float(batch.dT_std[0]) if hasattr(batch.dT_std, "__len__") else float(batch.dT_std)
        m = float(batch.dT_mean[0]) if hasattr(batch.dT_mean, "__len__") else float(batch.dT_mean)
    else:
        s = float(batch.Y_std[0]) if hasattr(batch.Y_std, "__len__") else float(batch.Y_std)
        m = 0.0
    return delta_T_pred.squeeze(-1).cpu() * s + m

def get_lambda(epoch: int, n_epochs: int) -> float:
    p = epoch / n_epochs
    if   p < 0.50: return 0.001
    elif p < 0.70: return 0.005
    elif p < 0.85: return 0.01
    else:          return 0.05


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, cfg, lam, epoch=0):
    model.train()
    criterion.lambda_physics = lam
    total_loss = total_cond = total_conv = total_rad = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        if hasattr(batch, "dgl_graph") and batch.dgl_graph is not None:
            dgl_graph = batch.dgl_graph
            if isinstance(dgl_graph, list):
                dgl_graph = dgl_graph[0]
            batch.dgl_graph = dgl_graph.to(device)
        optimizer.zero_grad()

        delta_T_pred = model(batch)
        Y_std = (float(batch.Y_std[0])
                 if hasattr(batch.Y_std, "__len__") else float(batch.Y_std))

        dT_std  = (float(batch.dT_std[0])  if hasattr(batch.dT_std,  "__len__") else float(batch.dT_std))
        dT_mean = (float(batch.dT_mean[0]) if hasattr(batch.dT_mean, "__len__") else float(batch.dT_mean))
        loss, breakdown = criterion(
            delta_T_pred=delta_T_pred, target=batch.y,
            batch=batch, Y_std=Y_std, dt=cfg.dt,
            dT_std=dT_std, dT_mean=dT_mean,
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
    return {"loss": total_loss/n, "cond": total_cond/n,
            "conv": total_conv/n, "rad":  total_rad/n}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (one-step, training window only)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_one_step(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        if hasattr(batch, "dgl_graph") and batch.dgl_graph is not None:
            dgl_graph = batch.dgl_graph
            if isinstance(dgl_graph, list):
                dgl_graph = dgl_graph[0]
            batch.dgl_graph = dgl_graph.to(device)
        delta_T_pred = model(batch)
        Y_std = (float(batch.Y_std[0])
                 if hasattr(batch.Y_std, "__len__") else float(batch.Y_std))

        loss, _ = criterion(
            delta_T_pred=delta_T_pred, target=batch.y,
            batch=batch, Y_std=Y_std, dt=cfg.dt,
        )
        total_loss += loss.item()
        n_batches  += 1

        T_pred_K = (batch.T_current.cpu() + _denorm_dT(delta_T_pred, batch)).numpy().ravel()
        T_true_K = batch.T_next.cpu().numpy().ravel()
        all_pred.append(T_pred_K)
        all_true.append(T_true_K)

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    m = compute_metrics(y_pred, y_true)
    m["loss"]       = total_loss / max(n_batches, 1)
    m["within_5K"]  = within_tolerance(y_pred, y_true,  5.0)
    m["within_10K"] = within_tolerance(y_pred, y_true, 10.0)
    m["within_20K"] = within_tolerance(y_pred, y_true, 20.0)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Option A verification rollout (3200 → 4000s)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_verification_rollout(model, cfg, device, save_dir):
    """
    Roll out all test simulations for 400 steps.
    Report Phase 1 (0–3200s, seen) and Phase 2 (3200–4000s, UNSEEN) separately.
    Phase 2 is the key thesis result.
    """
    model.eval()
    eval_ds  = get_evaluation_dataset(cfg)
    n_train  = cfg.n_train_steps   # 320
    n_total  = cfg.n_total_steps   # 400
    n_verify = cfg.n_verify_steps  # 80

    print(f"\n{'='*72}")
    print(f"  OPTION A — VERIFIED FUTURE PREDICTION")
    print(f"{'='*72}")
    print(f"  Training window    : t=0–{cfg.train_time_end:.0f}s "
          f"({n_train} steps) — model SAW this")
    print(f"  Verification window: t={cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s "
          f"({n_verify} steps) — model NEVER saw this")
    print(f"  Ground truth       : available for both windows\n")

    all_p1_mae, all_p1_r2 = [], []
    all_p2_mae, all_p2_r2 = [], []
    per_sim = {}

    print(f"  {'Sim':>4}  {'P1 MAE[K]':>10}  {'P1 R²':>8}  "
          f"{'P2 MAE[K]':>10}  {'P2 R²':>8}")
    print(f"  {'-'*55}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for sim_i in eval_ds.sim_indices:
        T_pred, T_true = rollout_from_dataset(
            model, eval_ds, sim_i,
            start_t=20, n_steps=n_total-20, device=str(device),
        )

        m1 = compute_metrics(T_pred[:n_train+1].ravel(), T_true[:n_train+1].ravel())
        m2 = compute_metrics(T_pred[n_train:].ravel(),  T_true[n_train:].ravel())
        m1["within_5K"]  = within_tolerance(T_pred[:n_train+1].ravel(),
                                             T_true[:n_train+1].ravel(), 5.0)
        m1["within_10K"] = within_tolerance(T_pred[:n_train+1].ravel(),
                                             T_true[:n_train+1].ravel(), 10.0)
        m2["within_5K"]  = within_tolerance(T_pred[n_train:].ravel(),
                                             T_true[n_train:].ravel(), 5.0)
        m2["within_10K"] = within_tolerance(T_pred[n_train:].ravel(),
                                             T_true[n_train:].ravel(), 10.0)

        all_p1_mae.append(m1["mae"]); all_p1_r2.append(m1["r2"])
        all_p2_mae.append(m2["mae"]); all_p2_r2.append(m2["r2"])

        print(f"  {sim_i:>4}  {m1['mae']:>10.2f}  {m1['r2']:>8.4f}  "
              f"{m2['mae']:>10.2f}  {m2['r2']:>8.4f}")

        step_m   = metrics_per_timestep(T_pred[:len(T_true)], T_true)
        step_mae = [s["mae"] for s in step_m]

        per_sim[f"sim_{sim_i:03d}"] = {
            "phase1": m1, "phase2": m2, "step_mae": step_mae
        }
        np.save(f"{save_dir}/T_pred_sim{sim_i:03d}.npy", T_pred)
        np.save(f"{save_dir}/T_true_sim{sim_i:03d}.npy", T_true)

    print(f"\n{'='*72}")
    print(f"  AGGREGATE  ({len(eval_ds.sim_indices)} test simulations)")
    print(f"{'='*72}")
    print(f"\n  Phase 1  t=0–{cfg.train_time_end:.0f}s  (training window, model SAW this):")
    print(f"    MAE = {np.mean(all_p1_mae):.2f} ± {np.std(all_p1_mae):.2f} K")
    print(f"    R²  = {np.mean(all_p1_r2):.4f}")
    print(f"\n  Phase 2  t={cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s  "
          f"(VERIFICATION — model NEVER saw this):")
    print(f"    MAE = {np.mean(all_p2_mae):.2f} ± {np.std(all_p2_mae):.2f} K")
    print(f"    R²  = {np.mean(all_p2_r2):.4f}")
    print(f"\n  *** KEY THESIS RESULT ***")
    print(f"  Phase 2 MAE = {np.mean(all_p2_mae):.2f} K  |  R² = {np.mean(all_p2_r2):.4f}")

    summary = {
        "description": (
            f"Option A: trained t=0–{cfg.train_time_end:.0f}s, "
            f"verified t={cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s"
        ),
        "phase1": {
            "t_range": f"0–{cfg.train_time_end:.0f}s",
            "model_seen": True,
            "mean_mae": float(np.mean(all_p1_mae)),
            "std_mae":  float(np.std(all_p1_mae)),
            "mean_r2":  float(np.mean(all_p1_r2)),
        },
        "phase2": {
            "t_range": f"{cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s",
            "model_seen": False,
            "ground_truth_available": True,
            "mean_mae": float(np.mean(all_p2_mae)),
            "std_mae":  float(np.std(all_p2_mae)),
            "mean_r2":  float(np.mean(all_p2_r2)),
        },
        "per_simulation": per_sim,
    }

    out = Path(save_dir) / "verification_summary.json"
    with open(str(out), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Print split summary
# ─────────────────────────────────────────────────────────────────────────────

def print_split_summary(cfg, train_loader, val_loader, test_loader):
    n_tr = len(train_loader.dataset)
    n_va = len(val_loader.dataset)
    n_te = len(test_loader.dataset)
    total = n_tr + n_va + n_te
    n_sims = 50
    n_test  = max(1, int(n_sims * cfg.test_fraction))
    n_val   = max(1, int(n_sims * cfg.val_fraction))
    n_train = n_sims - n_val - n_test

    print(f"\n{'='*72}")
    print(f"  OPTION A TEMPORAL SPLIT")
    print(f"{'='*72}")
    print(f"  Full simulation: t = 0 – {cfg.t_total:.0f}s  "
          f"({cfg.n_total_steps} steps, dt={cfg.dt:.0f}s)")
    print(f"  Training window: t = 0 – {cfg.train_time_end:.0f}s  "
          f"({cfg.n_train_steps} steps, 80%)")
    print(f"  Verify  window: t = {cfg.train_time_end:.0f} – {cfg.predict_time_end:.0f}s  "
          f"({cfg.n_verify_steps} steps, 20%, never seen during training)")
    print(f"\n  {'Split':<6} {'Sims':>5}  {'(sim,t) pairs':>14}  {'Timestep range'}")
    print(f"  {'-'*55}")
    print(f"  {'TRAIN':<6} {n_train:>5}  {n_tr:>14,}  t = 0–{cfg.train_time_end:.0f}s")
    print(f"  {'VAL':<6} {n_val:>5}  {n_va:>14,}  t = 0–{cfg.train_time_end:.0f}s")
    print(f"  {'TEST':<6} {n_test:>5}  {n_te:>14,}  t = 0–{cfg.train_time_end:.0f}s")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<6} {n_sims:>5}  {total:>14,}")
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: BaseConfig = CONFIG) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*72}")
    print(f"  PHYSICS-INFORMED HEAT TREATMENT GNN — OPTION A TEMPORAL SPLIT")
    print(f"{'='*72}")
    print(f"  Dataset   : {cfg.dataset_path}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {cfg.n_epochs}")
    print(f"  LR        : {cfg.learning_rate}")
    print(f"  Batch     : {cfg.batch_size}")
    print(f"\n  TEMPORAL SPLIT (Option A):")
    print(f"    Training  : t = 0 – {cfg.train_time_end:.0f}s  "
          f"(steps 0–{cfg.n_train_steps-1}, 80%)")
    print(f"    Verify    : t = {cfg.train_time_end:.0f} – {cfg.predict_time_end:.0f}s  "
          f"(steps {cfg.n_train_steps}–{cfg.n_total_steps-1}, 20%, UNSEEN)")
    print(f"\n  Physics equations:")
    print(f"    Conduction : rho*Cp*dT/dt = kappa*laplacian(T)  [Fourier]")
    print(f"    Convection : T_steel ≤ T_set                     [Newton]")
    print(f"    Radiation  : dT/dt ~ epsilon*sigma*(T_set^4-T^4) [Stefan-Boltzmann]")

    logger = setup_logging(cfg)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(cfg, rollout_steps=1)
    print_split_summary(cfg, train_loader, val_loader, test_loader)

    # ── Model ─────────────────────────────────────────────────────────
    model     = HeatTreatmentGNN(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg)
    criterion = PhysicsInformedLoss(
        lambda_physics = 0.001,
        w_cond         = cfg.w_conduction,
        w_conv         = cfg.w_convection,
        w_rad          = cfg.w_radiation,
        epsilon_steel  = cfg.epsilon_steel,
        char_thickness = cfg.char_thickness,
    )
    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)

    history: dict = {
        "train_loss": [], "val_loss":     [], "val_mae":      [],
        "val_rmse":   [], "val_r2":       [], "val_within_5K": [],
        "val_within_10K": [], "loss_cond": [], "loss_conv":   [],
        "loss_rad":   [], "lambda":       [],
    }

    # ── Training loop ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  TRAINING PROGRESS  (one-step on t=0–{cfg.train_time_end:.0f}s only)")
    print(f"{'='*100}")
    print(
        f"  {'Ep':>5} | {'TrLoss':>9} | {'VaLoss':>9} | "
        f"{'MAE[K]':>7} | {'R2':>7} | "
        f"{'W5K':>6} | {'W10K':>6} | "
        f"{'Cond':>9} | {'Conv':>9} | {'Rad':>9} | {'λ':>7}"
    )
    print(f"  {'-'*98}")

    for epoch in range(1, cfg.n_epochs + 1):
        lam = get_lambda(epoch, cfg.n_epochs)

        train_out   = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, lam
        )
        val_metrics = evaluate_one_step(model, val_loader, criterion, device, cfg)
        scheduler.step(val_metrics["loss"])

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

        is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_metrics)

        if epoch % cfg.save_every_n_epochs == 0:
            ckpt_mgr.save_periodic(model, optimizer, scheduler, epoch, val_metrics)

        if epoch % cfg.log_every_n_epochs == 0 or epoch == 1 or is_best:
            print(
                f"  {epoch:>5} | {train_out['loss']:>9.5f} | "
                f"{val_metrics['loss']:>9.5f} | "
                f"{val_metrics['mae']:>7.2f} | {val_metrics['r2']:>7.4f} | "
                f"{val_metrics['within_5K']:>6.1f} | {val_metrics['within_10K']:>6.1f} | "
                f"{train_out['cond']:>9.5f} | "
                f"{train_out['conv']:>9.5f} | "
                f"{train_out['rad']:>9.5f} | "
                f"{lam:>7.4f}"
                + ("  ◄ BEST" if is_best else "")
            )
            log_metrics(logger, epoch, train_out["loss"], val_metrics, cfg)

    # ── Load best model ────────────────────────────────────────────────
    model = HeatTreatmentGNN.load(ckpt_mgr.best_path, cfg, str(device))

    # ── One-step test accuracy on training window ──────────────────────
    print(f"\n{'='*72}")
    print(f"  ONE-STEP TEST ACCURACY  (t=0–{cfg.train_time_end:.0f}s)")
    print(f"{'='*72}")
    test_onestep = evaluate_one_step(model, test_loader, criterion, device, cfg)
    print(f"  MAE        = {test_onestep['mae']:.2f} K")
    print(f"  RMSE       = {test_onestep['rmse']:.2f} K")
    print(f"  R²         = {test_onestep['r2']:.4f}")
    print(f"  Within  5K = {test_onestep['within_5K']:.1f}%")
    print(f"  Within 10K = {test_onestep['within_10K']:.1f}%")

    # ── Option A verification rollout ──────────────────────────────────
    verify_dir   = str(Path(cfg.output_dir) / "verification")
    verification = run_verification_rollout(model, cfg, device, verify_dir)

    # ── Save history ───────────────────────────────────────────────────
    hist_path = Path(cfg.log_dir) / "training_history.json"
    with open(str(hist_path), "w") as f:
        json.dump({
            "temporal_split": {
                "strategy":        "Option A — 80% train / 20% verify",
                "train_window":    f"t=0–{cfg.train_time_end:.0f}s",
                "verify_window":   f"t={cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s",
                "n_train_steps":   cfg.n_train_steps,
                "n_verify_steps":  cfg.n_verify_steps,
            },
            "config": {
                "n_epochs":      cfg.n_epochs,
                "learning_rate": cfg.learning_rate,
                "batch_size":    cfg.batch_size,
                "hidden":        cfg.hidden_features,
                "layers":        cfg.n_message_passing_layers,
            },
            "best_val_mae":  ckpt_mgr.best_mae,
            "best_epoch":    ckpt_mgr.best_epoch,
            "test_onestep":  test_onestep,
            "verification":  verification,
            "history":       history,
        }, f, indent=2)

    print(f"\n  History    → {hist_path}")
    print(f"  Best model → {ckpt_mgr.best_path}")
    print(f"\n  Best val MAE = {ckpt_mgr.best_mae:.2f} K  (epoch {ckpt_mgr.best_epoch})")
    p2 = verification["phase2"]
    print(f"\n  KEY THESIS RESULT:")
    print(f"  Phase 2 MAE (t={cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s, UNSEEN) "
          f"= {p2['mean_mae']:.2f} K  |  R² = {p2['mean_r2']:.4f}\n")


if __name__ == "__main__":
    main()
