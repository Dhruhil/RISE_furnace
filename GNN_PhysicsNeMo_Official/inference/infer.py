"""
Single-simulation inference with Option A verification.

BUGS FIXED vs old version:
  1. HeatTreatmentDataset loaded without split_mode="evaluation"
     → rollout capped at 320 steps (training window only)
     → target_time=3500 or 4000 would silently return wrong results
  2. predict_at_time imported from inference.infer in rollout_infer.py
     — moved predict_at_time to models/rollout.py and imported from there.
     This file keeps a local copy for backward compatibility.

Usage:
    python infer.py --sim_idx 0 --target_time 3500   # in verification window
    python infer.py --sim_idx 0 --target_time 2000   # in training window
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.base_config import BaseConfig, CONFIG
from data.dataset import HeatTreatmentDataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout import rollout_from_dataset, predict_at_time
from utils.metrics import compute_metrics, metrics_per_timestep
from evaluation.visualise import (
    plot_parity,
    plot_temperature_field,
    plot_time_series_at_cells,
)


def run_inference(
    cfg:             BaseConfig,
    checkpoint_path: str,
    sim_idx:         int,
    target_time:     float,
    device:          str,
    save_dir:        str,
) -> dict:
    """Full inference pipeline for one simulation."""

    print(f"\n{'='*65}")
    print(f"  INFERENCE — Option A")
    print(f"  Sim index   : {sim_idx}")
    print(f"  Target time : {target_time:.0f} s")
    print(f"  Train end   : {cfg.train_time_end:.0f} s")
    if target_time > cfg.train_time_end:
        print(f"  Window      : VERIFICATION (unseen during training)")
    else:
        print(f"  Window      : training window (model saw this)")
    print(f"{'='*65}\n")

    model = HeatTreatmentGNN.load(checkpoint_path, cfg, device)

    # FIX: use split_mode="evaluation" to get all 400 steps
    dataset = HeatTreatmentDataset(
        cfg.dataset_path, cfg, split="test",
        rollout_steps=1, split_mode="evaluation"
    )

    sim    = dataset._simulations[sim_idx]
    n_times = sim["n_times"]
    n_cells = sim["n_cells"]
    coords  = sim["coords"]

    # Steps needed to reach target_time
    n_steps = int(round(target_time / cfg.dt))
    n_steps = max(n_steps, n_times - 1)   # at least full simulation

    print(f"  Rolling out {n_steps} steps (t=0 → {n_steps*cfg.dt:.0f}s)...")
    T_pred, T_true = rollout_from_dataset(
        model, dataset, sim_idx,
        start_t=0, n_steps=n_steps, device=device,
    )

    n_gt     = T_true.shape[0]
    times_gt = np.arange(n_gt) * cfg.dt

    # Overall metrics
    metrics  = compute_metrics(T_pred[:n_gt].ravel(), T_true.ravel())
    step_m   = metrics_per_timestep(T_pred[:n_gt], T_true)
    step_mae = np.array([m["mae"]  for m in step_m])
    step_rmse= np.array([m["rmse"] for m in step_m])

    print(f"\n  Full-window metrics (0–{times_gt[-1]:.0f}s):")
    print(f"    MAE  = {metrics['mae']:.2f} K")
    print(f"    RMSE = {metrics['rmse']:.2f} K")
    print(f"    R²   = {metrics['r2']:.4f}")

    # Metrics at target_time
    T_at_t, step_used = predict_at_time(T_pred, target_time, cfg.dt)
    in_window         = step_used < n_gt

    print(f"\n  At t={target_time:.0f}s (step {step_used}):")
    if in_window:
        T_ref  = T_true[step_used]
        m_at_t = compute_metrics(T_at_t, T_ref)
        print(f"    MAE  = {m_at_t['mae']:.2f} K")
        print(f"    RMSE = {m_at_t['rmse']:.2f} K")
        print(f"    R²   = {m_at_t['r2']:.4f}")
        window_label = (
            "VERIFICATION (unseen)" if target_time > cfg.train_time_end
            else "training window"
        )
        print(f"    Window: {window_label}")
    else:
        print(f"    [No ground truth at this time step]")

    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_dir}/T_pred_sim{sim_idx:03d}.npy", T_pred)
    np.save(f"{save_dir}/T_true_sim{sim_idx:03d}.npy", T_true)
    np.save(f"{save_dir}/coords_sim{sim_idx:03d}.npy",  coords)

    results = {
        "sim_idx":       sim_idx,
        "n_cells":       n_cells,
        "target_time":   target_time,
        "step_used":     step_used,
        "in_gt_window":  in_window,
        "overall":       metrics,
        "at_target_time": (m_at_t if in_window else None),
        "step_mae":      step_mae.tolist(),
        "step_rmse":     step_rmse.tolist(),
    }
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(times_gt, step_mae,  color="steelblue", lw=1.5, label="MAE [K]")
    ax.plot(times_gt, step_rmse, color="tomato",    lw=1.5, ls="--", label="RMSE [K]")
    ax.axvline(cfg.train_time_end, color="orange", ls="--", lw=2,
               label=f"Train end ({cfg.train_time_end:.0f}s)")
    if target_time <= times_gt[-1]:
        ax.axvline(target_time, color="green", ls=":", lw=2,
                   label=f"Target t={target_time:.0f}s")
    ax.axvspan(cfg.train_time_end, times_gt[-1],
               alpha=0.07, color="red", label="Unseen window")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Error [K]")
    ax.set_title(f"Rollout error — Sim {sim_idx}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/error_per_step.png", dpi=150)
    plt.close(fig)

    plot_parity(T_pred[:n_gt], T_true,
                title=f"Sim {sim_idx}",
                save_path=f"{save_dir}/parity.png")

    if in_window:
        plot_temperature_field(
            coords, T_at_t, T_true[step_used],
            title=f"t={target_time:.0f}s",
            save_path=f"{save_dir}/field_t{int(target_time):05d}.png",
        )

    plot_time_series_at_cells(
        T_pred[:n_gt], T_true, times_gt,
        n_cells_show=5,
        title=f"Time series — Sim {sim_idx}",
        save_path=f"{save_dir}/timeseries.png",
        t_end_train=cfg.train_time_end,
    )

    print(f"\n  Outputs → {save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--sim_idx",     type=int,   default=0)
    parser.add_argument("--target_time", type=float, default=3500.0,
                        help="Target time [s]. Use >3200 to test verification window.")
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    cfg    = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"
    save_dir = (
        f"{cfg.output_dir}/predictions/"
        f"sim{args.sim_idx:03d}_t{int(args.target_time)}s"
    )

    run_inference(
        cfg=cfg, checkpoint_path=ckpt, sim_idx=args.sim_idx,
        target_time=args.target_time, device=device, save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
