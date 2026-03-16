"""
Long-horizon rollout inference — extrapolation beyond training window.

This script demonstrates and verifies that the GNN can predict at
t = 3000 s (or any time) even when trained on 0-4000 s data:

  - The model is AUTOREGRESSIVE: T(t+dt) = T(t) + GNN(graph_at_t)
  - To get T at 3000 s, we simply roll out 300 steps (300 * 10 s = 3000 s)
  - To extrapolate BEYOND 4000 s, we continue rolling out past step 400

Usage:
    python inference/rollout_infer.py --target_times 500 1000 2000 3000 4000
    python inference/rollout_infer.py --target_times 3000 --sim_idx 5
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch

from configs.base_config import BaseConfig, CONFIG
from data.dataset import HeatTreatmentDataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout import rollout_from_dataset
from utils.metrics import compute_metrics
from inference.infer import predict_at_time


def multi_time_prediction(
    cfg: BaseConfig,
    checkpoint_path: str,
    sim_idx: int,
    target_times: list[float],
    device: str,
    save_dir: str,
):
    """
    Predict temperature fields at multiple target times for one simulation.

    This is the key demonstration of the temporal flexibility:
      - target_times can be ANY subset of 0-4000 s (or beyond)
      - All predictions come from a SINGLE rollout of the model

    Returns dict mapping target_time -> {mae, rmse, T_pred, T_true}
    """
    model = HeatTreatmentGNN.load(checkpoint_path, cfg, device)
    dataset = HeatTreatmentDataset(cfg.dataset_path, cfg, split="test", rollout_steps=1)

    sim = dataset._simulations[sim_idx]
    n_times = sim["n_times"]
    coords  = sim["coords"]

    # We need enough rollout steps to cover the max target time
    max_time = max(target_times)
    n_steps_needed = int(np.ceil(max_time / cfg.dt))
    extra = max(0, n_steps_needed - (n_times - 1))

    print(f"\nRollout: sim {sim_idx}, {n_steps_needed} steps → t = {n_steps_needed*cfg.dt:.0f} s")
    print(f"  Dataset covers: 0 - {(n_times-1)*cfg.dt:.0f} s")
    print(f"  Extrapolation steps: {extra}")

    T_pred, T_true = rollout_from_dataset(
        model, dataset, sim_idx,
        start_t=0,
        n_steps=n_steps_needed,
        device=device,
    )
    n_gt = T_true.shape[0]

    # ---- Extract and compare at each target time ----
    results = {}
    summary_rows = []

    for t_target in sorted(target_times):
        T_at_t, step = predict_at_time(T_pred, t_target, cfg.dt)
        in_gt_window = step < n_gt

        row = {
            "target_time": t_target,
            "step":        step,
            "in_gt_window": in_gt_window,
            "T_pred_mean": float(T_at_t.mean()),
            "T_pred_min":  float(T_at_t.min()),
            "T_pred_max":  float(T_at_t.max()),
        }

        if in_gt_window:
            T_ref = T_true[step]
            m = compute_metrics(T_at_t, T_ref)
            row.update({"mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
            row["T_true_mean"] = float(T_ref.mean())
            row["T_true_max"]  = float(T_ref.max())
            print(f"  t={t_target:6.0f}s  step={step:4d}  "
                  f"MAE={m['mae']:6.2f}K  RMSE={m['rmse']:6.2f}K  R2={m['r2']:.4f}  "
                  f"T_mean={T_at_t.mean():.1f}K")
        else:
            print(f"  t={t_target:6.0f}s  step={step:4d}  "
                  f"[EXTRAPOLATION]  T_mean={T_at_t.mean():.1f}K  "
                  f"T_max={T_at_t.max():.1f}K")

        results[t_target] = {"metrics": row, "T_pred": T_at_t}
        summary_rows.append(row)

    # ---- Summary plot ----
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Plot: mean temperature vs time (GNN vs OpenFOAM)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    times_gt = np.arange(n_gt) * cfg.dt
    T_true_mean = T_true.mean(axis=1)
    T_true_max  = T_true.max(axis=1)
    T_pred_mean = T_pred[:n_gt].mean(axis=1)
    T_pred_max  = T_pred[:n_gt].max(axis=1)

    ax = axes[0]
    ax.plot(times_gt, T_true_mean, "k-",  lw=2, label="OpenFOAM mean T")
    ax.plot(times_gt, T_pred_mean, "r--", lw=2, label="GNN mean T")
    ax.plot(times_gt, T_true_max,  "k:",  lw=1, label="OpenFOAM max T")
    ax.plot(times_gt, T_pred_max,  "r:",  lw=1, label="GNN max T")

    # Mark target times
    for t_target in target_times:
        T_at_t, step = predict_at_time(T_pred, t_target, cfg.dt)
        color = "green" if step < n_gt else "purple"
        ax.axvline(t_target, color=color, linestyle="--", alpha=0.5, lw=1)
        ax.annotate(f"{t_target:.0f}s",
                    xy=(t_target, T_pred[:n_gt].mean()),
                    fontsize=7, color=color, rotation=90, va="bottom")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"Temperature evolution — Sim {sim_idx}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot: MAE vs time
    ax2 = axes[1]
    err_mean = np.mean(np.abs(T_pred[:n_gt] - T_true), axis=1)
    ax2.plot(times_gt, err_mean, color="steelblue", lw=1.5)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("MAE [K]")
    ax2.set_title(f"Absolute error over time — Sim {sim_idx}")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = f"{save_dir}/multi_time_sim{sim_idx:03d}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {plot_path}")

    # Save summary
    summary_path = f"{save_dir}/multi_time_summary_sim{sim_idx:03d}.json"
    with open(summary_path, "w") as f:
        json.dump({"sim_idx": sim_idx, "predictions": summary_rows}, f, indent=2)
    print(f"Summary saved → {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-time rollout inference: predict at any target time"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sim_idx", type=int, default=0)
    parser.add_argument("--target_times", nargs="+", type=float,
                        default=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                        help="Target times [s] to predict at (any value, incl. 3000 s)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg    = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"
    save_dir = f"{cfg.output_dir}/predictions/multi_time"

    print(f"\nTarget times: {args.target_times}")
    print("Note: The GNN predicts autoregressively, so ANY target time is possible,")
    print("      including t=3000 s (within training range) and beyond 4000 s.")

    multi_time_prediction(
        cfg=cfg,
        checkpoint_path=ckpt,
        sim_idx=args.sim_idx,
        target_times=args.target_times,
        device=device,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()