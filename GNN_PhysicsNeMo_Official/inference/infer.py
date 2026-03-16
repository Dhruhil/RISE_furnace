"""
Single-case inference script.

Loads a trained MeshGraphNet checkpoint and runs autoregressive rollout
for a specified simulation, including extrapolation BEYOND the training window.

Key capability:
  - Dataset covers 0-4000 s
  - You can predict at ANY time, e.g. t=3000 s, by simply rolling out to that step
  - You can also predict BEYOND 4000 s (extrapolation)

Usage:
    python inference/infer.py --checkpoint outputs/checkpoints/best_model.pt \
                               --sim_idx 0 \
                               --target_time 3000 \
                               --extra_steps 20

    # extra_steps: rollout N steps beyond dataset end (4000+N*dt seconds)
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
from utils.metrics import compute_metrics, metrics_per_timestep
from evaluation.visualise import (
    plot_parity,
    plot_temperature_field,
    plot_time_series_at_cells,
)


def predict_at_time(
    T_rollout: np.ndarray,   # (n_steps+1, n_cells)
    target_time: float,       # seconds
    dt: float,                # time step in seconds
    start_time: float = 0.0,
) -> np.ndarray:
    """
    Extract the predicted temperature field at a specific time.

    Since the GNN is autoregressive (predicts ΔT per step), we simply
    take the rollout snapshot at the corresponding step index.
    This works for ANY target_time — including times within the training
    window (e.g. t=3000 s) and beyond it (extrapolation).

    Args:
        T_rollout:    Full rollout array (n_steps+1, n_cells)
        target_time:  Target time in seconds
        dt:           Simulation time step
        start_time:   Start time of rollout

    Returns:
        T_at_time: (n_cells,) temperature field at target_time
    """
    step_idx = int(round((target_time - start_time) / dt))
    step_idx = max(0, min(step_idx, T_rollout.shape[0] - 1))
    return T_rollout[step_idx], step_idx


def run_inference(
    cfg: BaseConfig,
    checkpoint_path: str,
    sim_idx: int,
    target_time: float | None,
    extra_steps: int,
    device: str,
    save_dir: str,
):
    """Full inference pipeline for one simulation."""

    print(f"\n{'='*60}")
    print(f"Inference: sim_idx={sim_idx}, target_time={target_time} s")
    print(f"Dataset: {cfg.dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    # ---- Load model ----
    model = HeatTreatmentGNN.load(checkpoint_path, cfg, device)

    # ---- Load dataset (test split) ----
    # We use the full dataset here so we can pick any sim_idx
    dataset = HeatTreatmentDataset(cfg.dataset_path, cfg, split="test", rollout_steps=1)

    sim = dataset._simulations[sim_idx]
    coords = sim["coords"]            # (n_cells, 3)
    n_times = sim["n_times"]
    n_cells = sim["n_cells"]

    # Number of rollout steps: dataset steps + extra
    n_steps_dataset = n_times - 1
    n_steps_total   = n_steps_dataset + extra_steps

    print(f"Simulation info:")
    print(f"  n_cells:       {n_cells}")
    print(f"  n_times:       {n_times}")
    print(f"  time range:    0 - {n_times * cfg.dt:.0f} s")
    print(f"  extra rollout: {extra_steps} steps → up to {(n_times + extra_steps) * cfg.dt:.0f} s")

    # ---- Run rollout ----
    print("\nRunning autoregressive rollout...")
    T_pred, T_true = rollout_from_dataset(
        model, dataset, sim_idx,
        start_t=0,
        n_steps=n_steps_total,
        device=device,
    )
    # T_pred: (n_steps_gt+1, n_cells) — only up to ground truth length
    # T_true: same shape

    n_steps_gt = T_true.shape[0]
    times_gt   = np.arange(n_steps_gt) * cfg.dt

    # ---- Metrics over ground-truth window ----
    metrics = compute_metrics(T_pred.ravel(), T_true.ravel())
    step_metrics = metrics_per_timestep(T_pred, T_true)
    step_mae  = np.array([m["mae"]  for m in step_metrics])
    step_rmse = np.array([m["rmse"] for m in step_metrics])

    print(f"\nOverall metrics (ground-truth window):")
    print(f"  MAE:      {metrics['mae']:.2f} K")
    print(f"  RMSE:     {metrics['rmse']:.2f} K")
    print(f"  Max Err:  {metrics['max_err']:.2f} K")
    print(f"  R2:       {metrics['r2']:.4f}")

    # ---- Predict at specific target time ----
    if target_time is not None:
        T_at_t, step_used = predict_at_time(T_pred, target_time, cfg.dt)
        print(f"\nPrediction at t = {target_time:.0f} s  (step {step_used}):")
        if step_used < n_steps_gt:
            T_true_at_t = T_true[step_used]
            m_t = compute_metrics(T_at_t, T_true_at_t)
            print(f"  MAE at t={target_time:.0f}s:  {m_t['mae']:.2f} K")
            print(f"  RMSE at t={target_time:.0f}s: {m_t['rmse']:.2f} K")
            print(f"  R2 at t={target_time:.0f}s:   {m_t['r2']:.4f}")
            print(f"  T range predicted:  [{T_at_t.min():.1f}, {T_at_t.max():.1f}] K")
            print(f"  T range true:       [{T_true_at_t.min():.1f}, {T_true_at_t.max():.1f}] K")
        else:
            print(f"  [EXTRAPOLATION - no ground truth]")
            print(f"  T range predicted:  [{T_at_t.min():.1f}, {T_at_t.max():.1f}] K")

    # ---- Save outputs ----
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save predicted arrays
    np.save(f"{save_dir}/T_pred_sim{sim_idx:03d}.npy", T_pred)
    np.save(f"{save_dir}/T_true_sim{sim_idx:03d}.npy", T_true)
    np.save(f"{save_dir}/coords_sim{sim_idx:03d}.npy", coords)

    # Save metrics
    results = {
        "sim_idx":        sim_idx,
        "n_cells":        n_cells,
        "n_times":        n_times,
        "extra_steps":    extra_steps,
        "target_time":    target_time,
        "overall_metrics": metrics,
        "step_mae":       step_mae.tolist(),
        "step_rmse":      step_rmse.tolist(),
    }
    with open(f"{save_dir}/metrics_sim{sim_idx:03d}.json", "w") as fout:
        json.dump(results, fout, indent=2)

    # ---- Plots ----
    print("\nGenerating plots...")

    # 1. MAE per time step
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times_gt, step_mae, color="steelblue", lw=1.5, label="MAE")
    ax.plot(times_gt, step_rmse, color="tomato", lw=1.5, linestyle="--", label="RMSE")
    if target_time is not None and target_time <= times_gt[-1]:
        ax.axvline(target_time, color="green", linestyle=":", lw=2,
                   label=f"Target t={target_time:.0f}s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [K]")
    ax.set_title(f"Rollout error per time step — Sim {sim_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/error_per_step_sim{sim_idx:03d}.png", dpi=150)
    plt.close(fig)

    # 2. Parity plot (all time steps)
    plot_parity(T_pred, T_true,
                title=f"Sim {sim_idx} — Full Rollout",
                save_path=f"{save_dir}/parity_sim{sim_idx:03d}.png")

    # 3. Temperature field at target time
    if target_time is not None:
        T_at_t, step_used = predict_at_time(T_pred, target_time, cfg.dt)
        T_ref = T_true[step_used] if step_used < n_steps_gt else None
        plot_temperature_field(
            coords, T_at_t, T_ref,
            title=f"t={target_time:.0f}s",
            save_path=f"{save_dir}/field_t{int(target_time):05d}_sim{sim_idx:03d}.png",
            projection="yz",
        )

    # 4. Time series at 5 random cells
    plot_time_series_at_cells(
        T_pred, T_true, times_gt,
        n_cells_show=5,
        title=f"Time series — Sim {sim_idx}",
        save_path=f"{save_dir}/timeseries_sim{sim_idx:03d}.png",
    )

    print(f"\nAll outputs saved to: {save_dir}")
    print("Done!")

    return results


def main():
    parser = argparse.ArgumentParser(description="GNN inference for heat treatment")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--sim_idx", type=int, default=0,
                        help="Simulation index to predict")
    parser.add_argument("--target_time", type=float, default=3000.0,
                        help="Target prediction time [s] (can be any time, e.g. 3000 s)")
    parser.add_argument("--extra_steps", type=int, default=0,
                        help="Extra rollout steps beyond dataset end for extrapolation")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (auto-detected if not set)")
    args = parser.parse_args()

    cfg = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"
    save_dir = f"{cfg.output_dir}/predictions/sim{args.sim_idx:03d}"

    run_inference(
        cfg=cfg,
        checkpoint_path=ckpt,
        sim_idx=args.sim_idx,
        target_time=args.target_time,
        extra_steps=args.extra_steps,
        device=device,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()