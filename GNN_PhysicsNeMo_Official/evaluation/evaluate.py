"""
Evaluation script — runs full rollout on test set and reports per-timestep error.

Usage:
    python evaluation/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
    python evaluation/evaluate.py  # auto-loads best_model.pt
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from configs.base_config import BaseConfig, CONFIG
from data.dataset import HeatTreatmentDataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout import rollout_from_dataset
from utils.metrics import compute_metrics, metrics_per_timestep


def evaluate_all_test_sims(
    model: HeatTreatmentGNN,
    dataset: HeatTreatmentDataset,
    cfg: BaseConfig,
    device: str,
    extra_steps: int = 0,
) -> dict:
    """
    Roll out for every test simulation and collect metrics.

    Args:
        extra_steps: Additional steps beyond dataset window (extrapolation test).
                     These steps won't have ground truth → reported separately.
    """
    all_mae, all_rmse, all_r2 = [], [], []
    per_step_mae = []

    for sim_i in dataset.sim_indices:
        T_pred, T_true = rollout_from_dataset(
            model, dataset, sim_i,
            start_t=0,
            n_steps=dataset._simulations[sim_i]["n_times"] - 1 + extra_steps,
            device=device,
        )

        metrics = compute_metrics(T_pred.ravel(), T_true.ravel())
        all_mae.append(metrics["mae"])
        all_rmse.append(metrics["rmse"])
        all_r2.append(metrics["r2"])

        step_metrics = metrics_per_timestep(T_pred, T_true)
        per_step_mae.append([m["mae"] for m in step_metrics])

        sim_name = f"sim_{sim_i:03d}"
        print(f"  {sim_name:12s}  MAE={metrics['mae']:7.2f} K  "
              f"RMSE={metrics['rmse']:7.2f} K  R2={metrics['r2']:.4f}")

    summary = {
        "mean_mae":  float(np.mean(all_mae)),
        "mean_rmse": float(np.mean(all_rmse)),
        "mean_r2":   float(np.mean(all_r2)),
        "std_mae":   float(np.std(all_mae)),
    }

    print(f"\n  Summary: MAE = {summary['mean_mae']:.2f} ± {summary['std_mae']:.2f} K  "
          f"| RMSE = {summary['mean_rmse']:.2f} K  | R2 = {summary['mean_r2']:.4f}")

    # Plot mean MAE per time step
    if per_step_mae:
        min_len = min(len(x) for x in per_step_mae)
        arr = np.array([x[:min_len] for x in per_step_mae])
        mean_step = arr.mean(axis=0)
        times = np.arange(min_len) * cfg.dt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, mean_step, color="steelblue", lw=1.5)
        ax.fill_between(times, arr.min(axis=0), arr.max(axis=0),
                        alpha=0.2, color="steelblue", label="Min-Max range")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MAE [K]")
        ax.set_title("Mean absolute error per time step (test set)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = f"{cfg.output_dir}/plots/mae_per_timestep.png"
        fig.savefig(plot_path, dpi=150)
        print(f"  Plot saved → {plot_path}")
        plt.close(fig)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--extra_steps", type=int, default=0,
                        help="Extra rollout steps beyond dataset window")
    args = parser.parse_args()

    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model = HeatTreatmentGNN.load(ckpt_path, cfg, device)

    test_ds = HeatTreatmentDataset(cfg.dataset_path, cfg, "test", rollout_steps=1)

    print(f"\n{'='*60}")
    print("TEST SET EVALUATION — FULL ROLLOUT")
    print(f"{'='*60}")

    summary = evaluate_all_test_sims(
        model, test_ds, cfg, device, extra_steps=args.extra_steps
    )

    out_path = f"{cfg.output_dir}/logs/test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()