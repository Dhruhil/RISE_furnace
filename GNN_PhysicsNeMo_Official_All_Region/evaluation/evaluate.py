"""
Full test-set evaluation with per-simulation rollout metrics.

BUGS FIXED vs old version:
  1. _plot_step_mae() used cfg.t_end which doesn't exist in new config
     → AttributeError crash. Fixed to use cfg.train_time_end
  2. main() used cfg.predict_future_time which was removed from config
     → AttributeError crash. Fixed to use cfg.predict_time_end
  3. dataset loaded without split_mode — rollout was capped at 320 steps
     instead of 400. Fixed to use split_mode="evaluation"

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
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
from data.dataset import HeatTreatmentDataset, get_evaluation_dataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout import rollout_from_dataset, predict_at_time
from utils.metrics import compute_metrics, within_tolerance, metrics_per_timestep


def evaluate_all_test_sims(
    model:   HeatTreatmentGNN,
    cfg:     BaseConfig,
    device:  str,
) -> dict:
    """
    Roll out all test simulations and report Phase 1 and Phase 2 metrics.

    Phase 1: t=0–3200s (training window, model saw this)
    Phase 2: t=3200–4000s (verification window, model NEVER saw this)
    """
    model.eval()

    # FIX: use evaluation dataset with all 400 steps
    dataset = get_evaluation_dataset(cfg)

    n_train = cfg.n_train_steps   # 320
    n_total = cfg.n_total_steps   # 400

    all_p1_mae, all_p1_r2 = [], []
    all_p2_mae, all_p2_r2 = [], []
    per_step_mae_all = []
    per_sim_results  = {}

    print(f"\n{'='*70}")
    print(f"  TEST EVALUATION — Option A rollout (0 → 4000s)")
    print(f"{'='*70}")
    print(f"  {'Sim':>4}  {'P1 MAE':>10}  {'P1 R²':>8}  "
          f"{'P2 MAE':>10}  {'P2 R²':>8}  {'P2=unseen'}")
    print(f"  {'-'*60}")

    save_dir = Path(cfg.output_dir) / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    for sim_i in dataset.sim_indices:
        T_pred, T_true = rollout_from_dataset(
            model, dataset, sim_i,
            start_t=0, n_steps=n_total, device=device,
        )

        # Phase 1: training window
        m1 = compute_metrics(T_pred[:n_train+1].ravel(), T_true[:n_train+1].ravel())
        m1["within_5K"]  = within_tolerance(T_pred[:n_train+1].ravel(),
                                             T_true[:n_train+1].ravel(), 5.0)
        m1["within_10K"] = within_tolerance(T_pred[:n_train+1].ravel(),
                                             T_true[:n_train+1].ravel(), 10.0)

        # Phase 2: verification window (NEVER seen during training)
        m2 = compute_metrics(T_pred[n_train:].ravel(), T_true[n_train:].ravel())
        m2["within_5K"]  = within_tolerance(T_pred[n_train:].ravel(),
                                             T_true[n_train:].ravel(), 5.0)
        m2["within_10K"] = within_tolerance(T_pred[n_train:].ravel(),
                                             T_true[n_train:].ravel(), 10.0)

        all_p1_mae.append(m1["mae"]); all_p1_r2.append(m1["r2"])
        all_p2_mae.append(m2["mae"]); all_p2_r2.append(m2["r2"])

        step_m   = metrics_per_timestep(T_pred[:len(T_true)], T_true)
        step_mae = np.array([s["mae"] for s in step_m])
        per_step_mae_all.append(step_mae)

        print(
            f"  {sim_i:>4}  {m1['mae']:>10.2f}  {m1['r2']:>8.4f}  "
            f"{m2['mae']:>10.2f}  {m2['r2']:>8.4f}"
        )

        per_sim_results[f"sim_{sim_i:03d}"] = {
            "phase1_training":     m1,
            "phase2_verification": m2,
            "step_mae":            step_mae.tolist(),
        }

        np.save(str(save_dir / f"T_pred_sim{sim_i:03d}.npy"), T_pred)
        np.save(str(save_dir / f"T_true_sim{sim_i:03d}.npy"), T_true)

    print(f"\n  {'':>4}  {'MEAN':>10}  {'MEAN':>8}  {'MEAN':>10}  {'MEAN':>8}")
    print(
        f"  {'ALL':>4}  {np.mean(all_p1_mae):>10.2f}  {np.mean(all_p1_r2):>8.4f}  "
        f"{np.mean(all_p2_mae):>10.2f}  {np.mean(all_p2_r2):>8.4f}"
    )

    # FIX: use cfg.train_time_end instead of cfg.t_end
    _plot_step_mae(per_step_mae_all, cfg, str(save_dir))

    summary = {
        "phase1_training_window": {
            "t_range":  f"0–{cfg.train_time_end:.0f}s",
            "mean_mae": float(np.mean(all_p1_mae)),
            "mean_r2":  float(np.mean(all_p1_r2)),
            "model_seen_this": True,
        },
        "phase2_verification_window": {
            "t_range":  f"{cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s",
            "mean_mae": float(np.mean(all_p2_mae)),
            "mean_r2":  float(np.mean(all_p2_r2)),
            "model_seen_this": False,
            "ground_truth_available": True,
        },
        "per_simulation": per_sim_results,
    }

    with open(str(save_dir / "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved → {save_dir}")
    return summary


def _plot_step_mae(per_step_mae_list: list, cfg: BaseConfig, save_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.colormaps["tab10"]
    for i, step_mae in enumerate(per_step_mae_list):
        times = np.arange(len(step_mae)) * cfg.dt
        ax.plot(times, step_mae, color=cmap(i), lw=1.5, label=f"Sim {i}")
    # FIX: use cfg.train_time_end — cfg.t_end doesn't exist
    ax.axvline(cfg.train_time_end, color="black", ls="--", lw=2,
               label=f"Training end ({cfg.train_time_end:.0f}s)")
    ax.axvspan(cfg.train_time_end, cfg.predict_time_end,
               alpha=0.08, color="red", label="Verification window (unseen)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("MAE [K]")
    ax.set_title("Per-step MAE — all test simulations")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/per_step_mae.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg    = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # FIX: use cfg.checkpoint_dir, not cfg.predict_future_time
    ckpt   = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"

    model   = HeatTreatmentGNN.load(ckpt, cfg, device)
    summary = evaluate_all_test_sims(model, cfg, device)

    p1 = summary["phase1_training_window"]
    p2 = summary["phase2_verification_window"]
    print(f"\n  Phase 1 ({p1['t_range']}, training window):")
    print(f"    MAE = {p1['mean_mae']:.2f} K  R² = {p1['mean_r2']:.4f}")
    print(f"\n  Phase 2 ({p2['t_range']}, UNSEEN verification):")
    print(f"    MAE = {p2['mean_mae']:.2f} K  R² = {p2['mean_r2']:.4f}")


if __name__ == "__main__":
    main()
