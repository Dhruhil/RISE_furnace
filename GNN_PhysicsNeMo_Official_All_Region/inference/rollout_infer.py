"""
Multi-time rollout inference — Option A verification window.

BUGS FIXED vs old version:
  1. HeatTreatmentDataset loaded without split_mode="evaluation"
     → _graphs only had test sims, rollout capped at 320 steps
  2. predict_at_time imported from inference.infer — circular import risk.
     Now imported from models.rollout where it belongs.
  3. Reports Phase 1 (training) and Phase 2 (verification) separately.

Usage:
    python inference/rollout_infer.py --sim_idx 0
    python inference/rollout_infer.py --target_times 500 1000 2000 3000 3500 4000
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
from utils.metrics import compute_metrics, within_tolerance


def multi_time_prediction(
    cfg:             BaseConfig,
    checkpoint_path: str,
    sim_idx:         int,
    target_times:    list[float],
    device:          str,
    save_dir:        str,
) -> dict:
    """
    Run one rollout, extract predictions at multiple target times.
    Clearly labels each result as training window or verification window.
    """
    print(f"\n{'='*68}")
    print(f"  MULTI-TIME ROLLOUT — Sim {sim_idx}")
    print(f"  Training window  : t = 0 – {cfg.train_time_end:.0f}s (model SAW this)")
    print(f"  Verification     : t = {cfg.train_time_end:.0f} – {cfg.predict_time_end:.0f}s (UNSEEN)")
    print(f"{'='*68}")

    model = HeatTreatmentGNN.load(checkpoint_path, cfg, device)

    # FIX: use split_mode="evaluation" so rollout covers 0–4000s
    dataset = HeatTreatmentDataset(
        cfg.dataset_path, cfg, split="test",
        rollout_steps=1, split_mode="evaluation"
    )

    sim     = dataset._simulations[sim_idx]
    n_times = sim["n_times"]

    # Roll out to cover all target times
    max_time      = max(target_times)
    n_steps_total = max(int(np.ceil(max_time / cfg.dt)), n_times - 1)

    print(f"\n  Rolling out {n_steps_total} steps → t={n_steps_total*cfg.dt:.0f}s...")
    T_pred, T_true = rollout_from_dataset(
        model, dataset, sim_idx,
        start_t=0, n_steps=n_steps_total, device=device,
    )
    n_gt = T_true.shape[0]

    print(f"\n  {'Time[s]':>8}  {'Step':>5}  {'T_mean':>9}  "
          f"{'MAE[K]':>9}  {'R²':>8}  {'Window'}")
    print(f"  {'-'*62}")

    results      = {}
    summary_rows = []

    for t_target in sorted(target_times):
        T_at_t, step = predict_at_time(T_pred, t_target, cfg.dt)
        in_window    = step < n_gt

        # Determine which window this time belongs to
        if t_target <= cfg.train_time_end:
            window_label = "training"
        elif t_target <= cfg.predict_time_end:
            window_label = "VERIFICATION"
        else:
            window_label = "beyond-GT"

        row = {
            "target_time":  float(t_target),
            "step":         int(step),
            "window":       window_label,
            "in_gt_window": bool(in_window),
            "T_pred_mean":  float(T_at_t.mean()),
        }

        if in_window:
            T_ref = T_true[step]
            m     = compute_metrics(T_at_t, T_ref)
            row.update({
                "mae":        m["mae"],
                "rmse":       m["rmse"],
                "r2":         m["r2"],
                "within_5K":  within_tolerance(T_at_t, T_ref,  5.0),
                "within_10K": within_tolerance(T_at_t, T_ref, 10.0),
            })
            print(
                f"  {t_target:>8.0f}  {step:>5}  "
                f"{T_at_t.mean():>9.1f}  "
                f"{m['mae']:>9.2f}  {m['r2']:>8.4f}  [{window_label}]"
            )
        else:
            row.update({"mae": None, "rmse": None, "r2": None})
            print(
                f"  {t_target:>8.0f}  {step:>5}  "
                f"{T_at_t.mean():>9.1f}  {'—':>9}  {'—':>8}  [{window_label}]"
            )

        results[float(t_target)] = row
        summary_rows.append(row)

    # Plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n_gt_avail = min(n_gt, n_steps_total + 1)
    times_gt   = np.arange(n_gt_avail) * cfg.dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times_gt, T_true[:n_gt_avail].mean(axis=1),
             "k-", lw=2, label="OpenFOAM")
    ax1.plot(times_gt, T_pred[:n_gt_avail].mean(axis=1),
             "r--", lw=2, label="GNN")
    ax1.axvline(cfg.train_time_end, color="orange", ls="--", lw=2,
                label=f"Train end ({cfg.train_time_end:.0f}s)")
    ax1.axvspan(cfg.train_time_end, cfg.predict_time_end,
                alpha=0.08, color="red", label="Verification window")
    for t_target in target_times:
        col = "green" if t_target <= cfg.train_time_end else "red"
        ax1.axvline(t_target, color=col, ls=":", lw=1, alpha=0.7)
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Mean T [K]")
    ax1.set_title(f"Temperature evolution — Sim {sim_idx}")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    err_mean = np.mean(np.abs(T_pred[:n_gt_avail] - T_true[:n_gt_avail]), axis=1)
    ax2.plot(times_gt, err_mean, "steelblue", lw=1.5)
    ax2.axvline(cfg.train_time_end, color="orange", ls="--", lw=2)
    ax2.axvspan(cfg.train_time_end, cfg.predict_time_end,
                alpha=0.08, color="red", label="Verification (unseen)")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("MAE [K]")
    ax2.set_title("Error over time")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Sim {sim_idx} — Train: 0–{cfg.train_time_end:.0f}s  "
                 f"| Verify: {cfg.train_time_end:.0f}–{cfg.predict_time_end:.0f}s",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/multi_time_sim{sim_idx:03d}.png", dpi=150)
    plt.close(fig)

    with open(f"{save_dir}/results_sim{sim_idx:03d}.json", "w") as f:
        json.dump({"sim_idx": sim_idx, "predictions": summary_rows}, f, indent=2)

    print(f"\n  Outputs → {save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--sim_idx",      type=int, default=0)
    parser.add_argument("--target_times", type=float, nargs="+",
                        default=[500, 1000, 2000, 3000, 3200, 3500, 3800, 4000],
                        help="Include times > 3200 to test verification window")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg    = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"
    save_dir = f"{cfg.output_dir}/predictions/multi_time_sim{args.sim_idx:03d}"

    multi_time_prediction(
        cfg=cfg, checkpoint_path=ckpt, sim_idx=args.sim_idx,
        target_times=args.target_times, device=device, save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
