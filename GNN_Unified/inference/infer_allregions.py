"""
Future prediction — ALL REGIONS.
Usage:
    python inference/infer_allregions.py --target_time 5000
    python inference/infer_allregions.py --target_time 6000 --sim_idx 0
"""
from __future__ import annotations
import argparse, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.base_config import BaseConfig, CONFIG
from data.dataset_all_regions import AllRegionsDataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout_all_regions import rollout_all_regions
from utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sim_idx", type=int, default=0)
    parser.add_argument("--target_time", type=float, default=5000.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = CONFIG
    cfg.node_in_features = 7
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = str(Path(cfg.checkpoint_dir).parent / "checkpoints_allregions")
    ckpt = args.checkpoint or f"{ckpt_dir}/best_model.pt"

    print(f"\n{'='*65}")
    print(f"  FUTURE PREDICTION — ALL REGIONS")
    print(f"  Target time : {args.target_time:.0f} s")
    print(f"  Train end   : {cfg.train_time_end:.0f} s")
    print(f"  Data end    : {cfg.predict_time_end:.0f} s")
    print(f"{'='*65}\n")

    model = HeatTreatmentGNN.load(ckpt, cfg, device)

    dataset = AllRegionsDataset(
        cfg.all_regions_dataset_path, cfg,
        split="test", split_mode="evaluation",
    )

    sim_idx = dataset.sim_indices[args.sim_idx]
    sim = dataset._simulations[sim_idx]
    print(f"  Sim: {sim['name']}  T_set={sim['T_set']:.0f}K")

    start_t = 40
    dt = cfg.dt
    n_steps = int(args.target_time / dt) - start_t

    results = rollout_all_regions(
        model, dataset, sim_idx,
        start_t=start_t, n_steps=n_steps, device=device,
    )

    save_dir = Path(cfg.output_dir) / "future_prediction"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_train_step = cfg.n_train_steps - start_t

    print(f"\n  {'Region':>16}  {'Pred steps':>10}  {'GT steps':>8}  "
          f"{'P1 MAE':>8}  {'P2 MAE':>8}  {'T_final':>10}")
    print(f"  {'-'*70}")

    all_results = {}
    for region, (T_pred, T_true) in results.items():
        n_pred = T_pred.shape[0]
        n_gt = T_true.shape[0]
        p1_end = min(n_train_step + 1, n_gt)

        m1 = compute_metrics(T_pred[:p1_end].ravel(), T_true[:p1_end].ravel())
        if p1_end < n_gt:
            m2 = compute_metrics(T_pred[p1_end:n_gt].ravel(), T_true[p1_end:n_gt].ravel())
        else:
            m2 = {"mae": 0.0}

        T_final = float(T_pred[-1].mean())
        print(f"  {region:>16}  {n_pred:>10}  {n_gt:>8}  "
              f"{m1['mae']:>8.2f}  {m2['mae']:>8.2f}  {T_final:>10.1f}K")

        all_results[region] = {
            "p1_mae": m1["mae"], "p2_mae": m2["mae"],
            "T_final_mean": T_final,
            "n_pred_steps": n_pred, "n_gt_steps": n_gt,
        }

    with open(save_dir / f"future_sim{args.sim_idx}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot ALL regions
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.colormaps["tab10"]
    for i, region in enumerate(results):
        T_pred, T_true = results[region]
        times_pred = np.arange(T_pred.shape[0]) * dt + start_t * dt
        times_true = np.arange(T_true.shape[0]) * dt + start_t * dt
        ax.plot(times_pred, T_pred.mean(axis=1), "--", lw=1.2, color=colors(i),
                label=f"{region} (GNN)")
        ax.plot(times_true, T_true.mean(axis=1), "-", lw=1.2, color=colors(i), alpha=0.5)
    ax.axvline(cfg.train_time_end, color="orange", ls="--", lw=2, label="Train end (3200s)")
    ax.axvline(cfg.predict_time_end, color="red", ls="--", lw=2, label="Data end (4000s)")
    ax.axvspan(cfg.predict_time_end, args.target_time, alpha=0.05, color="red",
               label="Future (no ground truth)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Mean T [K]")
    ax.set_title(f"Future prediction to {args.target_time:.0f}s — all regions")
    ax.legend(fontsize=7, ncol=3, loc="upper left"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"future_sim{args.sim_idx}.png", dpi=150)
    plt.close(fig)
    print(f"\n  Plot: {save_dir}/future_sim{args.sim_idx}.png")

if __name__ == "__main__":
    main()
