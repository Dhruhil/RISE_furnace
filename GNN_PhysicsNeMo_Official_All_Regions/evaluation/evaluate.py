"""
Full test-set evaluation — ALL REGIONS.
"""
from __future__ import annotations
import argparse, json, sys
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
from utils.metrics import compute_metrics, within_tolerance

def evaluate_all_test_sims(model, cfg, device):
    model.eval()
    dataset = AllRegionsDataset(cfg.all_regions_dataset_path, cfg, split="test", split_mode="evaluation")
    n_train_steps = cfg.n_train_steps
    start_t = 40
    p1_end = n_train_steps - start_t
    all_p1_mae, all_p1_r2 = [], []
    all_p2_mae, all_p2_r2 = [], []
    per_sim_results = {}
    per_region_mae = {}
    print(f"\n{'='*70}")
    print(f"  TEST EVALUATION -- ALL REGIONS rollout (t={start_t*cfg.dt:.0f} -> 4000s)")
    print(f"{'='*70}")
    save_dir = Path(cfg.output_dir) / "evaluation_allregions"
    save_dir.mkdir(parents=True, exist_ok=True)
    for sim_i in dataset.sim_indices:
        results = rollout_all_regions(model, dataset, sim_i, start_t=start_t, device=device)
        sim_p1_pred, sim_p1_true = [], []
        sim_p2_pred, sim_p2_true = [], []
        for region, (T_pred, T_true) in results.items():
            n_steps = T_pred.shape[0]
            p1_slice = min(p1_end + 1, n_steps)
            if p1_slice > 0:
                m1 = compute_metrics(T_pred[:p1_slice].ravel(), T_true[:p1_slice].ravel())
                sim_p1_pred.append(T_pred[:p1_slice].ravel())
                sim_p1_true.append(T_true[:p1_slice].ravel())
            else:
                m1 = {"mae": float("nan"), "r2": float("nan")}
            if p1_end < n_steps:
                m2 = compute_metrics(T_pred[p1_end:].ravel(), T_true[p1_end:].ravel())
                sim_p2_pred.append(T_pred[p1_end:].ravel())
                sim_p2_true.append(T_true[p1_end:].ravel())
            else:
                m2 = {"mae": float("nan"), "r2": float("nan")}
            print(f"  {sim_i:>4}  {region:>16}  P1 MAE={m1['mae']:.2f}K  P2 MAE={m2['mae']:.2f}K")
            if region not in per_region_mae:
                per_region_mae[region] = {"p1": [], "p2": []}
            per_region_mae[region]["p1"].append(m1["mae"])
            per_region_mae[region]["p2"].append(m2["mae"])
        agg_p1, agg_p2 = {}, {}
        if sim_p1_pred:
            agg_p1 = compute_metrics(np.concatenate(sim_p1_pred), np.concatenate(sim_p1_true))
            all_p1_mae.append(agg_p1["mae"])
            all_p1_r2.append(agg_p1.get("r2", 0))
        if sim_p2_pred:
            agg_p2 = compute_metrics(np.concatenate(sim_p2_pred), np.concatenate(sim_p2_true))
            all_p2_mae.append(agg_p2["mae"])
            all_p2_r2.append(agg_p2.get("r2", 0))
        per_sim_results[sim_i] = {"p1_mae": agg_p1.get("mae"), "p2_mae": agg_p2.get("mae")}
        print(f"  {sim_i:>4}  {'AGGREGATE':>16}  P1={agg_p1.get('mae',0):.2f}K  P2={agg_p2.get('mae',0):.2f}K")
        print()
    summary = {
        "phase1": {"mean_mae": float(np.mean(all_p1_mae)) if all_p1_mae else None, "mean_r2": float(np.mean(all_p1_r2)) if all_p1_r2 else None},
        "phase2": {"mean_mae": float(np.mean(all_p2_mae)) if all_p2_mae else None, "mean_r2": float(np.mean(all_p2_r2)) if all_p2_r2 else None},
        "per_region": {r: {"p1": float(np.mean(v["p1"])), "p2": float(np.mean(v["p2"]))} for r, v in per_region_mae.items()},
    }
    with open(save_dir / "evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    regions = sorted(per_region_mae.keys())
    p1m = [np.mean(per_region_mae[r]["p1"]) for r in regions]
    p2m = [np.mean(per_region_mae[r]["p2"]) for r in regions]
    x = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.175, p1m, 0.35, label="Phase 1", color="steelblue")
    ax.bar(x + 0.175, p2m, 0.35, label="Phase 2", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.set_ylabel("MAE [K]"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(save_dir / "per_region_mae.png", dpi=150); plt.close(fig)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = str(Path(cfg.checkpoint_dir).parent / "checkpoints_allregions")
    ckpt = args.checkpoint or f"{ckpt_dir}/best_model.pt"
    cfg.node_in_features = 7
    model = HeatTreatmentGNN.load(ckpt, cfg, device)
    summary = evaluate_all_test_sims(model, cfg, device)
    p1 = summary["phase1"]
    p2 = summary["phase2"]
    print(f"\n{'='*70}")
    print(f"  Phase 1 (training):     MAE={p1['mean_mae']:.2f}K  R2={p1['mean_r2']:.4f}")
    print(f"  Phase 2 (verification): MAE={p2['mean_mae']:.2f}K  R2={p2['mean_r2']:.4f}")
    print(f"{'='*70}")
    for r, v in summary["per_region"].items():
        print(f"    {r:>16}: P1={v['p1']:.2f}K  P2={v['p2']:.2f}K")

if __name__ == "__main__":
    main()
