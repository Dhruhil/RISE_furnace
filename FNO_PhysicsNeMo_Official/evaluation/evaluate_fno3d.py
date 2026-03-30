"""
3D FNO Rollout Evaluation — per-region MAE.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, REGION_IDS, HEATER_REGIONS
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_per_region
from utils.metrics import compute_metrics


def run_fno3d_evaluation(model, cfg, device="cuda", n_sims=None):
    """Run rollout evaluation on test simulations."""
    dataset = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
    
    sim_indices = dataset.sim_indices
    if n_sims is not None:
        sim_indices = sim_indices[:n_sims]

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  3D FNO ROLLOUT EVALUATION")
    print(f"  {len(sim_indices)} test sims | grid {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}")
    print(f"  P1: 0-{cfg.train_time_end:.0f}s | P2: {cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s")
    print(f"{sep}")
    print(f"  {'Sim':>4}  {'Region':>16}  {'Cells':>6}  {'P1 MAE':>8}  {'P2 MAE':>8}")
    print(f"  {'-'*60}")

    all_results = {}
    all_p1, all_p2 = [], []

    for sim_i in sim_indices:
        results = rollout_per_region(model, dataset, sim_i, device=device)
        all_results[f"sim_{sim_i}"] = results

        for region, r in sorted(results.items()):
            p2_str = f"{r['mae_p2']:.2f}K" if not np.isnan(r['mae_p2']) else "N/A"
            print(f"  {sim_i:>4}  {region:>16}  {r['n_cells']:>6}  "
                  f"{r['mae_p1']:>7.2f}K  {p2_str:>8}")
            all_p1.append(r["mae_p1"])
            if not np.isnan(r["mae_p2"]):
                all_p2.append(r["mae_p2"])
        print()

    print(f"{sep}")
    print(f"  SUMMARY ({len(sim_indices)} test sims)")
    print(f"  Phase 1 MAE: {np.mean(all_p1):.2f}K")
    if all_p2:
        print(f"  Phase 2 MAE: {np.mean(all_p2):.2f}K")
    print(f"{sep}")

    out_path = f"{cfg.output_dir}/evaluation/fno3d_rollout_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_sims", type=int, default=5)
    args = parser.parse_args()

    cfg = CONFIG
    ckpt_path = f"{cfg.checkpoint_dir}/best_model.pt"
    print(f"  Loading: {ckpt_path}")
    model = HeatTreatmentFNO3D.load(ckpt_path, cfg, args.device)
    run_fno3d_evaluation(model, cfg, device=args.device, n_sims=args.n_sims)


if __name__ == "__main__":
    main()
