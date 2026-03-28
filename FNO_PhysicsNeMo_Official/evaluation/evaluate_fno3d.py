"""
3D FNO Rollout Evaluation — per-region MAE.
Interpolates grid predictions back to original mesh cells.
Reports Phase 1 (0-3200s) and Phase 2 (3200-4000s) accuracy.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, REGION_IDS, HEATER_REGIONS
from models.fno_model import HeatTreatmentFNO3D
from scipy.interpolate import NearestNDInterpolator
import h5py


def rollout_fno3d(model, dataset, sim_i, device, start_t=20):
    """Autoregressive rollout on 3D grid, interpolate back to mesh."""
    sim = dataset._simulations[sim_i]
    static = dataset._static_grids[sim_i]
    fields = static["interp_fields"]
    cfg = dataset.cfg

    T_set = sim["T_set"]
    times = sim["times"]
    coords = sim["coords"]
    T_all = sim["T_all"]
    n_times = sim["n_times"]
    total_cells = sim["total_cells"]

    Gx, Gy, Gz = dataset.grid_shape
    grid_points = dataset.grid_points

    # Build reverse interpolator: grid -> mesh
    # For each mesh cell, find nearest grid point
    gx = np.linspace(cfg.x_min, cfg.x_max, cfg.grid_x)
    gy = np.linspace(cfg.y_min, cfg.y_max, cfg.grid_y)
    gz = np.linspace(cfg.z_min, cfg.z_max, cfg.grid_z)

    # Start from ground truth at start_t
    T_cur_cells = T_all[start_t].copy()  # (total_cells,)

    # Interpolate initial T to grid
    interp_init = NearestNDInterpolator(coords, T_cur_cells)
    T_cur_grid = interp_init(grid_points).reshape(Gx, Gy, Gz)

    # Storage for predictions (on original mesh cells)
    n_rollout = n_times - start_t
    T_pred_all = np.zeros((n_rollout, total_cells), dtype=np.float32)
    T_pred_all[0] = T_cur_cells  # t=start_t is ground truth

    model.eval()
    with torch.no_grad():
        for step in range(1, n_rollout):
            t_idx = start_t + step
            if t_idx >= n_times:
                break
            t_val = times[t_idx - 1]  # time of current state

            # Build input on grid
            T_norm = (T_cur_grid - dataset.T_mean) / dataset.T_std
            Tset_norm = (T_set - dataset.T_mean) / dataset.T_std
            t_norm = t_val / 4000.0

            x = np.zeros((1, 7, Gx, Gy, Gz), dtype=np.float32)
            x[0, 0] = T_norm
            x[0, 1] = Tset_norm
            x[0, 2] = fields["region_id"].squeeze(-1)
            x[0, 3] = t_norm
            x[0, 4] = fields["is_heater"].squeeze(-1)
            x[0, 5] = fields["kappa"].squeeze(-1)
            x[0, 6] = fields["rho"].squeeze(-1)

            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            pred = model(x_t)

            # Denormalise dT on grid
            dT_grid = pred[0, 0].cpu().numpy() * dataset.dT_std + dataset.dT_mean
            T_next_grid = T_cur_grid + dT_grid

            # Interpolate from grid back to mesh cells
            interp_back = NearestNDInterpolator(grid_points, T_next_grid.ravel())
            T_next_cells = interp_back(coords).astype(np.float32)

            # Clamp heater nodes to ground truth (boundary conditions)
            is_heater = sim["is_heater"]
            heater_mask = is_heater > 0.5
            if t_idx < n_times:
                T_next_cells[heater_mask] = T_all[t_idx][heater_mask]

            T_pred_all[step] = T_next_cells

            # Update grid for next step
            interp_next = NearestNDInterpolator(coords, T_next_cells)
            T_cur_grid = interp_next(grid_points).reshape(Gx, Gy, Gz)

    return T_pred_all, T_all[start_t:start_t + n_rollout]


def compute_region_metrics(T_pred, T_true, sim, start_t, n_train_steps):
    """Compute per-region MAE for Phase 1 and Phase 2."""
    results = {}

    # Find region boundaries from the sim data
    region_onehot = sim["region_onehot"]  # (total_cells, 12)
    region_ids = np.argmax(region_onehot, axis=1)
    is_heater = sim["is_heater"]

    region_names = {v: k for k, v in REGION_IDS.items()}

    p1_end = min(n_train_steps - start_t, T_pred.shape[0])

    for rid in range(12):
        rname = region_names.get(rid, f"region_{rid}")
        mask = (region_ids == rid)
        n_cells = mask.sum()
        if n_cells == 0:
            continue

        # Phase 1: 0 to 3200s (training window)
        if p1_end > 1:
            p1_pred = T_pred[1:p1_end, mask]
            p1_true = T_true[1:p1_end, mask]
            p1_mae = float(np.mean(np.abs(p1_pred - p1_true)))
        else:
            p1_mae = 0.0

        # Phase 2: 3200s to 4000s (extrapolation)
        if p1_end < T_pred.shape[0]:
            p2_pred = T_pred[p1_end:, mask]
            p2_true = T_true[p1_end:, mask]
            if p2_pred.size > 0:
                p2_mae = float(np.mean(np.abs(p2_pred - p2_true)))
            else:
                p2_mae = float('nan')
        else:
            p2_mae = float('nan')

        is_h = "heater" in rname or rname == "brick_heater"
        results[rname] = {
            "n_cells": int(n_cells),
            "p1_mae": p1_mae,
            "p2_mae": p2_mae,
            "is_heater": is_h,
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_sims", type=int, default=3, help="Number of test sims to evaluate")
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  3D FNO ROLLOUT EVALUATION — Per-Region Accuracy")
    print(f"  Phase 1: 0-3200s (training window)")
    print(f"  Phase 2: 3200-4000s (extrapolation)")
    print(f"{sep}\n")

    # Load test dataset
    print("  Loading test dataset...")
    test_ds = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")

    # Load best model
    ckpt_path = f"{cfg.checkpoint_dir}/best_model.pt"
    print(f"  Loading model from {ckpt_path}")
    model = HeatTreatmentFNO3D.load(ckpt_path, cfg, device)
    model.eval()

    start_t = 20
    n_train_steps = cfg.n_train_steps  # 320

    all_results = {}
    n_eval = min(args.n_sims, len(test_ds.sim_indices))

    for i, sim_i in enumerate(test_ds.sim_indices[:n_eval]):
        sim = test_ds._simulations[sim_i]
        print(f"\n  Sim {sim_i} (T_set={sim['T_set']:.0f}K)")
        print(f"  {'Region':>18} | {'Cells':>6} | {'P1 MAE':>8} | {'P2 MAE':>8}")
        print(f"  {'-'*50}")

        t0 = time.time()
        T_pred, T_true = rollout_fno3d(model, test_ds, sim_i, device, start_t)
        rollout_time = time.time() - t0

        metrics = compute_region_metrics(T_pred, T_true, sim, start_t, n_train_steps)

        for rname, m in sorted(metrics.items()):
            p2_str = f"{m['p2_mae']:.2f}K" if not np.isnan(m['p2_mae']) else "N/A"
            tag = " (BC)" if m['is_heater'] else ""
            print(f"  {rname:>18} | {m['n_cells']:>6} | {m['p1_mae']:>7.2f}K | {p2_str:>8}{tag}")

        # Aggregate non-heater
        nh = {k: v for k, v in metrics.items() if not v['is_heater']}
        if nh:
            p1_all = np.mean([v['p1_mae'] for v in nh.values()])
            p2_vals = [v['p2_mae'] for v in nh.values() if not np.isnan(v['p2_mae'])]
            p2_all = np.mean(p2_vals) if p2_vals else float('nan')
            print(f"  {'NON-HEATER AVG':>18} |        | {p1_all:>7.2f}K | {p2_all:.2f}K")

        print(f"  Rollout time: {rollout_time:.1f}s ({T_pred.shape[0]} steps)")
        all_results[f"sim_{sim_i}"] = metrics

    # Summary
    print(f"\n{sep}")
    print(f"  SUMMARY — 3D FNO ROLLOUT")
    print(f"{sep}")

    # Collect across all sims
    for rname in ["steel_cylinder", "inner_box", "outer_box"]:
        p1_vals = []
        p2_vals = []
        for sim_key, metrics in all_results.items():
            if rname in metrics:
                p1_vals.append(metrics[rname]["p1_mae"])
                if not np.isnan(metrics[rname]["p2_mae"]):
                    p2_vals.append(metrics[rname]["p2_mae"])
        if p1_vals:
            p2_str = f"{np.mean(p2_vals):.2f}K" if p2_vals else "N/A"
            print(f"  {rname:>18}: P1={np.mean(p1_vals):.2f}K  P2={p2_str}")

    # Save results
    out_path = f"{cfg.output_dir}/evaluation/fno3d_rollout_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
