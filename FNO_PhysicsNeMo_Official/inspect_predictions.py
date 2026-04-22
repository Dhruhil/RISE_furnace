"""
Inspect real vs predicted temperatures for test sims.
Saves per-region T(t) mean curves to CSV + prints key timestamps.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, HEATER_REGIONS
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_fno3d
from scipy.interpolate import NearestNDInterpolator


def inspect_sim(model, dataset, sim_i, device="cuda", start_t=20):
    """Get real + predicted T curves for each region of one sim."""
    sim = dataset._simulations[sim_i]
    T_set = sim["T_set"]
    times = sim["times"]
    coords = sim["coords"]

    T_pred_grids, _ = rollout_fno3d(model, dataset, sim_i, device, start_t)
    n_steps = T_pred_grids.shape[0]
    grid_points = dataset.grid_points

    out = {"sim_i": sim_i, "T_set": T_set, "regions": {}}

    for region, slc in sim["region_slices"].items():
        if region in HEATER_REGIONS:
            continue  # skip heaters (they're clamped)
        s, e = slc
        region_coords = coords[s:e]

        # Ground truth mean temp per timestep
        T_true_mean = np.zeros(n_steps)
        T_pred_mean = np.zeros(n_steps)
        T_true_max  = np.zeros(n_steps)
        T_pred_max  = np.zeros(n_steps)
        T_true_min  = np.zeros(n_steps)
        T_pred_min  = np.zeros(n_steps)

        for step in range(n_steps):
            t_idx = start_t + step
            if t_idx >= sim["n_times"]:
                break
            T_true_cells = sim["T_all"][t_idx, s:e]
            T_true_mean[step] = T_true_cells.mean()
            T_true_max[step]  = T_true_cells.max()
            T_true_min[step]  = T_true_cells.min()

            interp = NearestNDInterpolator(grid_points, T_pred_grids[step].ravel())
            T_pred_cells = interp(region_coords)
            T_pred_mean[step] = T_pred_cells.mean()
            T_pred_max[step]  = T_pred_cells.max()
            T_pred_min[step]  = T_pred_cells.min()

        time_arr = times[start_t:start_t + n_steps]
        out["regions"][region] = {
            "times":       time_arr.tolist(),
            "T_true_mean": T_true_mean.tolist(),
            "T_pred_mean": T_pred_mean.tolist(),
            "T_true_max":  T_true_max.tolist(),
            "T_pred_max":  T_pred_max.tolist(),
            "T_true_min":  T_true_min.tolist(),
            "T_pred_min":  T_pred_min.tolist(),
        }
    return out


def main():
    cfg = CONFIG
    device = "cuda"
    model = HeatTreatmentFNO3D.load(f"{cfg.checkpoint_dir}/best_model.pt", cfg, device)
    dataset = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")

    all_data = {}
    for sim_i in dataset.sim_indices:
        sim = dataset._simulations[sim_i]
        print(f"\n{'='*76}")
        print(f"  sim_{sim_i}: T_set={sim['T_set']:.0f}K")
        print(f"{'='*76}")

        data = inspect_sim(model, dataset, sim_i, device=device)
        all_data[f"sim_{sim_i}"] = data

        # Print key timestamps for steel_cylinder + inner_box
        for region in ["steel_cylinder", "inner_box"]:
            if region not in data["regions"]:
                continue
            rd = data["regions"][region]
            t = np.array(rd["times"])
            true_m = np.array(rd["T_true_mean"])
            pred_m = np.array(rd["T_pred_mean"])

            print(f"\n  {region} — mean temperature over time:")
            print(f"    {'Time[s]':>8}  {'Real[K]':>9}  {'Pred[K]':>9}  {'Error[K]':>9}  {'Rel%':>6}")
            print(f"    {'-'*52}")
            # Print every 50th timestep + last
            idxs = list(range(0, len(t), 50)) + [len(t) - 1]
            for i in idxs:
                err = pred_m[i] - true_m[i]
                rel = 100 * abs(err) / true_m[i] if true_m[i] > 0 else 0
                print(f"    {t[i]:>8.0f}  {true_m[i]:>9.2f}  {pred_m[i]:>9.2f}  "
                      f"{err:>+9.2f}  {rel:>5.2f}%")

    # Save everything as JSON
    out_path = f"{cfg.output_dir}/evaluation/real_vs_predicted.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  Full T(t) curves saved to: {out_path}")


if __name__ == "__main__":
    main()
