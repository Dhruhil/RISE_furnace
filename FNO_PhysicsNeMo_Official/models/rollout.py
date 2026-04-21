"""
Autoregressive rollout for 3D FNO — all regions on regular grid.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Rolls out on the 3D grid, then interpolates back to mesh cells
for per-region accuracy reporting.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator


@torch.no_grad()
def rollout_fno3d(model, dataset, sim_i, device="cuda", start_t=20):
    """
    Autoregressive rollout on 3D grid for one simulation.
    
    Returns:
        T_pred_grid: (n_steps, Gx, Gy, Gz) — predictions on grid
        T_true_grid: (n_steps, Gx, Gy, Gz) — ground truth on grid
    """
    model.eval()
    model.to(device)

    sim = dataset._simulations[sim_i]
    static = dataset._static_grids[sim_i]
    fields = static["interp_fields"]
    cfg = dataset.cfg

    T_set = sim["T_set"]
    times = sim["times"]
    n_times = sim["n_times"]
    Gx, Gy, Gz = dataset.grid_shape

    T_mean = dataset.T_mean
    T_std = dataset.T_std

    # Start from ground truth at start_t
    T_t = sim["T_all"][start_t]
    interp_init = NearestNDInterpolator(sim["coords"], T_t)
    T_cur_grid = interp_init(dataset.grid_points).reshape(Gx, Gy, Gz)

    n_rollout = n_times - start_t
    T_pred_grids = np.zeros((n_rollout, Gx, Gy, Gz), dtype=np.float32)
    T_true_grids = np.zeros((n_rollout, Gx, Gy, Gz), dtype=np.float32)
    T_pred_grids[0] = T_cur_grid
    T_true_grids[0] = T_cur_grid

    # Precompute ground truth on grid
    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_times:
            break
        T_gt = sim["T_all"][t_idx]
        interp_gt = NearestNDInterpolator(sim["coords"], T_gt)
        T_true_grids[step] = interp_gt(dataset.grid_points).reshape(Gx, Gy, Gz)

    # Precompute heater mask on mesh (for clamping)
    heater_mask = fields["is_heater"].squeeze(-1) > 0.5
    heater_cells = sim["is_heater"] > 0.5

    # Autoregressive rollout
    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_times:
            break

        t_val = times[t_idx - 1]
        T_norm = (T_cur_grid - T_mean) / T_std
        Tset_norm = (T_set - T_mean) / T_std
        t_norm = t_val / dataset.cfg.t_total

        # Build 8-channel input (matches dataset.py)
        x = np.zeros((1, 8, Gx, Gy, Gz), dtype=np.float32)
        x[0, 0] = T_norm
        x[0, 1] = Tset_norm
        x[0, 2] = fields["region_id"].squeeze(-1)
        x[0, 3] = t_norm
        x[0, 4] = fields["is_heater"].squeeze(-1)
        x[0, 5] = fields["kappa"].squeeze(-1)
        x[0, 6] = fields["Cp"].squeeze(-1)
        x[0, 7] = fields["rho"].squeeze(-1)

        x_t = torch.tensor(x, dtype=torch.float32).to(device)
        pred_norm = model(x_t).squeeze(0).squeeze(0).cpu().numpy()

        # Denormalise
        T_next_grid = pred_norm * T_std + T_mean
        T_next_grid = np.clip(T_next_grid, 290.0, T_set + 50.0)

        # Clamp heaters to ground-truth temperature (not flat T_set)
        T_gt_all = sim["T_all"][t_idx]
        interp_ht = NearestNDInterpolator(
            sim["coords"][heater_cells], T_gt_all[heater_cells])
        T_heater_grid = interp_ht(dataset.grid_points).reshape(Gx, Gy, Gz)
        T_next_grid = np.where(heater_mask, T_heater_grid, T_next_grid)

        T_pred_grids[step] = T_next_grid
        T_cur_grid = T_next_grid

    return T_pred_grids, T_true_grids


def rollout_per_region(model, dataset, sim_i, device="cuda", start_t=20):
    """
    Rollout on 3D grid, then report per-region MAE on original mesh.
    
    For heater regions: uses ground truth directly (boundary conditions).
    For other regions: interpolates grid predictions back to mesh cells,
    and compares against original mesh ground truth (no grid round-trip).
    
    Returns dict: {region_name: {"mae_p1": float, "mae_p2": float, "n_cells": int}}
    """
    T_pred_grids, T_true_grids = rollout_fno3d(
        model, dataset, sim_i, device, start_t)

    sim = dataset._simulations[sim_i]
    cfg = dataset.cfg
    n_train_steps = cfg.n_train_steps - start_t
    grid_points = dataset.grid_points
    coords = sim["coords"]
    n_steps = T_pred_grids.shape[0]
    n_times = sim["n_times"]

    from data.dataset import HEATER_REGIONS

    results = {}
    for region, slc in sim["region_slices"].items():
        s, e = slc
        region_coords = coords[s:e]
        n_cells = region_coords.shape[0]

        # Ground truth: always use original mesh data (no grid round-trip)
        T_true_region = np.zeros((n_steps, n_cells), dtype=np.float32)
        for step in range(n_steps):
            t_idx = start_t + step
            if t_idx < n_times:
                T_true_region[step] = sim["T_all"][t_idx, s:e]

        # Heater regions are boundary conditions — pred = ground truth
        if region in HEATER_REGIONS:
            T_pred_region = T_true_region.copy()
        else:
            # Non-heater: interpolate grid predictions back to mesh
            T_pred_region = np.zeros((n_steps, n_cells), dtype=np.float32)
            for step in range(n_steps):
                interp_pred = NearestNDInterpolator(
                    grid_points, T_pred_grids[step].ravel())
                T_pred_region[step] = interp_pred(region_coords)

        p1_end = min(n_train_steps + 1, n_steps)
        p1_mae = float(np.mean(np.abs(
            T_pred_region[:p1_end] - T_true_region[:p1_end])))

        p2_mae = float("nan")
        if p1_end < n_steps:
            p2_mae = float(np.mean(np.abs(
                T_pred_region[p1_end:] - T_true_region[p1_end:])))

        results[region] = {
            "mae_p1": p1_mae,
            "mae_p2": p2_mae,
            "n_cells": n_cells,
        }

    return results
