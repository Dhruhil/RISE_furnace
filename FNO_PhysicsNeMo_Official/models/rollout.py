"""
Autoregressive rollout for 3D FNO — all regions on regular grid.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Reports full metrics: MAE, RMSE, R², max error, within-tolerance %,
relative MAE, and per-timestep MAE trajectory.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator


@torch.no_grad()
def rollout_fno3d(model, dataset, sim_i, device="cuda", start_t=20):
    """Autoregressive rollout on 3D grid for one simulation."""
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

    T_t = sim["T_all"][start_t]
    interp_init = NearestNDInterpolator(sim["coords"], T_t)
    T_cur_grid = interp_init(dataset.grid_points).reshape(Gx, Gy, Gz)

    n_rollout = n_times - start_t
    T_pred_grids = np.zeros((n_rollout, Gx, Gy, Gz), dtype=np.float32)
    T_true_grids = np.zeros((n_rollout, Gx, Gy, Gz), dtype=np.float32)
    T_pred_grids[0] = T_cur_grid
    T_true_grids[0] = T_cur_grid

    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_times:
            break
        T_gt = sim["T_all"][t_idx]
        interp_gt = NearestNDInterpolator(sim["coords"], T_gt)
        T_true_grids[step] = interp_gt(dataset.grid_points).reshape(Gx, Gy, Gz)

    heater_mask = fields["is_heater"].squeeze(-1) > 0.5
    heater_cells = sim["is_heater"] > 0.5

    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_times:
            break

        t_val = times[t_idx - 1]
        T_norm = (T_cur_grid - T_mean) / T_std
        Tset_norm = (T_set - T_mean) / T_std
        t_norm = t_val / dataset.cfg.t_total

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

        T_next_grid = pred_norm * T_std + T_mean
        T_next_grid = np.clip(T_next_grid, 290.0, T_set + 50.0)

        T_gt_all = sim["T_all"][t_idx]
        interp_ht = NearestNDInterpolator(
            sim["coords"][heater_cells], T_gt_all[heater_cells])
        T_heater_grid = interp_ht(dataset.grid_points).reshape(Gx, Gy, Gz)
        T_next_grid = np.where(heater_mask, T_heater_grid, T_next_grid)

        T_pred_grids[step] = T_next_grid
        T_cur_grid = T_next_grid

    return T_pred_grids, T_true_grids


def _compute_full_metrics(pred, true):
    """MAE, RMSE, R², max error, within-tolerance %, relative MAE."""
    err = (pred - true).ravel()
    true_flat = true.ravel()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    maxe = float(np.max(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true_flat - true_flat.mean())**2)) + 1e-8
    r2 = float(1.0 - ss_res / ss_tot)
    within_5  = 100.0 * float(np.mean(np.abs(err) <=  5.0))
    within_10 = 100.0 * float(np.mean(np.abs(err) <= 10.0))
    within_20 = 100.0 * float(np.mean(np.abs(err) <= 20.0))
    t_range = float(true_flat.max() - true_flat.min()) + 1e-8
    rel_mae = 100.0 * mae / t_range
    return {
        "mae": mae, "rmse": rmse, "r2": r2, "max_err": maxe,
        "within_5K": within_5, "within_10K": within_10, "within_20K": within_20,
        "rel_mae_pct": rel_mae,
    }


def rollout_per_region(model, dataset, sim_i, device="cuda", start_t=20):
    """
    Rollout on 3D grid, report per-region full metrics on original mesh.

    Heater regions are treated as boundary conditions (pred = ground truth).
    Other regions: interpolate grid predictions back to mesh cells.

    Returns dict: {region_name: {mae_p1, rmse_p1, r2_p1, max_err_p1,
                                 within_5K_p1, within_10K_p1, within_20K_p1,
                                 rel_mae_pct_p1, ... same for _p2,
                                 step_mae (list), n_cells, n_steps}}
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

        T_true_region = np.zeros((n_steps, n_cells), dtype=np.float32)
        for step in range(n_steps):
            t_idx = start_t + step
            if t_idx < n_times:
                T_true_region[step] = sim["T_all"][t_idx, s:e]

        if region in HEATER_REGIONS:
            T_pred_region = T_true_region.copy()
        else:
            T_pred_region = np.zeros((n_steps, n_cells), dtype=np.float32)
            for step in range(n_steps):
                interp_pred = NearestNDInterpolator(
                    grid_points, T_pred_grids[step].ravel())
                T_pred_region[step] = interp_pred(region_coords)

        p1_end = min(n_train_steps + 1, n_steps)
        m_p1 = _compute_full_metrics(T_pred_region[:p1_end], T_true_region[:p1_end])

        m_p2 = {k: float("nan") for k in m_p1.keys()}
        if p1_end < n_steps:
            m_p2 = _compute_full_metrics(T_pred_region[p1_end:], T_true_region[p1_end:])

        step_mae = np.mean(np.abs(T_pred_region - T_true_region), axis=1).tolist()

        results[region] = {
            "n_cells": n_cells,
            "n_steps": n_steps,
            "mae_p1":         m_p1["mae"],
            "rmse_p1":        m_p1["rmse"],
            "r2_p1":          m_p1["r2"],
            "max_err_p1":     m_p1["max_err"],
            "within_5K_p1":   m_p1["within_5K"],
            "within_10K_p1":  m_p1["within_10K"],
            "within_20K_p1":  m_p1["within_20K"],
            "rel_mae_pct_p1": m_p1["rel_mae_pct"],
            "mae_p2":         m_p2["mae"],
            "rmse_p2":        m_p2["rmse"],
            "r2_p2":          m_p2["r2"],
            "max_err_p2":     m_p2["max_err"],
            "within_5K_p2":   m_p2["within_5K"],
            "within_10K_p2":  m_p2["within_10K"],
            "within_20K_p2":  m_p2["within_20K"],
            "rel_mae_pct_p2": m_p2["rel_mae_pct"],
            "step_mae":       step_mae,
        }

    return results
