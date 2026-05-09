"""
Autoregressive rollout helpers for the 3D FNO.

Two entry points:
  rollout_fno3d(...)     -> raw grid-level rollout, returns (T_pred, T_true)
                            arrays on the FNO's native voxel grid
  rollout_per_region(...) -> wraps rollout_fno3d, projects the
                            predictions back onto the OpenFOAM mesh,
                            and reports a full metrics breakdown per
                            region for both Phase 1 and Phase 2

The metrics reported (MAE, RMSE, R^2, max error, within-tolerance %,
relative MAE, and per-step MAE trajectory) are the same set used in
the GNN evaluation script, so the two pipelines stay directly
comparable in the thesis tables.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator


@torch.no_grad()
def rollout_fno3d(model, dataset, sim_i, device="cuda", start_t=20):
    """
    Run an autoregressive rollout on the FNO's native voxel grid for
    one simulation.

    Each step the model predicts T_next from T_current; the prediction
    becomes the next T_current. Heater cells get clamped to the
    ground-truth values at every step (Dirichlet BC), mirroring how
    OpenFOAM enforces T = T_set on the heaters and brick heater.
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

    # Seed the rollout with the OpenFOAM ground-truth field at
    # start_t, projected onto the voxel grid via nearest-neighbour.
    T_t = sim["T_all"][start_t]
    interp_init = NearestNDInterpolator(sim["coords"], T_t)
    T_cur_grid = interp_init(dataset.grid_points).reshape(Gx, Gy, Gz)

    n_rollout = n_times - start_t
    T_pred_grids = np.zeros((n_rollout, Gx, Gy, Gz), dtype=np.float32)
    T_true_grids = np.zeros((n_rollout, Gx, Gy, Gz), dtype=np.float32)
    # Step 0 is the seed itself — both pred and true match the
    # OpenFOAM ground truth at start_t.
    T_pred_grids[0] = T_cur_grid
    T_true_grids[0] = T_cur_grid

    # ---- precompute the ground-truth field at every rollout step ----
    # Done up front in a small loop so the main rollout loop below
    # only deals with the FNO forward pass + heater clamping.
    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_times:
            break
        T_gt = sim["T_all"][t_idx]
        interp_gt = NearestNDInterpolator(sim["coords"], T_gt)
        T_true_grids[step] = interp_gt(dataset.grid_points).reshape(Gx, Gy, Gz)

    # Heater masks at both grid level and mesh level — used together
    # below to project the ground-truth heater temperatures back onto
    # the voxel grid at every step.
    heater_mask = fields["is_heater"].squeeze(-1) > 0.5
    heater_cells = sim["is_heater"] > 0.5

    # ---- main rollout loop -----------------------------------------
    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_times:
            break

        # Same 8-channel input layout as in dataset.__getitem__ —
        # any drift between training-time and rollout-time features
        # would quietly tank the rollout, so keep these in sync.
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

        # Denormalise back to physical Kelvin, then clip to a sane
        # range. The clip catches the rare case where a long rollout
        # drifts below room temperature or overshoots T_set badly —
        # without this, one bad step can poison everything that
        # follows through the autoregressive feedback.
        T_next_grid = pred_norm * T_std + T_mean
        T_next_grid = np.clip(T_next_grid, 290.0, T_set + 50.0)

        # Re-apply the Dirichlet BC on the heater cells. The heater
        # ground-truth values live on the mesh, so they need to be
        # interpolated onto the grid before the np.where below can
        # pick them up.
        T_gt_all = sim["T_all"][t_idx]
        interp_ht = NearestNDInterpolator(
            sim["coords"][heater_cells], T_gt_all[heater_cells])
        T_heater_grid = interp_ht(dataset.grid_points).reshape(Gx, Gy, Gz)
        T_next_grid = np.where(heater_mask, T_heater_grid, T_next_grid)

        T_pred_grids[step] = T_next_grid
        T_cur_grid = T_next_grid

    return T_pred_grids, T_true_grids


def _compute_full_metrics(pred, true):
    """
    Compute MAE, RMSE, R^2, max error, the three within-tolerance
    coverage figures, and relative MAE for one (pred, true) pair.

    All arrays get flattened first, so the same function works for
    per-step (1D), per-region (2D), or full-rollout (3D) inputs.
    """
    err = (pred - true).ravel()
    true_flat = true.ravel()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    maxe = float(np.max(np.abs(err)))

    # R^2 the textbook way: 1 - SS_res / SS_tot. The +1e-8 keeps the
    # denominator non-zero when true is constant (e.g. early heater
    # rollout where T_set hasn't moved).
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true_flat - true_flat.mean())**2)) + 1e-8
    r2 = float(1.0 - ss_res / ss_tot)

    # Three tolerance bands. 10K is the headline number reported in
    # the thesis since it matches the industrial control resolution;
    # 5K and 20K bracket it.
    within_5  = 100.0 * float(np.mean(np.abs(err) <=  5.0))
    within_10 = 100.0 * float(np.mean(np.abs(err) <= 10.0))
    within_20 = 100.0 * float(np.mean(np.abs(err) <= 20.0))

    # Relative MAE in percent of the ground-truth range, useful for
    # comparing across regions whose absolute temperatures differ.
    t_range = float(true_flat.max() - true_flat.min()) + 1e-8
    rel_mae = 100.0 * mae / t_range

    return {
        "mae": mae, "rmse": rmse, "r2": r2, "max_err": maxe,
        "within_5K": within_5, "within_10K": within_10, "within_20K": within_20,
        "rel_mae_pct": rel_mae,
    }


def rollout_per_region(model, dataset, sim_i, device="cuda", start_t=20):
    """
    Run a rollout on the 3D grid and report full per-region metrics
    on the ORIGINAL OpenFOAM mesh.

    Heater regions are treated as boundary conditions — the
    "prediction" is set equal to the ground truth on those cells, so
    the MAE comes out at zero by construction (matches the GNN
    evaluator behaviour). For all other regions, the FNO grid output
    is interpolated back onto the mesh cells of that region.

    Returns
    -------
    dict
        One entry per region. Each contains:
          mae_p1 / rmse_p1 / r2_p1 / max_err_p1            -- Phase-1 metrics
          within_5K_p1 / within_10K_p1 / within_20K_p1     -- Phase-1 coverage
          rel_mae_pct_p1                                   -- Phase-1 relative MAE
          ... same set for _p2 (Phase-2 / extrapolation)
          step_mae    (list)  -- per-timestep MAE trajectory
          n_cells, n_steps    -- bookkeeping
    """
    T_pred_grids, T_true_grids = rollout_fno3d(
        model, dataset, sim_i, device, start_t)

    sim = dataset._simulations[sim_i]
    cfg = dataset.cfg
    # Phase-1 length on the rollout time axis. cfg.n_train_steps is
    # measured from t=0; subtract start_t to convert it into a
    # rollout-step index.
    n_train_steps = cfg.n_train_steps - start_t
    grid_points = dataset.grid_points
    coords = sim["coords"]
    n_steps = T_pred_grids.shape[0]
    n_times = sim["n_times"]

    # Imported lazily to avoid a circular import — data.dataset
    # imports from models.* indirectly during testing.
    from data.dataset import HEATER_REGIONS

    results = {}
    for region, slc in sim["region_slices"].items():
        s, e = slc
        region_coords = coords[s:e]
        n_cells = region_coords.shape[0]

        # Pull the ground-truth temperature trajectory for this
        # region straight off the mesh — no interpolation needed.
        T_true_region = np.zeros((n_steps, n_cells), dtype=np.float32)
        for step in range(n_steps):
            t_idx = start_t + step
            if t_idx < n_times:
                T_true_region[step] = sim["T_all"][t_idx, s:e]

        if region in HEATER_REGIONS:
            # Heaters are clamped — copy the ground truth so the
            # downstream metrics are well-defined (and zero by
            # construction).
            T_pred_region = T_true_region.copy()
        else:
            # Predicted regions: project the FNO grid output back
            # onto the mesh cells of this region. One interpolator
            # per step — the kd-tree build is cheap relative to the
            # rollout itself.
            T_pred_region = np.zeros((n_steps, n_cells), dtype=np.float32)
            for step in range(n_steps):
                interp_pred = NearestNDInterpolator(
                    grid_points, T_pred_grids[step].ravel())
                T_pred_region[step] = interp_pred(region_coords)

        # ---- Phase-1 / Phase-2 split ---------------------------------
        # Phase 1 is the in-distribution (training-window) portion;
        # Phase 2 is the temporal-extrapolation portion that runs
        # past train_time_end. Some short rollouts don't reach
        # Phase 2 at all, in which case those metrics come back NaN.
        p1_end = min(n_train_steps + 1, n_steps)
        m_p1 = _compute_full_metrics(T_pred_region[:p1_end], T_true_region[:p1_end])

        m_p2 = {k: float("nan") for k in m_p1.keys()}
        if p1_end < n_steps:
            m_p2 = _compute_full_metrics(T_pred_region[p1_end:], T_true_region[p1_end:])

        # Per-timestep MAE trajectory across the full rollout —
        # plotted as the "MAE vs rollout time" curves in the thesis.
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