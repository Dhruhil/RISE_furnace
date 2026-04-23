"""
DeepONet rollout — matches FNO's rollout logic exactly.
Per FNO_PhysicsNeMo_Official/models/rollout.py:
  - Clip T to [290, T_set + 50] each step
  - Heaters = ground-truth temperature at current time (NOT constant T_set)
  - No dT clipping, no per-region bounds
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator


@torch.no_grad()
def rollout_deeponet(model, dataset, sim_i, device="cuda",
                     start_t=20, chunk_size=8192):
    model.eval()
    model.to(device)
    sim  = dataset._simulations[sim_i]
    sens = dataset._static_sensors[sim_i]
    cfg  = dataset.cfg

    T_mean = dataset.T_mean
    T_std  = dataset.T_std
    T_set  = sim["T_set"]
    Tset_norm = (T_set - dataset.Tset_mean) / dataset.Tset_std

    coords       = sim["coords"]
    n_cells      = sim["total_cells"]
    heater_cells = sim["is_heater"] > 0.5
    times        = sim["times"]
    n_t          = sim["n_times"]
    T_all_gt     = sim["T_all"]

    n_rollout = n_t - start_t
    T_pred = np.zeros((n_rollout, n_cells), dtype=np.float32)
    T_true = np.zeros((n_rollout, n_cells), dtype=np.float32)
    T_pred[0] = T_all_gt[start_t]
    T_true[0] = T_all_gt[start_t]
    for step in range(1, n_rollout):
        T_true[step] = T_all_gt[start_t + step]

    trunk_static = np.stack([
        coords[:, 0], coords[:, 1], coords[:, 2],
        sim["region_id"] / 11.0,
        sim["is_heater"],
        sim["kappa"]  / 100.0,
        sim["Cp"]    / 1000.0,
        sim["rho"]  / 10000.0,
    ], axis=1).astype(np.float32)
    trunk_all = torch.from_numpy(trunk_static).to(device)

    T_cur = T_pred[0].copy()

    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_t:
            break
        t_val = times[t_idx - 1]

        # Build branch (T_cur -> sensor lattice)
        interp = NearestNDInterpolator(coords, T_cur)
        T_sens = interp(dataset.sensor_points).astype(np.float32)
        T_sens_norm = (T_sens - T_mean) / T_std

        branch = np.stack([
            T_sens_norm, sens["region_id"], sens["is_heater"],
            sens["kappa"], sens["Cp"], sens["rho"],
        ], axis=0).astype(np.float32)
        branch = torch.from_numpy(branch).unsqueeze(0).to(device)
        scalars = torch.tensor(
            [Tset_norm, t_val / cfg.t_total], dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Forward (chunked over cells)
        preds = torch.zeros(n_cells, dtype=torch.float32, device=device)
        for s in range(0, n_cells, chunk_size):
            e = min(n_cells, s + chunk_size)
            y = trunk_all[s:e].unsqueeze(0)
            out = model(branch, scalars, y)
            preds[s:e] = out.squeeze(0)

        T_next_norm = preds.cpu().numpy()
        T_next = T_next_norm * T_std + T_mean

        # === MATCH FNO: clip all cells to [290, T_set + 50] ===
        T_next = np.clip(T_next, 290.0, T_set + 50.0)

        # === MATCH FNO: heaters = ground-truth T at this timestep ===
        T_next[heater_cells] = T_all_gt[t_idx, heater_cells]

        # NaN guard
        bad = np.isnan(T_next) | np.isinf(T_next)
        if bad.any():
            T_next[bad] = T_cur[bad]

        T_pred[step] = T_next.astype(np.float32)
        T_cur = T_next

    return T_pred, T_true
