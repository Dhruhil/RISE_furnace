"""
Autoregressive rollout for FNO — all regions.
FIXED: brick_heater + heater_1-8 clamped as boundary conditions.
"""
from __future__ import annotations

import numpy as np
import torch
from models.fno_model import HeatTreatmentFNO

HEATER_REGIONS = {
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
    "brick_heater",
}


@torch.no_grad()
def rollout_fno_all_regions(
    model:    HeatTreatmentFNO,
    dataset,
    sim_idx:  int,
    start_t:  int  = 40,
    n_steps:  int  = None,
    device:   str  = "cuda",
) -> dict:
    model.eval()
    model.to(device)

    sim     = dataset._simulations[sim_idx]
    n_times = sim["n_times"]
    T_set   = sim["T_set"]
    times   = sim["times"]

    data_steps = n_times - start_t - 1
    if n_steps is None:
        n_steps = data_steps

    results = {}

    for region, rdata in sim["region_data"].items():
        n_cells   = rdata["n_cells"]
        region_id = rdata["region_id"]

        T_max_region = sim["region_T_max"][region]
        T_min_region = sim["region_T_min"][region]

        # Heaters + brick_heater are boundary conditions — use ground truth
        if region in HEATER_REGIONS:
            gt_len = min(n_steps + 1, data_steps + 1)
            T_true = rdata["T_array"][start_t: start_t + gt_len]
            results[region] = (T_true.copy(), T_true)
            continue

        T_current = rdata["T_array"][start_t].copy().astype(np.float64)

        T_rollout    = np.zeros((n_steps + 1, n_cells), dtype=np.float32)
        T_rollout[0] = T_current

        for step in range(n_steps):
            t_idx = start_t + step
            if t_idx < n_times:
                t_val = float(times[t_idx])
            else:
                t_val = float(times[-1]) + (t_idx - n_times + 1) * 10.0
            t_norm = t_val / 4000.0

            T_norm    = ((T_current - dataset.T_mean) /
                         (dataset.T_std + 1e-8)).astype(np.float32)
            Tset_norm = float((T_set - dataset.Tset_mean) / dataset.Tset_std)
            rid_norm  = float(region_id / 11.0)

            x = np.stack([
                T_norm,
                np.full(n_cells, Tset_norm, dtype=np.float32),
                np.full(n_cells, rid_norm,  dtype=np.float32),
                np.full(n_cells, t_norm,    dtype=np.float32),
            ], axis=0).astype(np.float32)

            x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            dT_norm = model(x_t).squeeze(0).squeeze(0).cpu().numpy()

            dT        = dT_norm * dataset.dT_std + dataset.dT_mean
            T_next    = T_current + dT
            T_current = np.clip(T_next, T_min_region, T_max_region).astype(np.float64)
            T_rollout[step + 1] = T_current.astype(np.float32)

        gt_len = min(n_steps + 1, data_steps + 1)
        T_true = rdata["T_array"][start_t: start_t + gt_len]
        results[region] = (T_rollout, T_true)

    return results
