"""
Rollout for ALL regions — v2 with heater clamping.
Heaters are boundary conditions (controlled at T_set), not predicted.
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from models.meshgraphnet import HeatTreatmentGNN

HEATER_REGIONS = {
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
}

@torch.no_grad()
def rollout_all_regions(model, dataset, sim_idx, start_t=40, device="cuda"):
    model.eval()
    model.to(device)

    sim     = dataset._simulations[sim_idx]
    n_times = sim["n_times"]
    n_steps = n_times - start_t - 1
    T_set   = sim["T_set"]
    times   = sim["times"]

    results = {}

    for region, rdata in sim["region_data"].items():
        n_cells   = len(rdata["T_array"][start_t])
        region_id = rdata["region_id"]
        coords    = rdata["coords"]

        # For heaters: skip autoregressive rollout, use ground truth
        # Heaters are boundary conditions — they are CONTROLLED, not predicted
        if region in HEATER_REGIONS:
            T_true = rdata["T_array"][start_t: start_t + n_steps + 1]
            T_pred = T_true.copy()  # heaters = known boundary condition
            results[region] = (T_pred, T_true)
            continue

        edge_index, edge_attr = dataset._graphs[sim_idx][region]
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.float().to(device)

        T_current = rdata["T_array"][start_t].copy().astype(np.float64)

        T_rollout    = np.zeros((n_steps + 1, n_cells), dtype=np.float32)
        T_rollout[0] = T_current

        for step in range(n_steps):
            t_idx  = start_t + step
            t_val  = float(times[t_idx])
            t_norm = t_val / 4000.0

            T_norm    = ((T_current - dataset.T_mean) /
                         (dataset.T_std + 1e-8)).astype(np.float32)
            Tset_norm = float((T_set - dataset.T_mean) /
                              (dataset.T_std + 1e-8))

            node_feats = np.column_stack([
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                T_norm,
                np.full(n_cells, Tset_norm,    dtype=np.float32),
                np.full(n_cells, region_id/10, dtype=np.float32),
                np.full(n_cells, t_norm,       dtype=np.float32),
            ]).astype(np.float32)

            x = torch.tensor(node_feats, dtype=torch.float32, device=device)
            batch = Batch.from_data_list([
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            ]).to(device)

            dT_norm   = model(batch).squeeze(-1).reshape(-1).cpu().numpy()
            dT        = dT_norm * dataset.dT_std + dataset.dT_mean
            T_current = T_current + dT

            # Clamp: steel can't exceed T_set, nothing below 290K
            if region == "steel_cylinder":
                T_current = np.clip(T_current, 290.0, T_set)
            else:
                T_current = np.clip(T_current, 290.0, T_set + 50.0)

            T_rollout[step + 1] = T_current.astype(np.float32)

        T_true = rdata["T_array"][start_t: start_t + n_steps + 1]
        results[region] = (T_rollout, T_true)

    return results
