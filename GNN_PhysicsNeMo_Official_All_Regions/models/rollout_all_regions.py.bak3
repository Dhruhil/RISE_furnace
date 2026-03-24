"""
Rollout for ALL regions — supports future prediction beyond data.
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from models.meshgraphnet import HeatTreatmentGNN


@torch.no_grad()
def rollout_all_regions(model, dataset, sim_idx, start_t=40, n_steps=None, device="cuda"):
    model.eval()
    model.to(device)

    sim     = dataset._simulations[sim_idx]
    n_times = sim["n_times"]
    T_set   = sim["T_set"]
    times   = sim["times"]

    # Default: roll out to end of data. If n_steps given, can go beyond.
    data_steps = n_times - start_t - 1
    if n_steps is None:
        n_steps = data_steps

    results = {}

    for region, rdata in sim["region_data"].items():
        edge_index, edge_attr = dataset._graphs[sim_idx][region]
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.float().to(device)

        coords    = rdata["coords"]
        region_id = rdata["region_id"]
        T_max_region = sim["region_T_max"][region]
        T_min_region = sim["region_T_min"][region]

        T_current = rdata["T_array"][start_t].copy().astype(np.float64)

        T_rollout    = np.zeros((n_steps + 1, len(T_current)), dtype=np.float32)
        T_rollout[0] = T_current

        for step in range(n_steps):
            t_idx = start_t + step
            # For future steps beyond data, extrapolate time linearly
            if t_idx < n_times:
                t_val = float(times[t_idx])
            else:
                t_val = float(times[-1]) + (t_idx - n_times + 1) * 10.0
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
                np.full(len(T_current), Tset_norm,    dtype=np.float32),
                np.full(len(T_current), region_id/10, dtype=np.float32),
                np.full(len(T_current), t_norm,       dtype=np.float32),
            ]).astype(np.float32)

            x = torch.tensor(node_feats, dtype=torch.float32, device=device)
            batch = Batch.from_data_list([
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            ]).to(device)

            dT_norm = model(batch).squeeze(-1).reshape(-1).cpu().numpy()
            dT      = dT_norm * dataset.dT_std + dataset.dT_mean
            dT      = np.clip(dT, -5.0, 5.0)

            T_current = T_current + dT
            T_current = np.minimum(T_current, T_max_region)
            T_current = np.maximum(T_current, T_min_region)

            T_rollout[step + 1] = T_current.astype(np.float32)

        # Ground truth: only available up to data_steps
        gt_len = min(n_steps + 1, data_steps + 1)
        T_true = rdata["T_array"][start_t: start_t + gt_len]
        results[region] = (T_rollout, T_true)

    return results
