"""
Autoregressive rollout for MeshGraphNet.

BUGS FIXED vs old version:
  1. model(batch).squeeze(-1).cpu().numpy() — if N=1 node, squeeze(-1) gives
     a scalar, not array. Fixed with .reshape(-1) after squeeze.
  2. rollout_from_dataset() rebuilt nmu/nstd from dataset.X_mean manually
     instead of using dataset._nmu/_nstd which are now always present.
     Both approaches give the same result but using _nmu/_nstd is cleaner
     and consistent with how __getitem__ normalises.
  3. rollout_from_dataset() used `sim_idx` as direct index into _simulations
     but if sim_idx is 45 (a test sim), _simulations[45] is correct since
     all sims are stored. However _graphs[sim_idx] only exists if sim_idx
     is in sim_indices — added a guard for this.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from models.meshgraphnet import HeatTreatmentGNN


@torch.no_grad()
def rollout_simulation(
    model:         HeatTreatmentGNN,
    initial_graph: Data,
    n_steps:       int,
    Y_std:         float,
    Y_mean:        float,
    node_mean:     np.ndarray,    # (10,) per-feature mean
    node_std:      np.ndarray,    # (10,) per-feature std
    device:        str   = "cuda",
    add_noise:     bool  = False,
    noise_std:     float = 0.003,
) -> np.ndarray:
    """
    Roll out the GNN autoregressively for n_steps time steps.

    Returns:
        T_rollout : (n_steps+1, n_nodes) temperatures in Kelvin
    """
    model.eval()
    model.to(device)

    T_current = initial_graph.T_current.numpy().copy()   # (n_nodes,)
    n_nodes   = T_current.shape[0]

    T_rollout = np.zeros((n_steps + 1, n_nodes), dtype=np.float32)
    T_rollout[0] = T_current

    x          = initial_graph.x.clone().to(device)
    edge_index = initial_graph.edge_index.to(device)
    edge_attr  = initial_graph.edge_attr.to(device)

    for step in range(n_steps):
        batch = Batch.from_data_list([
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        ]).to(device)

        # FIX: use .reshape(-1) instead of just .squeeze(-1) to handle N=1 case
        delta_T_norm = model(batch).squeeze(-1).reshape(-1).cpu().numpy()

        if add_noise:
            delta_T_norm += np.random.randn(*delta_T_norm.shape) * noise_std

        delta_T   = delta_T_norm * Y_std
        T_current = T_current + delta_T
        T_rollout[step + 1] = T_current

        # Update T_now (feature index 3) with new temperature
        T_norm_new   = (T_current - Y_mean) / (Y_std + 1e-8)
        x_np         = x.cpu().numpy().copy()
        x_np[:, 3]   = T_norm_new.astype(np.float32)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)

    return T_rollout


@torch.no_grad()
def rollout_from_dataset(
    model:     HeatTreatmentGNN,
    dataset,                        # HeatTreatmentDataset instance
    sim_idx:   int,
    start_t:   int   = 0,
    n_steps:   int | None = None,
    device:    str   = "cuda",
    add_noise: bool  = False,
    noise_std: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out for a specific simulation from the dataset.

    For Option A verification: call with n_steps=cfg.n_total_steps (400)
    to roll out all the way to t=4000s including the unseen 3200–4000s window.

    Returns:
        T_pred : (n_steps+1, n_nodes)   — full prediction
        T_true : (n_gt_steps+1, n_nodes) — ground truth (may be shorter)
    """
    sim     = dataset._simulations[sim_idx]
    n_times = sim["n_times"]

    if n_steps is None:
        n_steps = n_times - start_t - 1

    n_steps_gt = min(n_steps, n_times - start_t - 1)

    # FIX: if sim_idx not in _graphs (dataset loaded with different split),
    # build the graph on the fly
    if sim_idx in dataset._graphs:
        edge_index, edge_attr = dataset._graphs[sim_idx]
    else:
        coords = torch.tensor(sim["coords"], dtype=torch.float32)
        from data.graph_builder import build_knn_graph
        edge_index, edge_attr = build_knn_graph(coords, dataset.cfg.graph_k_neighbors)

    X_t0 = sim["X_3d"][start_t]
    T_t0 = sim["T_3d"][start_t]
    c    = dataset.col

    node_feats = np.column_stack([
        X_t0[:, c["x"]],
        X_t0[:, c["y"]],
        X_t0[:, c["z"]],
        T_t0,
        X_t0[:, c["T_set"]],
        X_t0[:, c["cy"]],
        X_t0[:, c["cz"]],
        X_t0[:, c["kappa"]],
        X_t0[:, c["Cp"]],
        X_t0[:, c["rho"]],
    ]).astype(np.float32)

    # FIX: use dataset._nmu/_nstd (always present in fixed dataset.py)
    nmu  = dataset._nmu    # (10,)
    nstd = dataset._nstd   # (10,)
    node_norm = (node_feats - nmu) / (nstd + 1e-8)

    initial_graph = Data(
        x          = torch.tensor(node_norm, dtype=torch.float32),
        edge_index = edge_index,
        edge_attr  = edge_attr.float(),
        T_current  = torch.tensor(T_t0, dtype=torch.float32),
    )

    T_pred = rollout_simulation(
        model, initial_graph, n_steps,
        Y_std      = dataset.Y_std,
        Y_mean     = dataset.Y_mean,
        node_mean  = nmu,
        node_std   = nstd,
        device     = device,
        add_noise  = add_noise,
        noise_std  = noise_std,
    )

    T_true = sim["T_3d"][start_t: start_t + n_steps_gt + 1]
    return T_pred, T_true


def predict_at_time(
    T_rollout:   np.ndarray,
    target_time: float,
    dt:          float,
    start_time:  float = 0.0,
) -> tuple[np.ndarray, int]:
    """Extract prediction at a specific time in seconds."""
    step_idx = int(round((target_time - start_time) / dt))
    step_idx = max(0, min(step_idx, T_rollout.shape[0] - 1))
    return T_rollout[step_idx], step_idx
