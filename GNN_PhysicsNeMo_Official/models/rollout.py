"""
Autoregressive temporal rollout for the MeshGraphNet surrogate.

The model predicts delta_T at each step:
    T(t+dt) = T(t) + delta_T_pred * Y_std

This can be rolled out for any number of steps — including
steps BEYOND the training window (extrapolation to t > 4000 s,
or predicting at arbitrary intermediate times like t=3000 s).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from models.meshgraphnet import HeatTreatmentGNN


@torch.no_grad()
def rollout_simulation(
    model: HeatTreatmentGNN,
    initial_graph: Data,
    n_steps: int,
    Y_std: float,
    Y_mean: float,
    node_mean: torch.Tensor,
    node_std: torch.Tensor,
    device: str = "cuda",
    add_noise: bool = False,
    noise_std: float = 0.003,
) -> np.ndarray:
    """
    Roll out the GNN autoregressively for n_steps time steps.

    Args:
        model:         Trained HeatTreatmentGNN
        initial_graph: PyG Data at t=0 (or any start time)
        n_steps:       Number of dt steps to predict forward
        Y_std:         Temperature std for denormalisation [K]
        Y_mean:        Temperature mean for denormalisation [K]
        node_mean:     Per-feature mean (10,) for re-normalising updated state
        node_std:      Per-feature std  (10,)
        device:        Torch device string
        add_noise:     Inject Gaussian noise (helps long rollouts)
        noise_std:     Std of injected noise (normalised units)

    Returns:
        T_rollout: np.ndarray  (n_steps+1, n_nodes)  temperature in K
    """
    model.eval()
    model.to(device)

    # Current temperature in raw K
    T_current = initial_graph.T_current.numpy().copy()   # (n_nodes,)
    n_nodes = T_current.shape[0]

    T_rollout = np.zeros((n_steps + 1, n_nodes), dtype=np.float32)
    T_rollout[0] = T_current

    # Clone graph for manipulation
    x = initial_graph.x.clone().to(device)         # (n_nodes, 10)
    edge_index = initial_graph.edge_index.to(device)
    edge_attr  = initial_graph.edge_attr.to(device)

    for step in range(n_steps):
        # Build batch (single graph)
        batch = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch = Batch.from_data_list([batch]).to(device)

        # Predict delta_T (normalised)
        delta_T_norm = model(batch).squeeze(-1).cpu().numpy()   # (n_nodes,)

        if add_noise:
            delta_T_norm += np.random.randn(*delta_T_norm.shape) * noise_std

        # Denormalise delta_T and update temperature
        delta_T = delta_T_norm * Y_std          # [K]
        T_current = T_current + delta_T
        T_rollout[step + 1] = T_current

        # Update the T_now feature in node feature vector (index 3)
        T_norm_new = (T_current - Y_mean) / (Y_std + 1e-8)
        x_np = x.cpu().numpy()
        x_np[:, 3] = T_norm_new
        x = torch.tensor(x_np, dtype=torch.float32).to(device)

    return T_rollout


@torch.no_grad()
def rollout_from_dataset(
    model: HeatTreatmentGNN,
    dataset,
    sim_idx: int,
    start_t: int = 0,
    n_steps: int | None = None,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out for a specific simulation from the dataset.

    Returns:
        T_pred:   (n_steps+1, n_nodes)  predicted temperatures [K]
        T_true:   (n_steps+1, n_nodes)  ground truth temperatures [K]
    """
    sim = dataset._simulations[sim_idx]
    n_times = sim["n_times"]

    if n_steps is None:
        n_steps = n_times - start_t - 1

    # Clamp to available ground truth (for comparison)
    n_steps_gt = min(n_steps, n_times - start_t - 1)

    # Build initial graph
    edge_index, edge_attr = dataset._graphs[sim_idx]

    X_t0   = sim["X_3d"][start_t]
    T_t0   = sim["T_3d"][start_t]

    c = dataset.col
    node_feats = np.column_stack([
        X_t0[:, c["x"]],  X_t0[:, c["y"]], X_t0[:, c["z"]],
        T_t0,
        X_t0[:, c["T_set"]], X_t0[:, c["cy"]], X_t0[:, c["cz"]],
        X_t0[:, c["kappa"]], X_t0[:, c["Cp"]], X_t0[:, c["rho"]],
    ]).astype(np.float32)

    keys = ["x","y","z","T_set","T_set","cy","cz","kappa","Cp","rho"]
    nmu  = np.array([dataset.X_mean[c[k]] for k in keys], dtype=np.float32)
    nstd = np.array([dataset.X_std[c[k]]  for k in keys], dtype=np.float32)
    nmu[3]  = dataset.Y_mean
    nstd[3] = dataset.Y_std

    node_norm = (node_feats - nmu) / (nstd + 1e-8)

    initial_graph = Data(
        x          = torch.tensor(node_norm, dtype=torch.float32),
        edge_index = edge_index,
        edge_attr  = edge_attr.float(),
        T_current  = torch.tensor(T_t0, dtype=torch.float32),
    )

    T_pred = rollout_simulation(
        model, initial_graph, n_steps,
        Y_std=dataset.Y_std, Y_mean=dataset.Y_mean,
        node_mean=torch.tensor(nmu), node_std=torch.tensor(nstd),
        device=device,
    )

    # Ground truth (only for steps within dataset)
    T_true = sim["T_3d"][start_t: start_t + n_steps_gt + 1]

    return T_pred[:n_steps_gt+1], T_true