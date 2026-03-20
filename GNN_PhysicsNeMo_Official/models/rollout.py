"""
Autoregressive rollout — updates ALL features at each step.
Key fix: T_set and material properties change over time and must
be taken from the actual simulation data at each timestep.
Only T_now is predicted autoregressively — all other features
come from the ground truth X_3d array.
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from models.meshgraphnet import HeatTreatmentGNN


@torch.no_grad()
def rollout_from_dataset(
    model:    HeatTreatmentGNN,
    dataset,
    sim_idx:  int,
    start_t:  int = 20,
    n_steps:  int | None = None,
    device:   str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out autoregressively starting from start_t.

    At each step:
    - T_now comes from the model prediction (autoregressive)
    - ALL other features (T_set, coords, kappa, Cp, rho) come
      from ground truth X_3d — these are boundary conditions
      that are KNOWN in advance (furnace schedule)

    This is physically correct: we know the furnace schedule
    but predict how the metal responds to it.
    """
    model.eval()
    model.to(device)

    sim     = dataset._simulations[sim_idx]
    n_times = sim["n_times"]

    if n_steps is None:
        n_steps = n_times - start_t - 1

    n_steps = min(n_steps, n_times - start_t - 1)

    edge_index, edge_attr = dataset._graphs[sim_idx]
    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.float().to(device)

    c    = dataset.col
    nmu  = dataset._nmu
    nstd = dataset._nstd

    # Initial temperature from ground truth
    T_current = sim["T_3d"][start_t].copy().astype(np.float64)

    T_rollout       = np.zeros((n_steps + 1, len(T_current)), dtype=np.float32)
    T_rollout[0]    = T_current

    for step in range(n_steps):
        t_idx = start_t + step

        # Get ALL features from ground truth at this timestep
        X_t = sim["X_3d"][t_idx]

        # Build node features with PREDICTED T_now but GROUND TRUTH everything else
        node_feats = np.column_stack([
            X_t[:, c["x"]], X_t[:, c["y"]], X_t[:, c["z"]],
            T_current,                          # ← predicted T (autoregressive)
            X_t[:, c["T_set"]],                 # ← ground truth furnace setpoint
            X_t[:, c["cy"]], X_t[:, c["cz"]],
            X_t[:, c["kappa"]], X_t[:, c["Cp"]], X_t[:, c["rho"]],
        ]).astype(np.float32)

        node_norm = (node_feats - nmu) / (nstd + 1e-8)
        x = torch.tensor(node_norm, dtype=torch.float32, device=device)

        batch = Batch.from_data_list([
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        ]).to(device)

        # Predict normalised delta_T
        delta_T_norm = model(batch).squeeze(-1).reshape(-1).cpu().numpy()

        # Denormalise using dT_std
        delta_T   = delta_T_norm * dataset.dT_std + dataset.dT_mean
        T_current = T_current + delta_T

        # Physical constraint: temperature cannot exceed furnace setpoint
        T_set = X_t[:, c["T_set"]]
        T_current = np.minimum(T_current, T_set + 2.0)
        T_current = np.maximum(T_current, 290.0)          # cannot go below room temp
        T_rollout[step + 1] = T_current.astype(np.float32)

    T_true = sim["T_3d"][start_t: start_t + n_steps + 1]
    return T_rollout, T_true


def predict_at_time(T_rollout, target_time, dt, start_time=0.0):
    step = int(round((target_time - start_time) / dt))
    step = max(0, min(step, T_rollout.shape[0] - 1))
    return T_rollout[step], step
