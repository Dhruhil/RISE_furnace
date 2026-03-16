"""
PyTorch Dataset for steel-cylinder temperature time series.

Loads the HDF5 produced by Dataset_creation and returns
(graph_t, delta_T) pairs for autoregressive MeshGraphNet training.
"""

from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import h5py
from configs.base_config import BaseConfig
from data.graph_builder import build_knn_graph


class HeatTreatmentDataset(Dataset):
    """
    Each item is a PyG Data object containing:
        x          : node features (n_cells, node_in_features)
        edge_index : (2, n_edges)
        edge_attr  : (n_edges, 4)  [dx, dy, dz, dist]
        y          : delta_T_norm  (n_cells, 1)
        T_current  : T(t)  raw K
        T_next     : T(t+dt) raw K  (ground truth for evaluation)
    """

    def __init__(self, h5_path: str, cfg: BaseConfig,
                 split: str = "train", rollout_steps: int = 1):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.rollout_steps = rollout_steps

        # ---- Load raw data from HDF5 ----
        with h5py.File(h5_path, "r") as f:
            self.X_raw       = f["X_raw"][:].astype(np.float32)
            self.Y_raw       = f["Y_raw"][:].astype(np.float32)
            self.X_mean      = f["X_mean"][:].astype(np.float32)
            self.X_std       = f["X_std"][:].astype(np.float32)
            self.Y_mean      = float(f["Y_mean"][()])
            self.Y_std       = float(f["Y_std"][()])
            self.sim_starts  = f["sim_start_indices"][:].tolist()
            self.sim_n_rows  = f["sim_n_rows"][:].tolist()
            n_sims           = int(f.attrs["n_simulations"])
            self.feature_cols = json.loads(f.attrs["feature_cols"])

        self.col = {k: j for j, k in enumerate(self.feature_cols)}

        # ---- Reconstruct per-simulation 3D arrays ----
        self._simulations = []
        for i in range(n_sims):
            start  = self.sim_starts[i]
            n_rows = self.sim_n_rows[i]
            X_sim  = self.X_raw[start: start + n_rows]   # (n_times*n_cells, 15)
            Y_sim  = self.Y_raw[start: start + n_rows]   # (n_times*n_cells, 1)

            t_vals   = X_sim[:, self.col["t"]]
            unique_t = np.unique(t_vals)
            n_times  = len(unique_t)
            n_cells  = n_rows // n_times

            self._simulations.append({
                "sim_idx": i,
                "coords":  X_sim[:n_cells, :3],                        # (n_cells,3)
                "X_3d":    X_sim.reshape(n_times, n_cells, 15),
                "T_3d":    Y_sim.reshape(n_times, n_cells),
                "n_times": n_times,
                "n_cells": n_cells,
            })

        # ---- Split by simulation (no data leakage) ----
        n_test  = max(1, int(n_sims * cfg.test_fraction))
        n_val   = max(1, int(n_sims * cfg.val_fraction))
        n_train = n_sims - n_val - n_test

        split_sims = {
            "train": list(range(0, n_train)),
            "val":   list(range(n_train, n_train + n_val)),
            "test":  list(range(n_train + n_val, n_sims)),
        }
        self.sim_indices = split_sims[split]

        # ---- Enumerate (sim, t) index pairs ----
        self._index: list[tuple[int, int]] = []
        for sim_i in self.sim_indices:
            n_t = self._simulations[sim_i]["n_times"] - rollout_steps
            for t_i in range(n_t):
                self._index.append((sim_i, t_i))

        # ---- Pre-build graphs (k-NN fixed per simulation) ----
        self._graphs: dict[int, tuple] = {}
        for sim_i in self.sim_indices:
            coords = torch.tensor(self._simulations[sim_i]["coords"])
            edge_index, edge_attr = build_knn_graph(coords, cfg.graph_k_neighbors)
            self._graphs[sim_i] = (edge_index, edge_attr)

        print(f"  [{split}] {len(self._index):,} pairs across {len(self.sim_indices)} sims")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Data:
        sim_i, t_i = self._index[idx]
        sim = self._simulations[sim_i]
        edge_index, edge_attr = self._graphs[sim_i]
        c = self.col

        X_t   = sim["X_3d"][t_i]                              # (n_cells, 15)
        T_t   = sim["T_3d"][t_i]                              # (n_cells,)
        T_tp1 = sim["T_3d"][t_i + self.rollout_steps]

        # Node feature vector: [x, y, z, T_now, T_set, cy, cz, kappa, Cp, rho]
        node_feats = np.column_stack([
            X_t[:, c["x"]],   X_t[:, c["y"]],  X_t[:, c["z"]],
            T_t,
            X_t[:, c["T_set"]], X_t[:, c["cy"]], X_t[:, c["cz"]],
            X_t[:, c["kappa"]], X_t[:, c["Cp"]], X_t[:, c["rho"]],
        ]).astype(np.float32)  # (n_cells, 10)

        # Per-feature normalisation arrays (10-dim)
        keys = ["x","y","z","T_set","T_set","cy","cz","kappa","Cp","rho"]
        # index 3 is T_now → use Y_mean/Y_std
        nmu  = np.array([self.X_mean[c[k]] for k in keys], dtype=np.float32)
        nstd = np.array([self.X_std[c[k]]  for k in keys], dtype=np.float32)
        nmu[3]  = self.Y_mean
        nstd[3] = self.Y_std

        node_norm = (node_feats - nmu) / (nstd + 1e-8)

        delta_T      = (T_tp1 - T_t).reshape(-1, 1)
        delta_T_norm = delta_T / (self.Y_std + 1e-8)

        data = Data(
            x          = torch.tensor(node_norm,      dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr.float(),
            y          = torch.tensor(delta_T_norm,   dtype=torch.float32),
            T_current  = torch.tensor(T_t,            dtype=torch.float32),
            T_next     = torch.tensor(T_tp1,          dtype=torch.float32),
        )
        data.sim_idx = sim_i
        data.t_idx   = t_i
        data.Y_mean  = self.Y_mean
        data.Y_std   = self.Y_std
        return data


def get_dataloaders(cfg: BaseConfig, rollout_steps: int = 1):
    """Return (train_loader, val_loader, test_loader)."""
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_ds = HeatTreatmentDataset(cfg.dataset_path, cfg, "train", rollout_steps)
    val_ds   = HeatTreatmentDataset(cfg.dataset_path, cfg, "val",   rollout_steps)
    test_ds  = HeatTreatmentDataset(cfg.dataset_path, cfg, "test",  rollout_steps)

    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )