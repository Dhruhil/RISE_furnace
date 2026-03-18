"""
Dataset — Option A temporal split with DGL graph pre-building.

SPEED FIX: DGL graphs built ONCE in __init__, reused every batch.
Was: 3 seconds per batch = 151 min/epoch
Now: ~0.05 seconds per batch = ~3 min/epoch (60x faster)
"""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import h5py
import dgl

from configs.base_config import BaseConfig
from data.graph_builder import build_knn_graph


class HeatTreatmentDataset(Dataset):

    def __init__(
        self,
        h5_path:       str,
        cfg:           BaseConfig,
        split:         str = "train",
        rollout_steps: int = 1,
        split_mode:    str = "training",
    ):
        super().__init__()
        self.cfg           = cfg
        self.split         = split
        self.rollout_steps = rollout_steps
        self.split_mode    = split_mode

        with h5py.File(h5_path, "r") as f:
            self.X_raw      = f["X_raw"][:].astype(np.float32)
            self.Y_raw      = f["Y_raw"][:].astype(np.float32)
            self.X_mean     = f["X_mean"][:].astype(np.float32)
            self.X_std      = f["X_std"][:].astype(np.float32)
            self.Y_mean     = float(f["Y_mean"][()])
            self.Y_std      = float(f["Y_std"][()])
            sim_starts      = f["sim_start_indices"][:].tolist()
            sim_n_rows      = f["sim_n_rows"][:].tolist()
            n_sims          = int(f.attrs["n_simulations"])
            feature_cols    = json.loads(f.attrs["feature_cols"])

        self.col = {k: j for j, k in enumerate(feature_cols)}
        c        = self.col

        node_keys  = ["x", "y", "z", "T_set", "T_set", "cy", "cz", "kappa", "Cp", "rho"]
        self._nmu  = np.array([self.X_mean[c[k]] for k in node_keys], dtype=np.float32)
        self._nstd = np.array([self.X_std[c[k]]  for k in node_keys], dtype=np.float32)
        self._nmu[3]  = self.Y_mean
        self._nstd[3] = self.Y_std

        self._simulations: list[dict] = []
        for i in range(n_sims):
            start  = sim_starts[i]
            n_rows = sim_n_rows[i]
            X_sim  = self.X_raw[start: start + n_rows]
            Y_sim  = self.Y_raw[start: start + n_rows]
            t_vals   = X_sim[:, c["t"]]
            unique_t = np.unique(t_vals)
            n_times  = len(unique_t)
            n_cells  = n_rows // n_times
            self._simulations.append({
                "sim_idx": i,
                "coords":  X_sim[:n_cells, :3],
                "X_3d":    X_sim.reshape(n_times, n_cells, 15),
                "T_3d":    Y_sim.reshape(n_times, n_cells),
                "n_times": n_times,
                "n_cells": n_cells,
            })

        n_test  = max(1, int(n_sims * cfg.test_fraction))
        n_val   = max(1, int(n_sims * cfg.val_fraction))
        n_train = n_sims - n_val - n_test
        split_map = {
            "train": list(range(0,              n_train)),
            "val":   list(range(n_train,        n_train + n_val)),
            "test":  list(range(n_train + n_val, n_sims)),
        }
        self.sim_indices: list[int] = split_map[split]

        self._index: list[tuple[int, int]] = []
        for sim_i in self.sim_indices:
            n_t = self._simulations[sim_i]["n_times"] - rollout_steps
            if split_mode == "training":
                t_max = min(n_t, cfg.n_train_steps - rollout_steps)
            else:
                t_max = n_t
            for t_i in range(t_max):
                self._index.append((sim_i, t_i))

        # Pre-build PyG k-NN graphs
        self._graphs: dict[int, tuple] = {}
        for sim_i in self.sim_indices:
            coords = torch.tensor(
                self._simulations[sim_i]["coords"], dtype=torch.float32
            )
            edge_index, edge_attr = build_knn_graph(coords, cfg.graph_k_neighbors)
            self._graphs[sim_i] = (edge_index, edge_attr)

        # SPEED FIX: Pre-build DGL graphs ONCE here
        # Without this fix: dgl.graph() called 12122 times per epoch = 151 min
        # With this fix: dgl.graph() called 50 times total = done in seconds
        # Detect GPU availability
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dgl_graphs: dict[int, dgl.DGLGraph] = {}
        for sim_i in self.sim_indices:
            edge_index, _ = self._graphs[sim_i]
            n_nodes = self._simulations[sim_i]["n_cells"]
            g = dgl.graph(
                (edge_index[0].cpu(), edge_index[1].cpu()),
                num_nodes=n_nodes,
            ).to(_device)   # move to GPU once here — never again during training
            self._dgl_graphs[sim_i] = g
        print(
            f"  [{split:5s}|{split_mode:10s}]  "
            f"{len(self._index):>8,} pairs  "
            f"across {len(self.sim_indices):>2} sims  "
            f"t = 0–{(cfg.n_train_steps if split_mode=='training' else cfg.n_total_steps)*cfg.dt:.0f}s"
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Data:
        sim_i, t_i = self._index[idx]
        sim = self._simulations[sim_i]
        edge_index, edge_attr = self._graphs[sim_i]
        c   = self.col

        X_t   = sim["X_3d"][t_i]
        T_t   = sim["T_3d"][t_i]
        T_tp1 = sim["T_3d"][t_i + self.rollout_steps]

        node_feats = np.column_stack([
            X_t[:, c["x"]], X_t[:, c["y"]], X_t[:, c["z"]],
            T_t,
            X_t[:, c["T_set"]], X_t[:, c["cy"]], X_t[:, c["cz"]],
            X_t[:, c["kappa"]], X_t[:, c["Cp"]], X_t[:, c["rho"]],
        ]).astype(np.float32)

        node_norm    = (node_feats - self._nmu) / (self._nstd + 1e-8)
        delta_T      = (T_tp1 - T_t).reshape(-1, 1).astype(np.float32)
        delta_T_norm = delta_T / (self.Y_std + 1e-8)

        data = Data(
            x          = torch.tensor(node_norm,    dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr.float(),
            y          = torch.tensor(delta_T_norm, dtype=torch.float32),
            T_current  = torch.tensor(T_t,          dtype=torch.float32),
            T_next     = torch.tensor(T_tp1,        dtype=torch.float32),
            T_set_raw  = torch.tensor(X_t[:, c["T_set"]].astype(np.float32), dtype=torch.float32),
            kappa_raw  = torch.tensor(X_t[:, c["kappa"]].astype(np.float32), dtype=torch.float32),
            Cp_raw     = torch.tensor(X_t[:, c["Cp"]].astype(np.float32),    dtype=torch.float32),
            rho_raw    = torch.tensor(X_t[:, c["rho"]].astype(np.float32),   dtype=torch.float32),
            node_mean  = torch.tensor(self._nmu,    dtype=torch.float32),
            node_std   = torch.tensor(self._nstd,   dtype=torch.float32),
            Y_mean     = float(self.Y_mean),
            Y_std      = float(self.Y_std),
        )
        # Attach pre-built DGL graph — forward() uses this directly
        data.dgl_graph = self._dgl_graphs[sim_i]
        data.sim_idx   = sim_i
        data.t_idx     = t_i
        return data


def get_dataloaders(
    cfg:           BaseConfig,
    rollout_steps: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Alvis A100 — batch_size=8, num_workers=4 for speed
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_ds = HeatTreatmentDataset(
        cfg.dataset_path, cfg, "train", rollout_steps, split_mode="training"
    )
    val_ds = HeatTreatmentDataset(
        cfg.dataset_path, cfg, "val", rollout_steps, split_mode="training"
    )
    test_ds = HeatTreatmentDataset(
        cfg.dataset_path, cfg, "test", rollout_steps, split_mode="training"
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


def get_evaluation_dataset(cfg: BaseConfig) -> HeatTreatmentDataset:
    return HeatTreatmentDataset(
        cfg.dataset_path, cfg,
        split="test", rollout_steps=1, split_mode="evaluation",
    )