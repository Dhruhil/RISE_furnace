"""
Unified Multi-Region Dataset for GNN.
All 12 regions combined into ONE graph per (sim, timestep).
Inter-region boundary edges connect nodes from different regions
that are physically close (within threshold distance).
"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import h5py

REGION_IDS = {
    "steel_cylinder": 0, "inner_box": 1, "outer_box": 2,
    "heater_1": 3, "heater_2": 4, "heater_3": 5, "heater_4": 6,
    "heater_5": 7, "heater_6": 8, "heater_7": 9, "heater_8": 10,
    "brick_heater": 11,
}

HEATER_REGIONS = {
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
    "brick_heater",
}


class UnifiedDataset(torch.utils.data.Dataset):
    """
    Each sample = ALL regions at one timestep of one simulation.
    Graph: 13,648 nodes, KNN intra-region + boundary inter-region edges.
    """

    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.split = split
        self.split_mode = split_mode
        self.cfg = cfg

        # Load all simulations
        self._simulations = []
        with h5py.File(h5_path, "r") as f:
            regions = json.loads(f.attrs["regions"])
            for sim_key in sorted(f.keys()):
                grp = f[sim_key]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)

                region_data = {}
                all_coords = []
                all_region_ids = []
                node_offsets = {}
                offset = 0

                for region in regions:
                    if region not in grp:
                        continue
                    coords = grp[region]["coords"][:].astype(np.float32)
                    T_array = grp[region]["T"][:].astype(np.float32)
                    n_nodes = coords.shape[0]
                    rid = REGION_IDS.get(region, 0)

                    region_data[region] = {
                        "coords": coords,
                        "T_array": T_array,
                        "n_cells": n_nodes,
                        "region_id": rid,
                        "offset": offset,
                    }
                    all_coords.append(coords)
                    all_region_ids.append(np.full(n_nodes, rid, dtype=np.int32))
                    node_offsets[region] = offset
                    offset += n_nodes

                self._simulations.append({
                    "T_set": T_set,
                    "times": times,
                    "n_times": len(times),
                    "region_data": region_data,
                    "all_coords": np.concatenate(all_coords),
                    "all_region_ids": np.concatenate(all_region_ids),
                    "total_nodes": offset,
                })

        n_sims = len(self._simulations)
        n_test = max(1, int(n_sims * cfg.test_fraction))
        n_val = max(1, int(n_sims * cfg.val_fraction))
        n_train = n_sims - n_val - n_test

        split_map = {
            "train": list(range(0, n_train)),
            "val": list(range(n_train, n_train + n_val)),
            "test": list(range(n_train + n_val, n_sims)),
        }
        self.sim_indices = split_map[split]

        # Normalisation from training data
        all_T, all_dT = [], []
        for i in range(n_train):
            for rdata in self._simulations[i]["region_data"].values():
                all_T.append(rdata["T_array"].ravel())
                dT = np.diff(rdata["T_array"], axis=0)[20:].ravel()
                all_dT.append(dT)

        all_T = np.concatenate(all_T).astype(np.float64)
        all_dT = np.concatenate(all_dT).astype(np.float64)
        self.T_mean = float(all_T.mean())
        self.T_std = float(all_T.std()) + 1e-8
        self.dT_mean = float(np.mean(all_dT))
        self.dT_std = float(np.std(all_dT)) + 1e-8

        print(f"  Unified: T_mean={self.T_mean:.1f}K  T_std={self.T_std:.1f}K")
        print(f"           dT_mean={self.dT_mean:.5f}K  dT_std={self.dT_std:.4f}K")

        # Build sample index: (sim_i, t_i) — no region dimension!
        self._index = []
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            n_t = sim["n_times"] - 1
            t_max = min(n_t, cfg.n_train_steps) if split_mode == "training" else n_t
            for t_i in range(20, t_max - 2):  # -2 for pushforward
                self._index.append((sim_i, t_i))

        # Build unified graphs (once per simulation)
        print(f"  Building unified graphs for {len(self.sim_indices)} sims...")
        self._graphs = {}
        k_intra = cfg.graph_k_neighbors  # KNN within region
        boundary_dist = 0.02  # 2cm threshold for inter-region edges

        for gi, sim_i in enumerate(self.sim_indices):
            sim = self._simulations[sim_i]
            all_coords = sim["all_coords"]
            total = sim["total_nodes"]

            # Intra-region edges (KNN within each region)
            src_list, dst_list, eattr_list = [], [], []
            for region, rdata in sim["region_data"].items():
                coords = rdata["coords"]
                offset = rdata["offset"]
                n = rdata["n_cells"]
                if n < 2:
                    continue
                k = min(k_intra, n - 1)
                tree = cKDTree(coords)
                dists, idxs = tree.query(coords, k=k + 1)
                for i in range(n):
                    for j_local in range(1, k + 1):
                        j = idxs[i, j_local]
                        src_list.append(offset + i)
                        dst_list.append(offset + j)
                        diff = coords[j] - coords[i]
                        dist = np.linalg.norm(diff)
                        eattr_list.append([diff[0], diff[1], diff[2], dist])

            # Inter-region boundary edges (connect close nodes across regions)
            region_list = list(sim["region_data"].keys())
            for r1_idx in range(len(region_list)):
                for r2_idx in range(r1_idx + 1, len(region_list)):
                    r1 = region_list[r1_idx]
                    r2 = region_list[r2_idx]
                    c1 = sim["region_data"][r1]["coords"]
                    c2 = sim["region_data"][r2]["coords"]
                    o1 = sim["region_data"][r1]["offset"]
                    o2 = sim["region_data"][r2]["offset"]

                    tree2 = cKDTree(c2)
                    dists, idxs = tree2.query(c1, k=1)
                    mask = dists < boundary_dist

                    for i in np.where(mask)[0]:
                        j = idxs[i]
                        diff = c2[j] - c1[i]
                        dist = dists[i]
                        # Bidirectional edges
                        src_list.extend([o1 + i, o2 + j])
                        dst_list.extend([o2 + j, o1 + i])
                        eattr_list.extend([
                            [diff[0], diff[1], diff[2], dist],
                            [-diff[0], -diff[1], -diff[2], dist],
                        ])

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(eattr_list, dtype=torch.float32)
            self._graphs[sim_i] = (edge_index, edge_attr)

            n_boundary = sum(1 for s, d in zip(src_list, dst_list)
                           if sim["all_region_ids"][s] != sim["all_region_ids"][d])
            if gi == 0:
                print(f"    Sim {sim_i}: {total} nodes, "
                      f"{len(src_list)} edges ({n_boundary} cross-region)")

        print(f"  [{split:5s}|{split_mode:10s}] "
              f"{len(self._index):>6,} samples | "
              f"{len(self.sim_indices)} sims | unified graph")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sim_i, t_i = self._index[idx]
        sim = self._simulations[sim_i]
        edge_index, edge_attr = self._graphs[sim_i]

        all_coords = sim["all_coords"]
        all_rids = sim["all_region_ids"]
        total = sim["total_nodes"]
        T_set = sim["T_set"]
        t_val = sim["times"][t_i]

        # Collect T at t, t+1, t+2, t+3 for all nodes
        T_t = np.zeros(total, dtype=np.float32)
        T_tp1 = np.zeros(total, dtype=np.float32)
        T_tp2 = np.zeros(total, dtype=np.float32)
        T_tp3 = np.zeros(total, dtype=np.float32)

        for region, rdata in sim["region_data"].items():
            o = rdata["offset"]
            n = rdata["n_cells"]
            T_t[o:o+n] = rdata["T_array"][t_i]
            T_tp1[o:o+n] = rdata["T_array"][t_i + 1]
            T_tp2[o:o+n] = rdata["T_array"][t_i + 2]
            T_tp3[o:o+n] = rdata["T_array"][t_i + 3]

        # Node features
        T_norm = (T_t - self.T_mean) / self.T_std
        Tset_norm = (T_set - self.T_mean) / self.T_std
        t_norm = t_val / 4000.0

        node_feats = np.column_stack([
            all_coords[:, 0],
            all_coords[:, 1],
            all_coords[:, 2],
            T_norm,
            np.full(total, Tset_norm, dtype=np.float32),
            all_rids / 11.0,
            np.full(total, t_norm, dtype=np.float32),
        ]).astype(np.float32)

        # Targets: delta_T normalised
        dT1 = ((T_tp1 - T_t - self.dT_mean) / self.dT_std).reshape(-1, 1).astype(np.float32)
        dT2 = ((T_tp2 - T_tp1 - self.dT_mean) / self.dT_std).reshape(-1, 1).astype(np.float32)
        dT3 = ((T_tp3 - T_tp2 - self.dT_mean) / self.dT_std).reshape(-1, 1).astype(np.float32)

        # Add noise during training
        if self.split == "train":
            noise = np.random.normal(0, 0.02, size=total).astype(np.float32)
            node_feats[:, 3] += noise

        # Region mask for heaters (clamped to T_set, not predicted)
        is_heater = np.zeros(total, dtype=np.float32)
        for region, rdata in sim["region_data"].items():
            if region in HEATER_REGIONS:
                o = rdata["offset"]
                n = rdata["n_cells"]
                is_heater[o:o+n] = 1.0

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr.float(),
            y=torch.tensor(dT1, dtype=torch.float32),
            y2=torch.tensor(dT2, dtype=torch.float32),
            y3=torch.tensor(dT3, dtype=torch.float32),
            T_current=torch.tensor(T_t, dtype=torch.float32),
            T_next=torch.tensor(T_tp1, dtype=torch.float32),
            T_tp2=torch.tensor(T_tp2, dtype=torch.float32),
            T_tp3=torch.tensor(T_tp3, dtype=torch.float32),
            T_set_raw=torch.tensor(
                np.full(total, T_set, dtype=np.float32), dtype=torch.float32),
            is_heater=torch.tensor(is_heater, dtype=torch.float32),
            region_ids=torch.tensor(all_rids, dtype=torch.long),
            Y_std=float(self.T_std),
            dT_mean=float(self.dT_mean),
            dT_std=float(self.dT_std),
        )
        data.sim_idx = sim_i
        data.t_idx = t_i
        return data
