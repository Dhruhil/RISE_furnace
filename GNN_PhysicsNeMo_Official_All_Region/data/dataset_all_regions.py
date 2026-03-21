"""
Dataset for ALL regions — reads dataset_all_regions.h5
Built from local VTK extraction covering:
  steel_cylinder, inner_box, heater_1-8, brick_heater
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


REGION_IDS = {
    "steel_cylinder": 0,
    "inner_box":      1,
    "heater_1":       2,
    "heater_2":       3,
    "heater_3":       4,
    "heater_4":       5,
    "heater_5":       6,
    "heater_6":       7,
    "heater_7":       8,
    "heater_8":       9,
    "brick_heater":   10,
}


class AllRegionsDataset(Dataset):
    """
    Dataset that includes all furnace regions.
    Each sample = one timestep of one region of one case.
    Node features: x, y, z, T_current, T_set, region_id, time
    Target: T at next timestep
    """

    def __init__(
        self,
        h5_path:    str,
        cfg:        BaseConfig,
        split:      str = "train",
        split_mode: str = "training",
    ):
        super().__init__()
        self.cfg        = cfg
        self.split      = split
        self.split_mode = split_mode

        self._simulations: list[dict] = []

        with h5py.File(h5_path, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])

            for ci in range(n_cases):
                grp  = f[f"case_{ci:03d}"]
                name = grp.attrs["name"]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)
                n_times = len(times)

                region_data = {}
                for region in regions:
                    if region not in grp:
                        continue
                    coords  = grp[region]["coords"][:].astype(np.float32)
                    T_array = grp[region]["T"][:].astype(np.float32)
                    region_data[region] = {
                        "coords":  coords,
                        "T_array": T_array,
                        "region_id": REGION_IDS.get(region, 0),
                    }

                # Compute actual T range per region for correct rollout clipping
                region_T_max = {}
                region_T_min = {}
                for reg, rdat in region_data.items():
                    region_T_max[reg] = float(rdat["T_array"].max()) * 1.05
                    region_T_min[reg] = float(rdat["T_array"].min()) * 0.95

                self._simulations.append({
                    "case_idx":    ci,
                    "name":        name,
                    "T_set":       T_set,
                    "times":       times,
                    "n_times":     n_times,
                    "region_data":  region_data,
                    "region_T_max": region_T_max,
                    "region_T_min": region_T_min,
                })

        n_sims  = len(self._simulations)
        n_test  = max(1, int(n_sims * cfg.test_fraction))
        n_val   = max(1, int(n_sims * cfg.val_fraction))
        n_train = n_sims - n_val - n_test

        split_map = {
            "train": list(range(0,              n_train)),
            "val":   list(range(n_train,        n_train + n_val)),
            "test":  list(range(n_train + n_val, n_sims)),
        }
        self.sim_indices = split_map[split]

        # Compute normalisation from training sims
        all_T, all_dT = [], []
        for i in range(n_train):
            for region, rdata in self._simulations[i]["region_data"].items():
                T = rdata["T_array"]
                all_T.append(T.ravel())
                all_dT.append(np.diff(T, axis=0)[20:].ravel())

        all_T  = np.concatenate(all_T).astype(np.float64)
        all_dT = np.concatenate(all_dT).astype(np.float64)

        self.T_mean  = float(all_T.mean())
        self.T_std   = float(all_T.std()) + 1e-8
        self.dT_mean = float(all_dT.mean())
        self.dT_std  = float(all_dT.std()) + 1e-8

        print(f"  T_mean={self.T_mean:.1f}K  T_std={self.T_std:.1f}K")
        print(f"  dT_mean={self.dT_mean:.5f}K  dT_std={self.dT_std:.4f}K")

        # Build index: (sim_i, region, t_i)
        self._index: list[tuple] = []
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            n_t = sim["n_times"] - 1
            if split_mode == "training":
                t_max = min(n_t, cfg.n_train_steps)
            else:
                t_max = n_t
            for region in sim["region_data"]:
                for t_i in range(20, t_max):
                    self._index.append((sim_i, region, t_i))

        # Build graphs per sim per region
        print(f"  Building graphs for {len(self.sim_indices)} sims x {len(list(self._simulations[self.sim_indices[0]]['region_data'].keys()))} regions...", flush=True)
        self._graphs: dict = {}
        for gi, sim_i in enumerate(self.sim_indices):
            sim = self._simulations[sim_i]
            self._graphs[sim_i] = {}
            for region, rdata in sim["region_data"].items():
                coords = torch.tensor(rdata["coords"], dtype=torch.float32)
                ei, ea = build_knn_graph(coords, cfg.graph_k_neighbors)
                self._graphs[sim_i][region] = (ei, ea)
            if (gi + 1) % 10 == 0 or gi == 0:
                print(f"  Graphs built: {gi+1}/{len(self.sim_indices)}", flush=True)

        print(f"  [{split:5s}|{split_mode:10s}] "
              f"{len(self._index):>8,} pairs "
              f"across {len(self.sim_indices):>2} sims "
              f"all regions")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Data:
        sim_i, region, t_i = self._index[idx]
        sim   = self._simulations[sim_i]
        rdata = sim["region_data"][region]
        edge_index, edge_attr = self._graphs[sim_i][region]

        coords    = rdata["coords"]
        T_t       = rdata["T_array"][t_i]
        T_tp1     = rdata["T_array"][t_i + 1]
        T_set     = sim["T_set"]
        region_id = rdata["region_id"]
        t_val     = sim["times"][t_i]

        # Node features: x, y, z, T_current_norm, T_set_norm, region_id, time_norm
        T_norm    = (T_t   - self.T_mean) / (self.T_std + 1e-8)
        Tset_norm = (T_set - self.T_mean) / (self.T_std + 1e-8)
        t_norm    = t_val / 4000.0

        node_feats = np.column_stack([
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            T_norm,
            np.full(len(T_t), Tset_norm,   dtype=np.float32),
            np.full(len(T_t), region_id/10, dtype=np.float32),
            np.full(len(T_t), t_norm,       dtype=np.float32),
        ]).astype(np.float32)

        delta_T      = (T_tp1 - T_t).reshape(-1, 1)
        delta_T_norm = ((delta_T - self.dT_mean) /
                        (self.dT_std + 1e-8)).astype(np.float32)

        # Add small noise during training
        if self.split == "train":
            noise = np.random.normal(0, 0.02, size=node_feats.shape[0])
            node_feats[:, 3] += noise.astype(np.float32)

        data = Data(
            x          = torch.tensor(node_feats,    dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr.float(),
            y          = torch.tensor(delta_T_norm,  dtype=torch.float32),
            T_current  = torch.tensor(T_t,           dtype=torch.float32),
            T_next     = torch.tensor(T_tp1,         dtype=torch.float32),
            T_set_raw  = torch.tensor(
                np.full(len(T_t), T_set, dtype=np.float32),
                dtype=torch.float32),
            Y_std      = float(self.T_std),
            dT_mean    = float(self.dT_mean),
            dT_std     = float(self.dT_std),
        )
        data.sim_idx   = sim_i
        data.region    = region
        data.region_id = region_id
        data.t_idx     = t_i
        return data


def get_all_regions_dataloaders(cfg: BaseConfig):
    kw = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=False)
    train_ds = AllRegionsDataset(
        cfg.all_regions_dataset_path, cfg, "train", "training")
    val_ds = AllRegionsDataset(
        cfg.all_regions_dataset_path, cfg, "val", "training")
    test_ds = AllRegionsDataset(
        cfg.all_regions_dataset_path, cfg, "test", "training")
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )
