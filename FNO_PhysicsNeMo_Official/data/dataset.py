"""
FNO Dataset — reads dataset_all_regions.h5 (same as GNN All Regions).
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

HDF5 structure:
    attrs: n_cases, regions (JSON list)
    case_000/
        attrs: name, T_set
        times: (n_times,)
        steel_cylinder/coords: (n_cells, 3), steel_cylinder/T: (n_times, n_cells)
        inner_box/coords, inner_box/T, heater_1/..., brick_heater/...
    case_001/ ...

FNO reshaping:
    Input:  (batch, 4, n_cells) — [T_now_norm, T_set_norm, region_id/10, time/4000]
    Output: (batch, 1, n_cells) — [T_next_norm]

Each sample = one region at one timestep transition of one case.
"""
from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

REGION_IDS = {
    "steel_cylinder": 0, "inner_box": 1,
    "heater_1": 2, "heater_2": 3, "heater_3": 4, "heater_4": 5,
    "heater_5": 6, "heater_6": 7, "heater_7": 8, "heater_8": 9,
    "brick_heater": 10,
}


class FNOAllRegionsDataset(Dataset):
    """
    1D FNO dataset for heat treatment — all furnace regions.

    Each sample returns:
        x:      (4, n_cells)  input channels
        y:      (1, n_cells)  target T_next normalised
        T_cur:  (n_cells,)    raw T at current step [K]
        T_next: (n_cells,)    raw T at next step [K]
        sim_i:  int           simulation index
        region: str           region name
        rid:    int           region ID
        t_i:    int           timestep index
    """

    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.cfg        = cfg
        self.split      = split
        self.split_mode = split_mode

        # ── Load all data from HDF5 ──────────────────────────────────
        self._simulations = []

        with h5py.File(h5_path, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])

            for ci in range(n_cases):
                grp     = f[f"case_{ci:03d}"]
                name    = grp.attrs["name"]
                T_set   = float(grp.attrs["T_set"])
                times   = grp["times"][:].astype(np.float32)
                n_times = len(times)

                region_data = {}
                for region in regions:
                    if region not in grp:
                        continue
                    coords  = grp[region]["coords"][:].astype(np.float32)
                    T_array = grp[region]["T"][:].astype(np.float32)
                    region_data[region] = {
                        "coords":    coords,
                        "T_array":   T_array,
                        "n_cells":   coords.shape[0],
                        "region_id": REGION_IDS.get(region, 0),
                    }

                # Per-region T bounds for rollout clipping
                region_T_max, region_T_min = {}, {}
                for reg, rdat in region_data.items():
                    region_T_max[reg] = float(rdat["T_array"].max()) * 1.05
                    region_T_min[reg] = float(rdat["T_array"].min()) * 0.95

                self._simulations.append({
                    "case_idx":     ci,
                    "name":         name,
                    "T_set":        T_set,
                    "times":        times,
                    "n_times":      n_times,
                    "region_data":  region_data,
                    "region_T_max": region_T_max,
                    "region_T_min": region_T_min,
                })

        # ── Train/val/test split by case ─────────────────────────────
        n_sims  = len(self._simulations)
        n_test  = max(1, int(n_sims * cfg.test_fraction))
        n_val   = max(1, int(n_sims * cfg.val_fraction))
        n_train = n_sims - n_val - n_test

        split_map = {
            "train": list(range(0, n_train)),
            "val":   list(range(n_train, n_train + n_val)),
            "test":  list(range(n_train + n_val, n_sims)),
        }
        self.sim_indices = split_map[split]

        # ── Normalisation from training cases only ───────────────────
        all_T = []
        for i in range(n_train):
            for rdata in self._simulations[i]["region_data"].values():
                all_T.append(rdata["T_array"].ravel())
        all_T = np.concatenate(all_T).astype(np.float64)

        self.T_mean = float(all_T.mean())
        self.T_std  = float(all_T.std()) + 1e-8

        all_Tset     = [self._simulations[i]["T_set"] for i in range(n_train)]
        self.Tset_mean = float(np.mean(all_Tset))
        self.Tset_std  = float(np.std(all_Tset)) + 1e-8

        print(f"  FNO dataset: T_mean={self.T_mean:.1f}K  T_std={self.T_std:.1f}K")
        print(f"               Tset_mean={self.Tset_mean:.1f}K  Tset_std={self.Tset_std:.1f}K")

        # ── Build sample index: (sim_i, region, t_i) ────────────────
        self._index = []
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            n_t = sim["n_times"] - 1
            t_max = min(n_t, cfg.n_train_steps) if split_mode == "training" else n_t
            for region in sim["region_data"]:
                for t_i in range(20, t_max):
                    self._index.append((sim_i, region, t_i))

        print(f"  FNO [{split:5s}|{split_mode:10s}] "
              f"{len(self._index):>8,} samples | "
              f"{len(self.sim_indices)} cases | all regions")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sim_i, region, t_i = self._index[idx]
        sim   = self._simulations[sim_i]
        rdata = sim["region_data"][region]

        T_t       = rdata["T_array"][t_i]
        T_tp1     = rdata["T_array"][t_i + 1]
        T_set     = sim["T_set"]
        region_id = rdata["region_id"]
        t_val     = sim["times"][t_i]
        n_cells   = rdata["n_cells"]

        # Build 4-channel input: (4, n_cells)
        T_norm    = ((T_t - self.T_mean) / self.T_std).astype(np.float32)
        Tset_norm = float((T_set - self.Tset_mean) / self.Tset_std)
        rid_norm  = float(region_id / 10.0)
        t_norm    = float(t_val / 4000.0)

        x = np.stack([
            T_norm,
            np.full(n_cells, Tset_norm, dtype=np.float32),
            np.full(n_cells, rid_norm,  dtype=np.float32),
            np.full(n_cells, t_norm,    dtype=np.float32),
        ], axis=0).astype(np.float32)

        # Target: normalised T_next
        T_next_norm = ((T_tp1 - self.T_mean) / self.T_std).astype(np.float32)
        y = T_next_norm.reshape(1, -1)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(T_t, dtype=torch.float32),
            torch.tensor(T_tp1, dtype=torch.float32),
            sim_i, region, region_id, t_i,
        )


def _collate_variable_size(batch):
    """Custom collate: pad regions with different n_cells to max in batch."""
    max_cells = max(b[0].shape[1] for b in batch)
    xs, ys, T_curs, T_nexts = [], [], [], []
    sim_is, regions, rids, t_is = [], [], [], []

    for x, y, T_cur, T_next, si, reg, rid, ti in batch:
        nc = x.shape[1]
        if nc < max_cells:
            pad = max_cells - nc
            x      = torch.nn.functional.pad(x,      (0, pad), value=0)
            y      = torch.nn.functional.pad(y,      (0, pad), value=0)
            T_cur  = torch.nn.functional.pad(T_cur,  (0, pad), value=0)
            T_next = torch.nn.functional.pad(T_next, (0, pad), value=0)
        xs.append(x); ys.append(y)
        T_curs.append(T_cur); T_nexts.append(T_next)
        sim_is.append(si); regions.append(reg)
        rids.append(rid); t_is.append(ti)

    return (
        torch.stack(xs), torch.stack(ys),
        torch.stack(T_curs), torch.stack(T_nexts),
        sim_is, regions, rids, t_is,
    )


def get_fno_dataloaders(cfg):
    """Create train/val/test DataLoaders."""
    kw = dict(num_workers=0, pin_memory=False, collate_fn=_collate_variable_size)
    train_ds = FNOAllRegionsDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds   = FNOAllRegionsDataset(cfg.dataset_path, cfg, "val",   "training")
    test_ds  = FNOAllRegionsDataset(cfg.dataset_path, cfg, "test",  "training")
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, **kw),
    )


def get_fno_eval_dataset(cfg):
    """Create evaluation dataset with full time range."""
    return FNOAllRegionsDataset(cfg.dataset_path, cfg, "test", "evaluation")
