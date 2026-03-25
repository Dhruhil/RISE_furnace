"""
PINN Dataset — reads dataset_all_regions.h5 (same as GNN/FNO).
Master's Thesis: Digital Twin Modeling of Heat Treatment in Cast Metals

Unlike GNN/FNO which predict T(t+1) from T(t), the PINN learns:
    T = f(x, y, z, t, T_set, region_id)

The neural network directly maps space-time coordinates to temperature.
Physics is enforced through PDE residual loss (autograd).
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
    "brick_heater": 10, "outer_box": 11,
}


class PINNAllRegionsDataset(Dataset):
    """
    PINN dataset for heat treatment — all furnace regions.

    Each sample:
        inputs:  (6,) → [x, y, z, t, T_set, region_id]  (normalised)
        target:  (1,) → [T]  (normalised)
        T_raw:   (1,) → [T]  (Kelvin, for metrics)
    """

    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        # ── Load HDF5 ────────────────────────────────────────────────
        simulations = []
        with h5py.File(h5_path, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])

            for ci in range(n_cases):
                grp = f[f"case_{ci:03d}"]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)
                n_times = len(times)

                for region in regions:
                    if region not in grp:
                        continue
                    coords = grp[region]["coords"][:].astype(np.float32)
                    T_array = grp[region]["T"][:].astype(np.float32)
                    rid = REGION_IDS.get(region, 0)
                    n_cells = coords.shape[0]

                    simulations.append({
                        "case_idx": ci,
                        "T_set": T_set,
                        "times": times,
                        "n_times": n_times,
                        "coords": coords,
                        "T_array": T_array,
                        "region": region,
                        "region_id": rid,
                        "n_cells": n_cells,
                    })

        # ── Split by case (same fractions as GNN/FNO) ────────────────
        case_indices = sorted(set(s["case_idx"] for s in simulations))
        n_cases_total = len(case_indices)
        n_test = max(1, int(n_cases_total * cfg.test_fraction))
        n_val = max(1, int(n_cases_total * cfg.val_fraction))
        n_train = n_cases_total - n_val - n_test

        split_map = {
            "train": set(case_indices[:n_train]),
            "val": set(case_indices[n_train:n_train + n_val]),
            "test": set(case_indices[n_train + n_val:]),
        }
        my_cases = split_map[split]

        # ── Build flat arrays: (x, y, z, t, T_set, region_id) → T ───
        all_x, all_y, all_z = [], [], []
        all_t, all_Tset, all_rid = [], [], []
        all_T = []

        for sim in simulations:
            if sim["case_idx"] not in my_cases:
                continue

            coords = sim["coords"]
            T_arr = sim["T_array"]
            times = sim["times"]
            n_cells = sim["n_cells"]
            n_times = sim["n_times"]
            T_set = sim["T_set"]
            rid = sim["region_id"]

            # Time range based on split_mode
            if split_mode == "training":
                t_max = min(n_times, cfg.n_train_steps)
            else:
                t_max = n_times

            # Skip first 20 timesteps (initial transient)
            for ti in range(20, t_max):
                all_x.append(coords[:, 0])
                all_y.append(coords[:, 1])
                all_z.append(coords[:, 2])
                all_t.append(np.full(n_cells, times[ti], dtype=np.float32))
                all_Tset.append(np.full(n_cells, T_set, dtype=np.float32))
                all_rid.append(np.full(n_cells, rid, dtype=np.float32))
                all_T.append(T_arr[ti])

        # Concatenate
        self.x_raw = np.concatenate(all_x)
        self.y_raw = np.concatenate(all_y)
        self.z_raw = np.concatenate(all_z)
        self.t_raw = np.concatenate(all_t)
        self.Tset_raw = np.concatenate(all_Tset)
        self.rid_raw = np.concatenate(all_rid)
        self.T_raw = np.concatenate(all_T)

        # ── Normalisation (mean/std from training split) ──────────────
        if split == "train":
            self.x_mean, self.x_std = float(self.x_raw.mean()), float(self.x_raw.std()) + 1e-8
            self.y_mean, self.y_std = float(self.y_raw.mean()), float(self.y_raw.std()) + 1e-8
            self.z_mean, self.z_std = float(self.z_raw.mean()), float(self.z_raw.std()) + 1e-8
            self.t_mean, self.t_std = float(self.t_raw.mean()), float(self.t_raw.std()) + 1e-8
            self.Tset_mean, self.Tset_std = float(self.Tset_raw.mean()), float(self.Tset_raw.std()) + 1e-8
            self.T_mean, self.T_std = float(self.T_raw.mean()), float(self.T_raw.std()) + 1e-8

        self.n_samples = len(self.T_raw)

        # Subsample if too many points (PINN doesn't need 155M points)
        max_points = 5_000_000 if split == "train" else 2_000_000
        if self.n_samples > max_points:
            print(f"    Subsampling {self.n_samples:,} → {max_points:,} points")
            idx = np.random.RandomState(42).permutation(self.n_samples)[:max_points]
            self.x_raw = self.x_raw[idx]
            self.y_raw = self.y_raw[idx]
            self.z_raw = self.z_raw[idx]
            self.t_raw = self.t_raw[idx]
            self.Tset_raw = self.Tset_raw[idx]
            self.rid_raw = self.rid_raw[idx]
            self.T_raw = self.T_raw[idx]

        self.n_samples = len(self.T_raw)
        print(f"  PINN [{split:5s}|{split_mode:10s}] {self.n_samples:>10,} points | "
              f"{len(my_cases)} cases | all regions")

    def set_norm_stats(self, train_ds):
        """Copy normalisation stats from training dataset."""
        for attr in ["x_mean", "x_std", "y_mean", "y_std", "z_mean", "z_std",
                      "t_mean", "t_std", "Tset_mean", "Tset_std", "T_mean", "T_std"]:
            setattr(self, attr, getattr(train_ds, attr))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Normalise inputs
        x_n = (self.x_raw[idx] - self.x_mean) / self.x_std
        y_n = (self.y_raw[idx] - self.y_mean) / self.y_std
        z_n = (self.z_raw[idx] - self.z_mean) / self.z_std
        t_n = (self.t_raw[idx] - self.t_mean) / self.t_std
        Tset_n = (self.Tset_raw[idx] - self.Tset_mean) / self.Tset_std
        rid_n = self.rid_raw[idx] / 11.0  # same as GNN/FNO

        inputs = np.array([x_n, y_n, z_n, t_n, Tset_n, rid_n], dtype=np.float32)

        # Normalise target
        T_n = (self.T_raw[idx] - self.T_mean) / self.T_std

        return (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor([T_n], dtype=torch.float32),
            torch.tensor([self.T_raw[idx]], dtype=torch.float32),  # raw K
        )


def get_pinn_dataloaders(cfg):
    """Create train/val/test DataLoaders."""
    train_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "val", "training")
    test_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "test", "evaluation")

    # Share normalisation stats
    val_ds.set_norm_stats(train_ds)
    test_ds.set_norm_stats(train_ds)

    kw = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


def get_pinn_eval_dataset(cfg):
    """Test dataset with full time range for evaluation."""
    train_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "train", "training")
    test_ds = PINNAllRegionsDataset(cfg.dataset_path, cfg, "test", "evaluation")
    test_ds.set_norm_stats(train_ds)
    return test_ds
