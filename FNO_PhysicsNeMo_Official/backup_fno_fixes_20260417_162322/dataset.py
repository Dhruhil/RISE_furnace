"""
3D FNO Dataset — interpolates all regions onto a regular 3D grid.
Same dataset_all_regions.h5 as GNN, no new data needed.

Each sample = ALL regions at one timestep → one 3D volume.
Input:  (C, Gx, Gy, Gz) with C = 19 channels
Output: (1, Gx, Gy, Gz) = normalised delta_T on grid
"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import NearestNDInterpolator
import h5py

REGION_IDS = {
    "steel_cylinder": 0, "inner_box": 1,
    "heater_1": 2, "heater_2": 3, "heater_3": 4, "heater_4": 5,
    "heater_5": 6, "heater_6": 7, "heater_7": 8, "heater_8": 9,
    "brick_heater": 10, "outer_box": 11,
}

HEATER_REGIONS = {
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
    "brick_heater",
}

# Material properties per region (approximate)
REGION_PROPERTIES = {
    "steel_cylinder": {"kappa": 60.0, "Cp": 450.0, "rho": 7800.0},
    "inner_box":      {"kappa": 0.026, "Cp": 1005.0, "rho": 1.2},
    "outer_box":      {"kappa": 1.5, "Cp": 900.0, "rho": 1800.0},
    "brick_heater":   {"kappa": 1.5, "Cp": 900.0, "rho": 1800.0},
}
# Heaters: same as brick
for i in range(1, 9):
    REGION_PROPERTIES[f"heater_{i}"] = {"kappa": 15.0, "Cp": 500.0, "rho": 2400.0}


class FNO3DDataset(Dataset):
    """
    3D grid dataset: all regions interpolated onto (Gx, Gy, Gz).

    Channels (8 total):
      [0]  T_norm        current temperature
      [1]  T_set_norm    furnace setpoint
      [2]  region_id/11  region encoding (0=steel, 11=outer_box)
      [3]  time/4000     normalised time
      [4]  is_heater     binary heater flag
      [5]  kappa/100     thermal conductivity (W/mK)
      [6]  Cp/1000       heat capacity (J/kgK)
      [7]  rho/10000     density (kg/m3)
    """

    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.split_mode = split_mode

        # Build regular 3D grid
        self.gx = np.linspace(cfg.x_min, cfg.x_max, cfg.grid_x).astype(np.float32)
        self.gy = np.linspace(cfg.y_min, cfg.y_max, cfg.grid_y).astype(np.float32)
        self.gz = np.linspace(cfg.z_min, cfg.z_max, cfg.grid_z).astype(np.float32)
        self.grid_points = np.stack(
            np.meshgrid(self.gx, self.gy, self.gz, indexing='ij'), axis=-1
        ).reshape(-1, 3)  # (Gx*Gy*Gz, 3)
        self.grid_shape = (cfg.grid_x, cfg.grid_y, cfg.grid_z)
        self.n_grid = cfg.grid_x * cfg.grid_y * cfg.grid_z

        # Load simulations
        self._simulations = []
        with h5py.File(h5_path, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])

            for ci in range(n_cases):
                grp = f[f"case_{ci:03d}"]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)

                # Collect ALL cells from all regions
                all_coords = []
                all_region_ids = []
                all_is_heater = []
                all_kappa = []
                all_Cp = []
                all_rho = []
                region_slices = {}
                offset = 0

                for region in regions:
                    if region not in grp:
                        continue
                    coords = grp[region]["coords"][:].astype(np.float32)
                    n = coords.shape[0]
                    rid = REGION_IDS.get(region, 0)
                    props = REGION_PROPERTIES.get(region, {"kappa": 1.0, "Cp": 1.0, "rho": 1.0})

                    all_coords.append(coords)
                    all_region_ids.append(np.full(n, rid, dtype=np.int32))
                    all_is_heater.append(np.full(n, 1.0 if region in HEATER_REGIONS else 0.0, dtype=np.float32))
                    all_kappa.append(np.full(n, props["kappa"], dtype=np.float32))
                    all_Cp.append(np.full(n, props["Cp"], dtype=np.float32))
                    all_rho.append(np.full(n, props["rho"], dtype=np.float32))

                    region_slices[region] = (offset, offset + n)
                    offset += n

                all_coords = np.concatenate(all_coords)
                all_region_ids = np.concatenate(all_region_ids)
                all_is_heater = np.concatenate(all_is_heater)
                all_kappa = np.concatenate(all_kappa)
                all_Cp = np.concatenate(all_Cp)
                all_rho = np.concatenate(all_rho)

                # Build T_array for all cells: (n_times, total_cells)
                T_all = np.zeros((len(times), offset), dtype=np.float32)
                for region in regions:
                    if region not in grp:
                        continue
                    s, e = region_slices[region]
                    T_all[:, s:e] = grp[region]["T"][:].astype(np.float32)

                # Build interpolators for static fields (once per sim)
                # Region one-hot: 12 binary channels
                region_onehot = np.zeros((offset, 12), dtype=np.float32)
                for j in range(offset):
                    region_onehot[j, all_region_ids[j]] = 1.0

                # Cylinder params from h5 attrs
                cy_val = float(grp.attrs.get("cy", 0.18))
                kappa_val = float(grp.attrs.get("kappa", 60.0))
                Cp_val = float(grp.attrs.get("Cp", 450.0))
                rho_val = float(grp.attrs.get("rho", 7800.0))

                # Override steel properties with per-sim values
                for region in regions:
                    if region == "steel_cylinder" and region in region_slices:
                        s, e = region_slices[region]
                        all_kappa[s:e] = kappa_val
                        all_Cp[s:e] = Cp_val
                        all_rho[s:e] = rho_val

                self._simulations.append({
                    "T_set": T_set,
                    "times": times,
                    "n_times": len(times),
                    "coords": all_coords,
                    "T_all": T_all,
                    "region_onehot": region_onehot,
                    "is_heater": all_is_heater,
                    "kappa": all_kappa,
                    "Cp": all_Cp,
                    "rho": all_rho,
                    "total_cells": offset,
                    "region_slices": region_slices,
                })

        # Train/val/test split by simulation
        n_sims = len(self._simulations)
        n_test = max(1, int(n_sims * cfg.test_fraction))
        n_val = max(1, int(n_sims * cfg.val_fraction))
        n_train = n_sims - n_val - n_test

        import random
        shuffled = list(range(n_sims))
        random.Random(42).shuffle(shuffled)

        if split == "train":
            self.sim_indices = shuffled[:n_train]
        elif split == "val":
            self.sim_indices = shuffled[n_train:n_train + n_val]
        else:
            self.sim_indices = shuffled[n_train + n_val:]

        # Compute stats from train sims
        all_T = []
        all_dT = []
        for si in shuffled[:n_train]:
            sim = self._simulations[si]
            all_T.append(sim["T_all"].ravel())
            dT = np.diff(sim["T_all"], axis=0).ravel()
            all_dT.append(dT)

        all_T = np.concatenate(all_T)
        all_dT = np.concatenate(all_dT)
        self.T_mean = float(np.mean(all_T))
        self.T_std = float(np.std(all_T)) + 1e-8
        self.dT_mean = float(np.mean(all_dT))
        self.dT_std = float(np.std(all_dT)) + 1e-8
        self.Tset_mean = self.T_mean  # T_set is in same range
        self.Tset_std = self.T_std

        # Build index: (sim_i, t_i) pairs
        self._index = []
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            n_t = sim["n_times"]
            t_max = min(n_t, cfg.n_train_steps) if split_mode == "training" else n_t
            for t_i in range(20, t_max - 1):
                self._index.append((sim_i, t_i))

        # Pre-build interpolators for static fields per sim
        print(f"  Building 3D interpolators for {len(self.sim_indices)} sims...")
        self._static_grids = {}
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            coords = sim["coords"]

            # Static channels interpolated to grid (done once)
            interp_fields = {}
            # Region ID as single float channel (not one-hot)
            region_ids_float = np.zeros((sim["total_cells"], 1), dtype=np.float32)
            for j in range(sim["total_cells"]):
                # Find which region this cell belongs to by checking onehot
                region_ids_float[j, 0] = np.argmax(sim["region_onehot"][j]) / 11.0

            for ch_name, ch_data in [
                ("region_id", region_ids_float),
                ("is_heater", sim["is_heater"][:, None]),
                ("kappa", sim["kappa"][:, None] / 100.0),
                ("Cp", sim["Cp"][:, None] / 1000.0),
                ("rho", sim["rho"][:, None] / 10000.0),
            ]:
                n_ch = ch_data.shape[1] if ch_data.ndim > 1 else 1
                grid_data = np.zeros((self.n_grid, n_ch), dtype=np.float32)
                for c in range(n_ch):
                    vals = ch_data[:, c] if ch_data.ndim > 1 else ch_data
                    interp = NearestNDInterpolator(coords, vals)
                    grid_data[:, c] = interp(self.grid_points)
                interp_fields[ch_name] = grid_data.reshape(
                    *self.grid_shape, n_ch)

            # T interpolator (reusable for any timestep)
            # Region weight map: steel=10, air=3, heaters=0.1, outer=0.1
            region_weights = np.ones(sim["total_cells"], dtype=np.float32)
            for j in range(sim["total_cells"]):
                rid = np.argmax(sim["region_onehot"][j])
                if rid == 0:       # steel_cylinder
                    region_weights[j] = 10.0
                elif rid == 1:     # inner_box
                    region_weights[j] = 3.0
                elif rid == 11:    # outer_box
                    region_weights[j] = 0.1
                else:              # heaters + brick
                    region_weights[j] = 0.1

            interp_w = NearestNDInterpolator(coords, region_weights)
            weight_grid = interp_w(self.grid_points).reshape(*self.grid_shape)

            self._static_grids[sim_i] = {
                "interp_fields": interp_fields,
                "T_interp": NearestNDInterpolator(coords, np.zeros(sim["total_cells"])),
                "weight_grid": weight_grid,
            }

        print(f"  [{split:5s}|{split_mode:10s}] {len(self._index):>6,} samples | "
              f"{len(self.sim_indices)} sims | grid {self.grid_shape}")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sim_i, t_i = self._index[idx]
        sim = self._simulations[sim_i]
        static = self._static_grids[sim_i]
        fields = static["interp_fields"]

        T_set = sim["T_set"]
        t_val = sim["times"][t_i]

        # Interpolate T at timestep t and t+1 onto grid
        T_t = sim["T_all"][t_i]
        T_tp1 = sim["T_all"][t_i + 1]

        interp_t = NearestNDInterpolator(sim["coords"], T_t)
        interp_tp1 = NearestNDInterpolator(sim["coords"], T_tp1)

        T_grid_t = interp_t(self.grid_points).reshape(self.grid_shape)
        T_grid_tp1 = interp_tp1(self.grid_points).reshape(self.grid_shape)

        # Normalise
        T_norm = (T_grid_t - self.T_mean) / self.T_std
        Tset_norm = (T_set - self.T_mean) / self.T_std
        t_norm = t_val / 4000.0
        # Target: T_next normalised (not delta_T!)
        # Predicting full temperature field gives much stronger loss signal
        dT = T_grid_tp1 - T_grid_t  # keep for reference
        dT_norm = (dT - self.dT_mean) / self.dT_std  # kept for compatibility

        # Build input: (8, Gx, Gy, Gz)
        Gx, Gy, Gz = self.grid_shape
        x = np.zeros((8, Gx, Gy, Gz), dtype=np.float32)
        x[0] = T_norm                                    # T_current
        x[1] = Tset_norm                                 # T_set (scalar broadcast)
        x[2] = fields["region_id"].squeeze(-1)           # region_id / 11
        x[3] = t_norm                                    # time
        x[4] = fields["is_heater"].squeeze(-1)           # is_heater
        x[5] = fields["kappa"].squeeze(-1)               # kappa/100
        x[6] = fields["Cp"].squeeze(-1)                  # Cp/1000
        x[7] = fields["rho"].squeeze(-1)                 # rho/10000

        # Target: T_next normalised (full field, not delta)
        T_next_norm = (T_grid_tp1 - self.T_mean) / self.T_std
        y = T_next_norm[None, ...]

        # Region weight for loss weighting
        weight = static["weight_grid"].copy()

        # Training noise: add small perturbation to T_norm
        if self.split == "train":
            noise = np.random.normal(0, 0.03, size=T_norm.shape).astype(np.float32)
            x[0] = T_norm + noise

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(T_grid_t, dtype=torch.float32),
            torch.tensor(T_grid_tp1, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


def get_fno3d_dataloaders(cfg):
    kw = dict(num_workers=2, pin_memory=True)
    train_ds = FNO3DDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds   = FNO3DDataset(cfg.dataset_path, cfg, "val",   "training")
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kw),
        DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw),
        train_ds, val_ds,
    )
