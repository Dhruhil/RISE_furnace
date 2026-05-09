"""
3D FNO dataset — interpolates all 12 regions onto a regular Cartesian grid.

Reuses the same dataset_all_regions.h5 file that the GNN consumes,
so no extra data preprocessing is needed. The GNN keeps the
unstructured mesh; this dataset rasterises everything onto a
regular grid that the FNO's spectral convolutions can act on.

Per-sample layout:
  Each sample = ALL regions at one timestep -> one 3D volume.
  Input:  (C, Gx, Gy, Gz) with C = 8 channels
  Output: (1, Gx, Gy, Gz) = normalised T_next on the grid

The Gx/Gy/Gz grid resolution is set in the FNOConfig.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import NearestNDInterpolator
import h5py


# Region IDs — same order and values as in the GNN dataset, so a
# region_id channel is comparable across architectures.
REGION_IDS = {
    "steel_cylinder": 0, "inner_box": 1,
    "heater_1": 2, "heater_2": 3, "heater_3": 4, "heater_4": 5,
    "heater_5": 6, "heater_6": 7, "heater_7": 8, "heater_8": 9,
    "brick_heater": 10, "outer_box": 11,
}

# These get clamped to T_set during rollout, so the model never has
# to predict their values. Same set as in the GNN pipeline.
HEATER_REGIONS = {
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
    "brick_heater",
}

# Per-region material properties (rough literature values).
# Used as fallback defaults — actual per-sim steel values get
# overridden later from the HDF5 case attributes.
REGION_PROPERTIES = {
    "steel_cylinder": {"kappa": 60.0,  "Cp": 450.0,  "rho": 7800.0},
    "inner_box":      {"kappa": 0.026, "Cp": 1005.0, "rho": 1.2},     # air
    "outer_box":      {"kappa": 1.5,   "Cp": 900.0,  "rho": 1800.0},
    "brick_heater":   {"kappa": 1.5,   "Cp": 900.0,  "rho": 1800.0},
}
# Heaters share one set of properties — bulk values don't matter
# much since heater cells are clamped to T_set anyway.
for i in range(1, 9):
    REGION_PROPERTIES[f"heater_{i}"] = {"kappa": 15.0, "Cp": 500.0, "rho": 2400.0}


class FNO3DDataset(Dataset):
    """
    3D regular-grid dataset built from the multi-region HDF5 file.

    All region cells get interpolated (nearest-neighbour) onto the
    same Cartesian grid so the FNO can run its 3D spectral
    convolutions on a single dense tensor.

    Channels (8 total):
      [0]  T_norm         current temperature, z-scored
      [1]  T_set_norm     furnace setpoint, z-scored
      [2]  region_id/11   region encoding (0=steel ... 11=outer_box)
      [3]  time/t_total   normalised simulation time
      [4]  is_heater      binary heater flag
      [5]  kappa/100      thermal conductivity in [0, 1]-ish range
      [6]  Cp/1000        heat capacity in [0, 1]-ish range
      [7]  rho/10000      density in [0, 1]-ish range
    """

    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.split_mode = split_mode

        # ---- build the regular 3D grid ---------------------------------
        # Stored as a flat (Gx*Gy*Gz, 3) array of grid points so the
        # NearestNDInterpolator below can take it in one shot.
        self.gx = np.linspace(cfg.x_min, cfg.x_max, cfg.grid_x).astype(np.float32)
        self.gy = np.linspace(cfg.y_min, cfg.y_max, cfg.grid_y).astype(np.float32)
        self.gz = np.linspace(cfg.z_min, cfg.z_max, cfg.grid_z).astype(np.float32)
        self.grid_points = np.stack(
            np.meshgrid(self.gx, self.gy, self.gz, indexing='ij'), axis=-1
        ).reshape(-1, 3)
        self.grid_shape = (cfg.grid_x, cfg.grid_y, cfg.grid_z)
        self.n_grid = cfg.grid_x * cfg.grid_y * cfg.grid_z

        # ---- load every simulation from the HDF5 file ------------------
        self._simulations = []
        with h5py.File(h5_path, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])

            for ci in range(n_cases):
                grp = f[f"case_{ci:03d}"]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)

                # Stack every cell from every region into one flat
                # array so a single interpolator can handle the whole
                # case at inference time.
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

                    # region_slices lets the time-loop below write
                    # each region's T-array straight into the right
                    # slot of the flat T_all matrix.
                    region_slices[region] = (offset, offset + n)
                    offset += n

                all_coords = np.concatenate(all_coords)
                all_region_ids = np.concatenate(all_region_ids)
                all_is_heater = np.concatenate(all_is_heater)
                all_kappa = np.concatenate(all_kappa)
                all_Cp = np.concatenate(all_Cp)
                all_rho = np.concatenate(all_rho)

                # T_all has shape (n_times, total_cells). Each row is
                # the full temperature field at one OpenFOAM dump.
                T_all = np.zeros((len(times), offset), dtype=np.float32)
                for region in regions:
                    if region not in grp:
                        continue
                    s, e = region_slices[region]
                    T_region = grp[region]["T"][:].astype(np.float32)

                    # Safety net for any NaNs left over from the
                    # cleaning script — replace them with the regional
                    # mean so they don't poison the interpolators or
                    # the normalisation stats.
                    if np.isnan(T_region).any():
                        n_nan = int(np.isnan(T_region).sum())
                        T_region = np.nan_to_num(T_region, nan=float(np.nanmean(T_region)))
                        if region == 'steel_cylinder':
                            print(f"  [NaN safety] {region}: {n_nan} NaN -> mean")
                    T_all[:, s:e] = T_region

                # Region one-hot: 12 binary channels per cell.
                # Used internally for the static grid build and for
                # picking the per-cell region when computing weights.
                region_onehot = np.zeros((offset, 12), dtype=np.float32)
                for j in range(offset):
                    region_onehot[j, all_region_ids[j]] = 1.0

                # Per-sim cylinder material properties — read off the
                # HDF5 attrs and used to override the steel defaults.
                # Falls back to nominal values if the attr is missing.
                cy_val    = float(grp.attrs.get("cy",    0.18))
                kappa_val = float(grp.attrs.get("kappa", 60.0))
                Cp_val    = float(grp.attrs.get("Cp",    450.0))
                rho_val   = float(grp.attrs.get("rho",   7800.0))

                # Override the steel cylinder properties with the
                # actual per-sim values so each case sees its own
                # material constants instead of a global fallback.
                for region in regions:
                    if region == "steel_cylinder" and region in region_slices:
                        s, e = region_slices[region]
                        all_kappa[s:e] = kappa_val
                        all_Cp[s:e]    = Cp_val
                        all_rho[s:e]   = rho_val

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

        # ---- stratified train/val/test split (same seed as GNN) -------
        # Stratification by T_set keeps every operating point in
        # every split, which matters because the high-T cases are
        # rarer and harder to learn.
        n_sims = len(self._simulations)
        import random as _rand
        _by_tset = {}
        for _i, _s in enumerate(self._simulations):
            _by_tset.setdefault(float(_s["T_set"]), []).append(_i)

        _train, _val, _test = [], [], []
        _rng = _rand.Random(42)
        for _tset in sorted(_by_tset):
            _idxs = _by_tset[_tset][:]
            _rng.shuffle(_idxs)
            _n = len(_idxs)
            _n_test = max(1, int(round(_n * cfg.test_fraction))) if _n >= 3 else 0
            _n_val  = max(1, int(round(_n * cfg.val_fraction)))  if _n >= 2 else 0
            # Keep at least one case in train regardless of how
            # rounding shakes out at the smallest setpoints.
            if _n - _n_test - _n_val < 1:
                _n_val = max(0, _n - _n_test - 1)
            _test.extend(_idxs[:_n_test])
            _val.extend(_idxs[_n_test:_n_test + _n_val])
            _train.extend(_idxs[_n_test + _n_val:])

        # Canonical order (train, val, test) so the normalisation
        # stats below pick out only the training cases.
        shuffled = _train + _val + _test
        n_train = len(_train)
        n_val   = len(_val)
        n_test  = len(_test)

        split_map = {"train": _train, "val": _val, "test": _test}
        print(f"  [stratified split] train={n_train} val={n_val} test={n_test}")
        for _tset in sorted(_by_tset):
            _tr = sum(1 for i in _train if float(self._simulations[i]["T_set"]) == _tset)
            _vl = sum(1 for i in _val   if float(self._simulations[i]["T_set"]) == _tset)
            _te = sum(1 for i in _test  if float(self._simulations[i]["T_set"]) == _tset)
            print(f"    T_set={_tset:.0f}K  train={_tr} val={_vl} test={_te}")

        self.sim_indices = split_map[split]

        # ---- normalisation stats from training cases only --------------
        # T_mean / T_std go on the temperature input;
        # dT_mean / dT_std are kept for compatibility with the
        # delta-T branch below (the model itself targets full T_next).
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
        # T_set lives in the same physical range as T, so reuse the
        # same stats — keeps the two channels on the same scale.
        self.Tset_mean = self.T_mean
        self.Tset_std = self.T_std

        # ---- sample index = (sim, timestep) pairs ---------------------
        # Skip the first 20 timesteps to avoid the heater warm-up
        # transient that would otherwise dominate the loss.
        # The -1 leaves room for the t+1 target.
        self._index = []
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            n_t = sim["n_times"]
            t_max = min(n_t, cfg.n_train_steps) if split_mode == "training" else n_t
            for t_i in range(20, t_max - 1):
                self._index.append((sim_i, t_i))

        # ---- pre-build static grids per simulation --------------------
        # The static channels (region_id, is_heater, kappa, Cp, rho)
        # don't change with time, so interpolating them once per sim
        # saves a lot of redundant work in __getitem__.
        print(f"  Building 3D interpolators for {len(self.sim_indices)} sims...")
        self._static_grids = {}
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            coords = sim["coords"]

            # Static fields are interpolated once and reused at
            # every timestep below.
            interp_fields = {}

            # Single-channel region encoding (rid / 11) — a single
            # float channel ends up far cheaper than a 12-way
            # one-hot stack at the FNO's input.
            region_ids_float = np.zeros((sim["total_cells"], 1), dtype=np.float32)
            for j in range(sim["total_cells"]):
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

            # Per-cell region weights, then interpolated to the grid.
            # Same weighting as the GNN: steel=10, air=3, others=0.1
            # so the steel cylinder dominates the gradient signal.
            region_weights = np.ones(sim["total_cells"], dtype=np.float32)
            for j in range(sim["total_cells"]):
                rid = np.argmax(sim["region_onehot"][j])
                if rid == 0:       # steel_cylinder — engineering target
                    region_weights[j] = 10.0
                elif rid == 1:     # inner_box (cavity air) — convective driver
                    region_weights[j] = 3.0
                elif rid == 11:    # outer_box — quasi-static
                    region_weights[j] = 0.1
                else:              # heaters + brick — clamped to T_set anyway
                    region_weights[j] = 0.1

            interp_w = NearestNDInterpolator(coords, region_weights)
            weight_grid = interp_w(self.grid_points).reshape(*self.grid_shape)

            self._static_grids[sim_i] = {
                "interp_fields": interp_fields,
                # Placeholder T interpolator — gets reused / re-fit
                # for each timestep inside __getitem__. Stored here
                # mostly for reference; the actual interpolation
                # happens on the fly per sample.
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

        # Interpolate the temperature field at t and t+1 onto the grid.
        # Two new interpolators per sample — could be cached, but the
        # NearestNDInterpolator construction is dominated by the kd-tree
        # build, which is already fast for this problem size.
        T_t = sim["T_all"][t_i]
        T_tp1 = sim["T_all"][t_i + 1]

        interp_t = NearestNDInterpolator(sim["coords"], T_t)
        interp_tp1 = NearestNDInterpolator(sim["coords"], T_tp1)

        T_grid_t = interp_t(self.grid_points).reshape(self.grid_shape)
        T_grid_tp1 = interp_tp1(self.grid_points).reshape(self.grid_shape)

        # ---- normalise inputs and targets -----------------------------
        T_norm = (T_grid_t - self.T_mean) / self.T_std
        Tset_norm = (T_set - self.T_mean) / self.T_std
        t_norm = t_val / self.cfg.t_total

        # The model targets full normalised T_next, not delta T —
        # earlier experiments with dT had a much weaker loss signal.
        # The dT values are still computed and kept here in case a
        # later experiment wants them back.
        dT = T_grid_tp1 - T_grid_t
        dT_norm = (dT - self.dT_mean) / self.dT_std

        # ---- build input tensor (8, Gx, Gy, Gz) -----------------------
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

        # Target: T_next as a normalised full field (not a delta)
        T_next_norm = (T_grid_tp1 - self.T_mean) / self.T_std
        y = T_next_norm[None, ...]

        # Per-voxel region weight, used by the loss in the training loop
        weight = static["weight_grid"].copy()

        # Light Gaussian noise on T_norm during training only —
        # same MeshGraphNet trick used in the GNN pipeline, helps
        # the model stay stable across long autoregressive rollouts.
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
    """
    Build the train and validation FNO dataloaders.

    Returns the two loaders along with the underlying datasets, so
    the training loop can pull T_mean / T_std off the train dataset
    when computing physics losses or denormalising predictions.
    """
    kw = dict(num_workers=2, pin_memory=True)
    train_ds = FNO3DDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds   = FNO3DDataset(cfg.dataset_path, cfg, "val",   "training")
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kw),
        DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw),
        train_ds, val_ds,
    )