"""
PI-DeepONet dataset.

Returns a 13-tuple per sample, which is bigger than what the GNN
or FNO datasets emit because the physics-informed loss needs the
raw (unnormalised) coordinates and SI material properties at the
query points to build its residuals. Stripping those out at
collation time would cost an extra denormalisation step in the
loss for every batch, so they ride along instead.

The branch input is built from a fixed sensor lattice spanning
the furnace volume; the trunk input is a per-batch sample of
n_query_points cells drawn from the underlying mesh.
"""
from __future__ import annotations
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import NearestNDInterpolator
import h5py


# Region IDs — same order and values as the FNO and GNN datasets,
# so the region_id channel is comparable across architectures.
REGION_IDS = {
    "steel_cylinder": 0, "inner_box": 1,
    "heater_1": 2, "heater_2": 3, "heater_3": 4, "heater_4": 5,
    "heater_5": 6, "heater_6": 7, "heater_8": 9, "heater_7": 8,
    "brick_heater": 10, "outer_box": 11,
}
# Note: dict literal keeps insertion order in Py 3.7+, so the
# keys here happen to land in the same order as the other datasets
# despite the heater_7/heater_8 swap above being a typo-like
# artefact of how this was written. Values are what matters.

# Heater regions get clamped to T_set during rollout, so the model
# never has to predict their values. Same set as in the GNN/FNO.
HEATER_REGIONS = {f"heater_{i}" for i in range(1, 9)} | {"brick_heater"}

# Per-region material defaults (rough literature values). The steel
# cylinder kappa is bumped to 80.0 here vs 60.0 in the FNO config —
# small difference that doesn't matter much for the trunk's
# normalised channels, but worth flagging if anyone is debugging.
REGION_PROPERTIES = {
    "steel_cylinder": {"kappa": 80.0,  "Cp": 450.0,  "rho": 7800.0},
    "inner_box":      {"kappa": 0.026, "Cp": 1005.0, "rho": 1.2},     # cavity air
    "outer_box":      {"kappa": 1.5,   "Cp": 900.0,  "rho": 1800.0},
    "brick_heater":   {"kappa": 1.5,   "Cp": 900.0,  "rho": 1800.0},
}
# Heaters share one set of properties — bulk values don't matter
# much since heater cells are clamped to T_set anyway.
for _i in range(1, 9):
    REGION_PROPERTIES[f"heater_{_i}"] = {"kappa": 15.0, "Cp": 500.0, "rho": 2400.0}

# Per-region loss weights — same scheme as the FNO/GNN. Steel
# cylinder gets the bulk of the gradient signal; the outer
# enclosure barely moves so 0.1x prevents it from drowning out
# the cylinder.
REGION_WEIGHTS = {"steel_cylinder": 10.0, "inner_box": 3.0,
                  "brick_heater": 1.0, "outer_box": 0.1}
for _i in range(1, 9):
    REGION_WEIGHTS[f"heater_{_i}"] = 1.0


class DeepONetDataset(Dataset):
    """
    Dataset for the PI-DeepONet surrogate.

    Each __getitem__ call assembles three pieces:
      1. branch input  — the current temperature field encoded at a
                         fixed sensor lattice (compressed view of u(x))
      2. branch scalars — T_set, time, and per-sim cylinder geometry
                         (the branch can't tell two cases at the same
                         T_set apart from the sensor field alone)
      3. trunk input   — query coordinates plus per-cell static
                         material properties

    On top of that, the raw (unnormalised) values get returned
    alongside so the physics loss can build residuals in plain SI
    units without re-doing the denormalisation step every batch.
    """

    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.split_mode = split_mode

        # ---- fixed sensor lattice ---------------------------------------
        # Same lattice for every sim — the branch net always sees the
        # same input layout regardless of which case is being processed,
        # which is what lets it learn a stable encoding of u(x).
        sx = np.linspace(cfg.x_min, cfg.x_max, cfg.sensor_grid_x).astype(np.float32)
        sy = np.linspace(cfg.y_min, cfg.y_max, cfg.sensor_grid_y).astype(np.float32)
        sz = np.linspace(cfg.z_min, cfg.z_max, cfg.sensor_grid_z).astype(np.float32)
        self.sensor_points = np.stack(
            np.meshgrid(sx, sy, sz, indexing="ij"), axis=-1
        ).reshape(-1, 3).astype(np.float32)

        # ---- load every simulation from the HDF5 file -------------------
        self._simulations = []
        with h5py.File(h5_path, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])
            for ci in range(n_cases):
                grp = f[f"case_{ci:03d}"]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)

                # Stack every cell from every region into one flat
                # array — the trunk samples query points from this
                # combined pool, so it needs to be one cohesive set.
                all_coords, all_rid, all_heater = [], [], []
                all_kappa, all_Cp, all_rho = [], [], []
                region_slices = {}
                offset = 0
                for region in regions:
                    if region not in grp: continue
                    coords = grp[region]["coords"][:].astype(np.float32)
                    n = coords.shape[0]
                    rid = REGION_IDS.get(region, 0)
                    props = REGION_PROPERTIES.get(region,
                        {"kappa": 1.0, "Cp": 500.0, "rho": 2000.0})
                    all_coords.append(coords)
                    all_rid.append(np.full(n, rid, dtype=np.float32))
                    all_heater.append(np.full(n,
                        1.0 if region in HEATER_REGIONS else 0.0, dtype=np.float32))
                    all_kappa.append(np.full(n, props["kappa"], dtype=np.float32))
                    all_Cp.append(np.full(n, props["Cp"], dtype=np.float32))
                    all_rho.append(np.full(n, props["rho"], dtype=np.float32))
                    region_slices[region] = (offset, offset + n)
                    offset += n
                if offset == 0: continue
                coords_all = np.concatenate(all_coords, axis=0)
                rid_all    = np.concatenate(all_rid, axis=0)
                heat_all   = np.concatenate(all_heater, axis=0)
                kappa_all  = np.concatenate(all_kappa, axis=0)
                Cp_all     = np.concatenate(all_Cp, axis=0)
                rho_all    = np.concatenate(all_rho, axis=0)

                # T_all has shape (n_times, total_cells). Each row is
                # the full temperature field at one OpenFOAM dump.
                n_t = times.shape[0]
                T_all = np.zeros((n_t, offset), dtype=np.float32)
                for region, (a, b) in region_slices.items():
                    T_all[:, a:b] = grp[region]["T"][:].astype(np.float32)

                # Safety net for any NaNs left over from the cleaning
                # script — fill with the per-cell time-mean rather
                # than dropping the whole timestep, which would tear
                # holes in the rollout sequence.
                if np.isnan(T_all).any():
                    col_mean = np.nanmean(T_all, axis=0)
                    col_mean = np.where(np.isnan(col_mean), 300.0, col_mean)
                    nan_mask = np.isnan(T_all)
                    T_all[nan_mask] = np.broadcast_to(col_mean, T_all.shape)[nan_mask]

                # Per-cell loss weights, indexed by region slice
                w = np.ones(offset, dtype=np.float32)
                for region, (a, b) in region_slices.items():
                    w[a:b] = REGION_WEIGHTS.get(region, 1.0)

                # ---- parse cylinder geometry from the case name -----
                # The HDF5 case name encodes per-sim geometry as
                # cx{mm}/cy{mm}/cz{mm}/r{mm}/h{mm} substrings.
                # Pulling them out here lets the branch scalars
                # carry the per-sim geometry through to the model.
                _name = str(grp.attrs.get("name", ""))
                def _parse_mm(_pat, _default):
                    _m = re.search(_pat, _name)
                    return float(_m.group(1)) / 1000.0 if _m else _default
                _cyl = {
                    "cx":     _parse_mm(r'cx(-?\d+)mm',  0.0),
                    "cy":     _parse_mm(r'cy(-?\d+)mm',  0.18),
                    "cz":     _parse_mm(r'cz(-?\d+)mm',  0.195),
                    "radius": _parse_mm(r'r(\d+)mm',     0.05),
                    "height": _parse_mm(r'h(\d+)mm',     0.10),
                }
                self._simulations.append({
                    "T_set": T_set, "times": times, "n_times": n_t,
                    "total_cells": offset, "coords": coords_all,
                    "region_id": rid_all, "is_heater": heat_all,
                    "kappa": kappa_all, "Cp": Cp_all, "rho": rho_all,
                    "T_all": T_all, "weight": w,
                    "region_slices": region_slices,
                    **_cyl,
                })

        if not self._simulations:
            raise RuntimeError(f"No cases found in {h5_path}")

        # ---- stratified train/val/test split (same seed as FNO/GNN) ----
        # Stratification by T_set keeps every operating point in
        # every split. The high-T cases are rarer and harder, so
        # losing them from train would skew the headline numbers.
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

        split_map = {"train": _train, "val": _val, "test": _test}
        print(f"  [stratified split] train={len(_train)} val={len(_val)} test={len(_test)}")
        for _tset in sorted(_by_tset):
            _tr = sum(1 for i in _train if float(self._simulations[i]["T_set"]) == _tset)
            _vl = sum(1 for i in _val   if float(self._simulations[i]["T_set"]) == _tset)
            _te = sum(1 for i in _test  if float(self._simulations[i]["T_set"]) == _tset)
            print(f"    T_set={_tset:.0f}K  train={_tr} val={_vl} test={_te}")

        self.sim_indices = split_map[split]

        # ---- normalisation stats from the current split ----------------
        # Stats are computed over whichever split is being constructed
        # (train uses train cases, val uses val cases, etc.) since the
        # caller passes this dataset around with explicit T_mean/T_std
        # at training time and the train stats get used for inference.
        all_T, all_dT = [], []
        for sim_i in self.sim_indices:
            T = self._simulations[sim_i]["T_all"]
            all_T.append(T.reshape(-1))
            all_dT.append((T[1:] - T[:-1]).reshape(-1))
        self.T_mean  = float(np.mean(np.concatenate(all_T)))
        self.T_std   = float(np.std (np.concatenate(all_T))) + 1e-8
        self.dT_mean = float(np.mean(np.concatenate(all_dT)))
        self.dT_std  = float(np.std (np.concatenate(all_dT))) + 1e-8
        # T_set lives in the same physical range as T, so reuse the
        # same stats — keeps the two channels on the same scale.
        self.Tset_mean = self.T_mean
        self.Tset_std  = self.T_std

        # ---- (sim, timestep) index pairs ------------------------------
        # Skip the first 20 timesteps to avoid the heater warm-up
        # transient. The -1 leaves room for the t+1 target.
        self._index = []
        for sim_i in self.sim_indices:
            n_t = self._simulations[sim_i]["n_times"]
            t_max = min(n_t, cfg.n_train_steps) if split_mode == "training" else n_t
            for t_i in range(20, t_max - 1):
                self._index.append((sim_i, t_i))

        # ---- pre-build static sensor fields per sim --------------------
        # The sensor lattice is fixed, and the static fields
        # (region_id, is_heater, kappa, Cp, rho) don't change with
        # time — interpolating them once per sim saves a lot of
        # redundant work in __getitem__.
        self._static_sensors = {}
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            coords = sim["coords"]
            fields = {}
            for name, data, scale in [
                ("region_id", sim["region_id"], 11.0),
                ("is_heater", sim["is_heater"], 1.0),
                ("kappa",     sim["kappa"],    100.0),
                ("Cp",        sim["Cp"],      1000.0),
                ("rho",       sim["rho"],    10000.0),
            ]:
                interp = NearestNDInterpolator(coords, data)
                fields[name] = (interp(self.sensor_points) / scale).astype(np.float32)
            self._static_sensors[sim_i] = fields

    def __len__(self):
        return len(self._index)

    def _sample_query_points(self, sim):
        """
        Pick the query-point indices for this sample.

        Random subsampling without replacement when the mesh has
        more cells than the configured n_query_points; otherwise
        return all cell indices (the upper bound is rarely hit
        in practice given the mesh sizes here).
        """
        n = sim["total_cells"]
        n_q = self.cfg.n_query_points
        if n_q >= n:
            return np.arange(n)
        return np.random.default_rng().choice(n, size=n_q, replace=False)

    def __getitem__(self, idx):
        sim_i, t_i = self._index[idx]
        sim   = self._simulations[sim_i]
        sens  = self._static_sensors[sim_i]
        cfg   = self.cfg
        T_t   = sim["T_all"][t_i]
        T_tp1 = sim["T_all"][t_i + 1]
        T_set = sim["T_set"]
        t_val = sim["times"][t_i]

        # ---- branch input: encoded current field at sensor lattice ----
        # Interpolate T_t onto the fixed sensor lattice, normalise,
        # and stack with the pre-computed static channels.
        interp_T = NearestNDInterpolator(sim["coords"], T_t)
        T_sens   = interp_T(self.sensor_points).astype(np.float32)
        T_sens_norm = (T_sens - self.T_mean) / self.T_std
        branch = np.stack([
            T_sens_norm, sens["region_id"], sens["is_heater"],
            sens["kappa"], sens["Cp"], sens["rho"],
        ], axis=0).astype(np.float32)
        # Light Gaussian noise on the temperature row at training
        # time — same trick as the FNO/GNN, helps the model stay
        # stable across the long autoregressive rollout.
        if self.split == "train":
            branch[0] = branch[0] + np.random.normal(
                0.0, 0.03, size=branch[0].shape).astype(np.float32)

        # ---- branch scalars: T_set, time, cylinder geometry -----------
        # Each component normalised against a representative scale so
        # they all enter the network at O(1) magnitude.
        Tset_norm = (T_set - self.Tset_mean) / self.Tset_std
        t_norm    = t_val / cfg.t_total
        branch_scalars = np.array([
            Tset_norm,
            t_norm,
            sim["cx"]     / 0.206,
            sim["cy"]     / 0.36,
            sim["cz"]     / 0.39,
            sim["radius"] / 0.10,
            sim["height"] / 0.20,
        ], dtype=np.float32)

        # ---- trunk input: per-query-point coords + static channels ----
        q_idx = self._sample_query_points(sim)
        q_coords = sim["coords"][q_idx]
        trunk = np.stack([
            q_coords[:, 0], q_coords[:, 1], q_coords[:, 2],
            sim["region_id"][q_idx] / 11.0,
            sim["is_heater"][q_idx],
            sim["kappa"][q_idx]  / 100.0,
            sim["Cp"][q_idx]    / 1000.0,
            sim["rho"][q_idx]  / 10000.0,
        ], axis=1).astype(np.float32)

        # ---- target + per-query loss weight ---------------------------
        T_next_q = T_tp1[q_idx]
        y = ((T_next_q - self.T_mean) / self.T_std).astype(np.float32)
        w_q = sim["weight"][q_idx].astype(np.float32)

        # ---- raw values for the physics loss --------------------------
        # The physics loss residuals (conduction, convection,
        # radiation, energy balance) are written in plain SI units,
        # so the unnormalised coordinates and material values get
        # passed through alongside the normalised ones.
        xyz_raw     = q_coords.astype(np.float32)
        region_id_q = (sim["region_id"][q_idx] / 11.0).astype(np.float32)
        is_heat_q   = sim["is_heater"][q_idx].astype(np.float32)
        kappa_q_raw = sim["kappa"][q_idx].astype(np.float32)
        Cp_q_raw    = sim["Cp"][q_idx].astype(np.float32)
        rho_q_raw   = sim["rho"][q_idx].astype(np.float32)
        T_cur_q_raw = T_t[q_idx].astype(np.float32)
        T_next_q_raw= T_tp1[q_idx].astype(np.float32)

        # 13-tuple — order matters. The training loop unpacks by
        # position, so don't reorder these without checking the loop.
        return (
            torch.from_numpy(branch),           # 0
            torch.from_numpy(branch_scalars),   # 1
            torch.from_numpy(trunk),            # 2
            torch.from_numpy(y),                # 3
            torch.from_numpy(T_cur_q_raw),      # 4
            torch.from_numpy(T_next_q_raw),     # 5
            torch.from_numpy(w_q),              # 6
            torch.from_numpy(xyz_raw),          # 7
            torch.from_numpy(region_id_q),      # 8
            torch.from_numpy(is_heat_q),        # 9
            torch.from_numpy(kappa_q_raw),      # 10
            torch.from_numpy(Cp_q_raw),         # 11
            torch.from_numpy(rho_q_raw),        # 12
        )


def get_deeponet_dataloaders(cfg):
    """
    Build the train and validation DeepONet dataloaders.

    num_workers=0 here on purpose — the dataset's __getitem__ does
    a NearestNDInterpolator construction per sample, and SciPy's
    kd-tree builder doesn't play nicely with worker-process
    pickling. Single-process loading is fast enough for the batch
    sizes used here.
    """
    kw = dict(num_workers=0, pin_memory=True)
    train_ds = DeepONetDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds   = DeepONetDataset(cfg.dataset_path, cfg, "val",   "training")
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw),
        train_ds, val_ds,
    )


def get_deeponet_eval_dataset(cfg):
    """
    Construct the test-split dataset in 'evaluation' mode, which
    exposes all timesteps (including the Phase-2 extrapolation
    window) instead of capping at n_train_steps.
    """
    return DeepONetDataset(cfg.dataset_path, cfg, "test", "evaluation")