"""
PI-DeepONet dataset — returns 13-tuple including raw xyz coords and
raw SI material properties at query points (needed for physics loss).
"""
from __future__ import annotations
import json
import re
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
HEATER_REGIONS = {f"heater_{i}" for i in range(1, 9)} | {"brick_heater"}
REGION_PROPERTIES = {
    "steel_cylinder": {"kappa": 80.0, "Cp": 450.0, "rho": 7800.0},  
    "inner_box":      {"kappa": 0.026, "Cp": 1005.0, "rho": 1.2},
    "outer_box":      {"kappa": 1.5, "Cp": 900.0, "rho": 1800.0},
    "brick_heater":   {"kappa": 1.5, "Cp": 900.0, "rho": 1800.0},
}
for _i in range(1, 9):
    REGION_PROPERTIES[f"heater_{_i}"] = {"kappa": 15.0, "Cp": 500.0, "rho": 2400.0}

REGION_WEIGHTS = {"steel_cylinder": 10.0, "inner_box": 3.0,
                  "brick_heater": 1.0, "outer_box": 0.1}
for _i in range(1, 9):
    REGION_WEIGHTS[f"heater_{_i}"] = 1.0


class DeepONetDataset(Dataset):
    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.split_mode = split_mode

        sx = np.linspace(cfg.x_min, cfg.x_max, cfg.sensor_grid_x).astype(np.float32)
        sy = np.linspace(cfg.y_min, cfg.y_max, cfg.sensor_grid_y).astype(np.float32)
        sz = np.linspace(cfg.z_min, cfg.z_max, cfg.sensor_grid_z).astype(np.float32)
        self.sensor_points = np.stack(
            np.meshgrid(sx, sy, sz, indexing="ij"), axis=-1
        ).reshape(-1, 3).astype(np.float32)

        self._simulations = []
        with h5py.File(h5_path, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])
            for ci in range(n_cases):
                grp = f[f"case_{ci:03d}"]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)
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
                n_t = times.shape[0]
                T_all = np.zeros((n_t, offset), dtype=np.float32)
                for region, (a, b) in region_slices.items():
                    T_all[:, a:b] = grp[region]["T"][:].astype(np.float32)
                if np.isnan(T_all).any():
                    col_mean = np.nanmean(T_all, axis=0)
                    col_mean = np.where(np.isnan(col_mean), 300.0, col_mean)
                    nan_mask = np.isnan(T_all)
                    T_all[nan_mask] = np.broadcast_to(col_mean, T_all.shape)[nan_mask]
                w = np.ones(offset, dtype=np.float32)
                for region, (a, b) in region_slices.items():
                    w[a:b] = REGION_WEIGHTS.get(region, 1.0)
                # Parse cylinder geometry from case name string
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

        all_T, all_dT = [], []
        for sim_i in self.sim_indices:
            T = self._simulations[sim_i]["T_all"]
            all_T.append(T.reshape(-1))
            all_dT.append((T[1:] - T[:-1]).reshape(-1))
        self.T_mean  = float(np.mean(np.concatenate(all_T)))
        self.T_std   = float(np.std (np.concatenate(all_T))) + 1e-8
        self.dT_mean = float(np.mean(np.concatenate(all_dT)))
        self.dT_std  = float(np.std (np.concatenate(all_dT))) + 1e-8
        self.Tset_mean = self.T_mean
        self.Tset_std  = self.T_std

        self._index = []
        for sim_i in self.sim_indices:
            n_t = self._simulations[sim_i]["n_times"]
            t_max = min(n_t, cfg.n_train_steps) if split_mode == "training" else n_t
            for t_i in range(20, t_max - 1):
                self._index.append((sim_i, t_i))

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

        interp_T = NearestNDInterpolator(sim["coords"], T_t)
        T_sens   = interp_T(self.sensor_points).astype(np.float32)
        T_sens_norm = (T_sens - self.T_mean) / self.T_std
        branch = np.stack([
            T_sens_norm, sens["region_id"], sens["is_heater"],
            sens["kappa"], sens["Cp"], sens["rho"],
        ], axis=0).astype(np.float32)
        if self.split == "train":
            branch[0] = branch[0] + np.random.normal(
                0.0, 0.03, size=branch[0].shape).astype(np.float32)

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

        T_next_q = T_tp1[q_idx]
        y = ((T_next_q - self.T_mean) / self.T_std).astype(np.float32)
        w_q = sim["weight"][q_idx].astype(np.float32)

        xyz_raw     = q_coords.astype(np.float32)
        region_id_q = (sim["region_id"][q_idx] / 11.0).astype(np.float32)
        is_heat_q   = sim["is_heater"][q_idx].astype(np.float32)
        kappa_q_raw = sim["kappa"][q_idx].astype(np.float32)
        Cp_q_raw    = sim["Cp"][q_idx].astype(np.float32)
        rho_q_raw   = sim["rho"][q_idx].astype(np.float32)
        T_cur_q_raw = T_t[q_idx].astype(np.float32)
        T_next_q_raw= T_tp1[q_idx].astype(np.float32)

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
    kw = dict(num_workers=0, pin_memory=True)
    train_ds = DeepONetDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds   = DeepONetDataset(cfg.dataset_path, cfg, "val",   "training")
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw),
        train_ds, val_ds,
    )


def get_deeponet_eval_dataset(cfg):
    return DeepONetDataset(cfg.dataset_path, cfg, "test", "evaluation")
