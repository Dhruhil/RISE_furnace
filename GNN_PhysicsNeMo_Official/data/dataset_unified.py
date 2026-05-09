"""
Unified multi-region dataset for the GNN.

All 12 regions get folded into ONE graph per (sim, timestep).
Inter-region edges connect nodes from different regions that sit
physically close together (within a small threshold).
"""
from __future__ import annotations
import json
import re
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import h5py
from configs.base_config import REGION_MATERIALS


# Region IDs — order matters because this maps onto the cellToRegion file
# from OpenFOAM. Steel cylinder is 0, outer_box is 11.
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
    "outer_box":      11,
}

# These regions get clamped to T_set during rollout, so the model
# never has to predict their values.
HEATER_REGIONS = {
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
    "brick_heater",
}


class UnifiedDataset(torch.utils.data.Dataset):
    """
    One sample = ALL regions at one timestep of one simulation.

    Graph stats: ~13.7k nodes, KNN intra-region edges + boundary
    inter-region edges where regions physically touch.
    """

    def __init__(self, h5_path, cfg, split="train", split_mode="training"):
        super().__init__()
        self.split = split
        self.split_mode = split_mode
        self.cfg = cfg

        # ---- load every simulation from the HDF5 file ------------------
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

                    # Safety net for any NaNs left over from the cleaning
                    # script — replace them with the regional mean so
                    # they don't poison training. Should not trigger in
                    # practice on the cleaned dataset.
                    if np.isnan(T_array).any():
                        n_nan = int(np.isnan(T_array).sum())
                        T_array = np.nan_to_num(
                            T_array, nan=float(np.nanmean(T_array))
                        )
                        if region == 'steel_cylinder':
                            print(f"  [NaN safety] case_{case_idx:03d} {region}: {n_nan} NaN -> mean")

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

                # Pull geometry info out of the case name string.
                # The HDF5 only stores 'name', 'T_set', 'original_case_num',
                # 'original_index' as attrs, so the geometry has to be
                # parsed from the encoded name like:
                #   cx<num>mm cy<num>mm cz<num>mm r<num>mm h<num>mm
                _name = str(grp.attrs.get("name", ""))

                def _parse_mm(pattern, _n, default):
                    _m = re.search(pattern, _n)
                    return float(_m.group(1)) / 1000.0 if _m else default

                cyl_params = {
                    "name":   _name,
                    "cx":     _parse_mm(r'cx(-?\d+)mm',  _name, 0.0),
                    "cy":     _parse_mm(r'cy(-?\d+)mm',  _name, 0.18),
                    "cz":     _parse_mm(r'cz(-?\d+)mm',  _name, 0.195),
                    "radius": _parse_mm(r'r(\d+)mm',     _name, 0.05),
                    "height": _parse_mm(r'h(\d+)mm',     _name, 0.10),
                    "kappa":  float(grp.attrs.get("kappa", 60.0)),
                    "Cp":     float(grp.attrs.get("Cp", 450.0)),
                    "rho":    float(grp.attrs.get("rho", 7800.0)),
                }

                self._simulations.append({
                    "T_set": T_set,
                    "times": times,
                    "n_times": len(times),
                    "region_data": region_data,
                    "all_coords": np.concatenate(all_coords),
                    "all_region_ids": np.concatenate(all_region_ids),
                    "total_nodes": offset,
                    **cyl_params,
                })

        n_sims = len(self._simulations)

        # ---- stratified train/val/test split (seeded for reproducibility)
        # Stratification is by T_set so each split gets cases from every
        # operating point — otherwise the test set could end up with
        # only the high-T cases that crashed.
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
            # Make sure at least one case lands in train, no matter how
            # rounding shakes out at the smallest setpoints.
            if _n - _n_test - _n_val < 1:
                _n_val = max(0, _n - _n_test - 1)
            _test.extend(_idxs[:_n_test])
            _val.extend(_idxs[_n_test:_n_test + _n_val])
            _train.extend(_idxs[_n_test + _n_val:])

        # Canonical order (train, then val, then test) — lets the
        # normalisation step below pick out only the training cases.
        shuffled = _train + _val + _test
        n_train = len(_train)
        n_val = len(_val)
        n_test = len(_test)

        split_map = {"train": _train, "val": _val, "test": _test}
        print(f"  [stratified split] train={len(_train)} val={len(_val)} test={len(_test)}")
        for _tset in sorted(_by_tset):
            _tr = sum(1 for i in _train if float(self._simulations[i]["T_set"]) == _tset)
            _vl = sum(1 for i in _val   if float(self._simulations[i]["T_set"]) == _tset)
            _te = sum(1 for i in _test  if float(self._simulations[i]["T_set"]) == _tset)
            print(f"    T_set={_tset:.0f}K  train={_tr} val={_vl} test={_te}")
        self.sim_indices = split_map[split]

        # ---- normalisation stats from training cases only --------------
        # Skip the first 20 timesteps when computing dT stats — the
        # heater ramp-up gives huge dT values there that would otherwise
        # blow up the std.
        all_T, all_dT = [], []
        for i in shuffled[:n_train]:
            for rdata in self._simulations[i]["region_data"].values():
                all_T.append(rdata["T_array"].ravel())
                dT = np.diff(rdata["T_array"], axis=0)[20:].ravel()
                all_dT.append(dT)

        all_T = np.concatenate(all_T).astype(np.float64)
        all_dT = np.concatenate(all_dT).astype(np.float64)
        self.T_mean = float(all_T.mean())
        self.T_std = float(all_T.std()) + 1e-8           # eps avoids div by zero
        self.dT_mean = float(np.mean(all_dT))
        self.dT_std = float(np.std(all_dT)) + 1e-8

        print(f"  Unified: T_mean={self.T_mean:.1f}K  T_std={self.T_std:.1f}K")
        print(f"           dT_mean={self.dT_mean:.5f}K  dT_std={self.dT_std:.4f}K")

        # ---- sample index = (sim, timestep) pairs ----------------------
        # No region dimension here — every sample is the whole graph.
        # Start from t=20 to skip the heater warm-up phase.
        # The -3 leaves room for the pushforward targets (t+1, t+2, t+3).
        self._index = []
        for sim_i in self.sim_indices:
            sim = self._simulations[sim_i]
            n_t = sim["n_times"] - 1
            t_max = min(n_t, cfg.n_train_steps) if split_mode == "training" else n_t
            for t_i in range(20, t_max - 3):  # was -2, bumped to -3 for y3
                self._index.append((sim_i, t_i))

        # ---- build the graph once per simulation -----------------------
        # The graph topology only depends on the geometry, not the
        # timestep, so it makes sense to cache it per sim and reuse.
        print(f"  Building unified graphs for {len(self.sim_indices)} sims...")
        self._graphs = {}
        k_intra = cfg.graph_k_neighbors      # KNN inside each region
        boundary_dist = 0.02                 # 2cm threshold for cross-region edges

        for gi, sim_i in enumerate(self.sim_indices):
            sim = self._simulations[sim_i]
            all_coords = sim["all_coords"]
            total = sim["total_nodes"]

            # Intra-region edges via KNN (nodes within the same region)
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
                        diff = coords[j] - coords[i]
                        dist = np.linalg.norm(diff)
                        # Each edge gets added in both directions —
                        # message passing needs both. edge_type=0 marks
                        # an intra-region edge.
                        src_list.extend([offset + i, offset + j])
                        dst_list.extend([offset + j, offset + i])
                        eattr_list.extend([
                            [diff[0], diff[1], diff[2], dist, 0.0],
                            [-diff[0], -diff[1], -diff[2], dist, 0.0],
                        ])

            # Inter-region (boundary) edges — only between regions that
            # actually touch in the geometry. The .geo file places the
            # heaters and cylinder inside inner_box, and inner_box sits
            # inside outer_box, so the valid pairs are:
            VALID_ADJACENCY = {
                frozenset({"heater_1", "inner_box"}),
                frozenset({"heater_2", "inner_box"}),
                frozenset({"heater_3", "inner_box"}),
                frozenset({"heater_4", "inner_box"}),
                frozenset({"heater_5", "inner_box"}),
                frozenset({"heater_6", "inner_box"}),
                frozenset({"heater_7", "inner_box"}),
                frozenset({"heater_8", "inner_box"}),
                frozenset({"brick_heater", "inner_box"}),
                frozenset({"steel_cylinder", "inner_box"}),
                frozenset({"inner_box", "outer_box"}),
                frozenset({"brick_heater", "outer_box"}),
            }
            region_list = list(sim["region_data"].keys())
            for r1_idx in range(len(region_list)):
                for r2_idx in range(r1_idx + 1, len(region_list)):
                    r1 = region_list[r1_idx]
                    r2 = region_list[r2_idx]
                    if frozenset({r1, r2}) not in VALID_ADJACENCY:
                        continue
                    c1 = sim["region_data"][r1]["coords"]
                    c2 = sim["region_data"][r2]["coords"]
                    o1 = sim["region_data"][r1]["offset"]
                    o2 = sim["region_data"][r2]["offset"]

                    # For each cell in r1, grab the nearest cell in r2
                    # and keep the edge only if they're closer than the
                    # boundary threshold.
                    tree2 = cKDTree(c2)
                    dists, idxs = tree2.query(c1, k=1)
                    mask = dists < boundary_dist

                    for i in np.where(mask)[0]:
                        j = idxs[i]
                        diff = c2[j] - c1[i]
                        dist = dists[i]
                        # Bidirectional, edge_type=1 for inter-region
                        src_list.extend([o1 + i, o2 + j])
                        dst_list.extend([o2 + j, o1 + i])
                        eattr_list.extend([
                            [diff[0], diff[1], diff[2], dist, 1.0],
                            [-diff[0], -diff[1], -diff[2], dist, 1.0],
                        ])

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(eattr_list, dtype=torch.float32)

            # Edge distances span several orders of magnitude across
            # regions, so divide by the mean to keep training stable.
            if edge_attr.shape[0] > 0:
                mean_dist = edge_attr[:, 3].mean().clamp(min=1e-8)
                edge_attr[:, :3] = edge_attr[:, :3] / mean_dist
                edge_attr[:, 3] = edge_attr[:, 3] / mean_dist
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

        # ---- collect targets at t, t+1, t+2, t+3 -----------------------
        # Three future steps are needed for the pushforward training trick.
        T_t   = np.zeros(total, dtype=np.float32)
        T_tp1 = np.zeros(total, dtype=np.float32)
        T_tp2 = np.zeros(total, dtype=np.float32)
        T_tp3 = np.zeros(total, dtype=np.float32)

        for region, rdata in sim["region_data"].items():
            o = rdata["offset"]
            n = rdata["n_cells"]
            T_t[o:o+n]   = rdata["T_array"][t_i]
            T_tp1[o:o+n] = rdata["T_array"][t_i + 1]
            T_tp2[o:o+n] = rdata["T_array"][t_i + 2]
            T_tp3[o:o+n] = rdata["T_array"][t_i + 3]

        # ---- build node features ---------------------------------------
        # Temperatures are z-score normalised; everything else gets
        # divided by a fixed scale so all features land roughly in [0, 1].
        T_norm = (T_t - self.T_mean) / self.T_std
        Tset_norm = (T_set - self.T_mean) / self.T_std
        t_norm = t_val / 4000.0   # full simulation length is < 4000s

        # Heater flag — 1 for heater/brick cells, 0 for everything else.
        # Used both as a feature and as a mask for the loss.
        is_heater_feat = np.zeros(total, dtype=np.float32)
        for region, rdata in sim["region_data"].items():
            if region in HEATER_REGIONS:
                o = rdata["offset"]
                n = rdata["n_cells"]
                is_heater_feat[o:o+n] = 1.0

        # Per-region material properties (rescaled into [0, 1] roughly)
        _kappa_feat = np.zeros(total, dtype=np.float32)
        _Cp_feat    = np.zeros(total, dtype=np.float32)
        _rho_feat   = np.zeros(total, dtype=np.float32)
        for _rname, _rdata in sim["region_data"].items():
            _o = _rdata["offset"]; _n = _rdata["n_cells"]
            _mat = REGION_MATERIALS.get(_rname, {"kappa": 80.0, "Cp": 450.0, "rho": 7800.0})
            _kappa_feat[_o:_o+_n] = _mat["kappa"] / 100.0
            _Cp_feat[_o:_o+_n]    = _mat["Cp"] / 1000.0
            _rho_feat[_o:_o+_n]   = _mat["rho"] / 10000.0

        # Same material props but raw (un-rescaled) — the physics loss
        # needs the actual SI values to compute fluxes correctly.
        _kappa_raw = np.zeros(total, dtype=np.float32)
        _Cp_raw    = np.zeros(total, dtype=np.float32)
        _rho_raw   = np.zeros(total, dtype=np.float32)
        for _rname, _rdata in sim["region_data"].items():
            _o = _rdata["offset"]; _n = _rdata["n_cells"]
            _mat = REGION_MATERIALS.get(_rname, {"kappa": 80.0, "Cp": 450.0, "rho": 7800.0})
            _kappa_raw[_o:_o+_n] = _mat["kappa"]
            _Cp_raw[_o:_o+_n]    = _mat["Cp"]
            _rho_raw[_o:_o+_n]   = _mat["rho"]

        # Final 16-dim feature stack. Future-proofed so it works even
        # if the dataset later includes varying cx/cz/radius/height.
        node_feats = np.column_stack([
            all_coords[:, 0],                                          # [0]  x
            all_coords[:, 1],                                          # [1]  y
            all_coords[:, 2],                                          # [2]  z
            T_norm,                                                    # [3]  T_current
            np.full(total, Tset_norm, dtype=np.float32),               # [4]  T_set
            all_rids / 11.0,                                           # [5]  region_id
            np.full(total, t_norm, dtype=np.float32),                  # [6]  time
            is_heater_feat,                                            # [7]  is_heater
            np.full(total, sim["cx"] / 0.206, dtype=np.float32),       # [8]  cx
            np.full(total, sim["cy"] / 0.36, dtype=np.float32),        # [9]  cy
            np.full(total, sim["cz"] / 0.39, dtype=np.float32),        # [10] cz
            np.full(total, sim["radius"] / 0.10, dtype=np.float32),    # [11] radius
            np.full(total, sim["height"] / 0.20, dtype=np.float32),    # [12] height
            _kappa_feat,    # [13] kappa  (per-region, /100)
            _Cp_feat,       # [14] Cp     (per-region, /1000)
            _rho_feat,      # [15] rho    (per-region, /10000)
        ]).astype(np.float32)

        # Targets are normalised dT (next-step temperatures, z-scored)
        T_tp1_norm = ((T_tp1 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)
        T_tp2_norm = ((T_tp2 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)
        T_tp3_norm = ((T_tp3 - self.T_mean) / self.T_std).reshape(-1, 1).astype(np.float32)

        # Light noise on the input temperature during training only —
        # this is the standard MeshGraphNet trick for stabilising
        # autoregressive rollouts at inference time.
        if self.split == "train":
            noise = np.random.normal(0, 0.02, size=total).astype(np.float32)
            node_feats[:, 3] += noise

        # Heater mask (used by the loss to skip heater cells, since
        # those are clamped to T_set and not predicted)
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
            y=torch.tensor(T_tp1_norm, dtype=torch.float32),
            y2=torch.tensor(T_tp2_norm, dtype=torch.float32),
            y3=torch.tensor(T_tp3_norm, dtype=torch.float32),
            T_current=torch.tensor(T_t, dtype=torch.float32),
            T_next=torch.tensor(T_tp1, dtype=torch.float32),
            T_tp2=torch.tensor(T_tp2, dtype=torch.float32),
            T_tp3=torch.tensor(T_tp3, dtype=torch.float32),
            T_set_raw=torch.tensor(
                np.full(total, T_set, dtype=np.float32), dtype=torch.float32),
            is_heater=torch.tensor(is_heater, dtype=torch.float32),
            region_ids=torch.tensor(all_rids, dtype=torch.long),
            kappa_raw=torch.tensor(_kappa_raw, dtype=torch.float32),
            Cp_raw=torch.tensor(_Cp_raw, dtype=torch.float32),
            rho_raw=torch.tensor(_rho_raw, dtype=torch.float32),
            Y_std=float(self.T_std),
            dT_mean=float(self.dT_mean),
            dT_std=float(self.dT_std),
        )
        data.sim_idx = sim_i
        data.t_idx = t_i
        return data