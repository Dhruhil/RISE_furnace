#!/bin/bash
# =============================================================================
#  install_pi_deeponet.sh
#
#  Upgrades DeepONet folder to full Physics-Informed version (Option B-lite):
#    - autograd spatial Laplacian
#    - finite-difference ∂T/∂t
#    - Newton convection + Stefan-Boltzmann radiation
#    - 4-term weighted physics loss matching GNN_Unified (0.5 conv + 0.3 cond + 0.2 rad)
#
#  Run from inside DeepONet_PhysicsNeMo_Official/:
#      bash install_pi_deeponet.sh
#
#  Backs up originals to *.bak_option_a.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "  Installing PI-DeepONet (Option B-lite)"
echo "============================================================"

# ----- backup originals -----
for f in training/loss.py training/train.py data/dataset.py models/rollout.py; do
    if [ -f "$f" ]; then
        cp "$f" "${f}.bak_option_a"
        echo "  backup: $f -> ${f}.bak_option_a"
    fi
done

# ----- update config: reduce n_query_points for memory safety -----
python3 - <<'PYEOF'
from pathlib import Path
p = Path("configs/deeponet_config.py")
src = p.read_text()
old = "n_query_points: int = 4096"
new = "n_query_points: int = 1024"
if old in src:
    p.write_text(src.replace(old, new))
    print("  config: n_query_points 4096 -> 1024")
else:
    # Already changed or not present: check and warn
    import re
    m = re.search(r"n_query_points:\s*int\s*=\s*(\d+)", src)
    if m:
        print(f"  config: n_query_points already = {m.group(1)} (leaving as-is)")
    else:
        print("  WARNING: n_query_points not found in config")
PYEOF

# ===========================================================================
#  training/loss.py
# ===========================================================================
cat > training/loss.py <<'LOSS_EOF'
"""
PI-DeepONet physics loss — full PDE residual with autograd.
rho*Cp*dT/dt = kappa*∇²T + h*(T_set-T)/δ + ε*σ*(T_set⁴-T⁴)/δ

∇²T via torch.autograd.grad (exact, spatial)
∂T/∂t via finite difference (matches FNO convention)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_SB         = 5.67e-8
EMISSIVITY_STEEL = 0.80
H_CONV           = 25.0
CHAR_THICKNESS   = 0.01


def weighted_mse(pred, target, weight):
    return ((pred - target).pow(2) * weight).sum() / (weight.sum() + 1e-8)


def spatial_laplacian(T_pred, xyz):
    grad_T = torch.autograd.grad(
        outputs=T_pred.sum(), inputs=xyz,
        create_graph=True, retain_graph=True,
    )[0]
    lap = torch.zeros_like(T_pred)
    for i in range(3):
        grad2 = torch.autograd.grad(
            outputs=grad_T[..., i].sum(), inputs=xyz,
            create_graph=True, retain_graph=True,
        )[0]
        lap = lap + grad2[..., i]
    return lap


def pi_deeponet_physics_loss(
    T_pred_norm, T_pred_next_norm, xyz, T_cur_K, T_set,
    region_id, is_heater, kappa, Cp, rho,
    T_mean, T_std, dt=10.0,
):
    T_pred_K = T_pred_norm      * T_std + T_mean
    T_next_K = T_pred_next_norm * T_std + T_mean
    non_heater = 1.0 - is_heater

    dT_dt = (T_next_K - T_pred_K) / dt

    lap_T      = spatial_laplacian(T_pred_K, xyz)
    alpha      = kappa / (rho * Cp + 1e-8)
    dT_dt_cond = alpha * lap_T
    cond_res   = (dT_dt - dT_dt_cond) / 10.0
    L_cond     = ((cond_res * non_heater).pow(2)).mean()

    T_set_b    = T_set.unsqueeze(-1)
    rho_cp     = rho * Cp + 1e-8
    dT_dt_conv = H_CONV * (T_set_b - T_pred_K) / (rho_cp * CHAR_THICKNESS)
    conv_res   = (dT_dt - dT_dt_conv) / 100.0
    L_conv_match = ((conv_res * non_heater).pow(2)).mean()
    overshoot    = F.relu(T_next_K - T_set_b) * non_heater
    L_overshoot  = (overshoot / T_set_b.clamp(min=300.0)).pow(2).mean()
    L_conv       = 0.5 * L_conv_match + 0.5 * L_overshoot

    dT_dt_rad = (EMISSIVITY_STEEL * SIGMA_SB *
                 (T_set_b.pow(4) - T_pred_K.pow(4)) / (rho_cp * CHAR_THICKNESS))
    rad_res = (dT_dt - dT_dt_rad) / 1000.0
    L_rad   = ((rad_res * non_heater).pow(2)).mean()

    L_phys = 0.5 * L_conv + 0.3 * L_cond + 0.2 * L_rad
    breakdown = {
        "cond": float(L_cond.detach()), "conv": float(L_conv.detach()),
        "rad":  float(L_rad.detach()),  "overshoot": float(L_overshoot.detach()),
        "physics": float(L_phys.detach()),
    }
    return L_phys, breakdown


class DeepONetLoss(nn.Module):
    def __init__(self, lambda_physics=0.003):
        super().__init__()
        self.lambda_physics = lambda_physics

    def forward(self, pred_norm, target_norm, weight,
                T_set, T_mean, T_std,
                pred_next_norm=None, xyz=None, T_cur_K=None,
                region_id=None, is_heater=None,
                kappa=None, Cp=None, rho=None, dt=10.0):
        data = weighted_mse(pred_norm, target_norm, weight)
        has_phys = all(v is not None for v in [pred_next_norm, xyz, T_cur_K,
                        region_id, is_heater, kappa, Cp, rho])
        if not has_phys or self.lambda_physics < 1e-10:
            return data, {"data": float(data.detach()), "physics": 0.0,
                          "cond": 0.0, "conv": 0.0, "rad": 0.0, "overshoot": 0.0}
        L_phys, bd = pi_deeponet_physics_loss(
            pred_norm, pred_next_norm, xyz, T_cur_K, T_set,
            region_id, is_heater, kappa, Cp, rho,
            T_mean, T_std, dt)
        total = data + self.lambda_physics * L_phys
        return total, {"data": float(data.detach()), **bd}
LOSS_EOF
echo "  wrote: training/loss.py"

# ===========================================================================
#  data/dataset.py
# ===========================================================================
cat > data/dataset.py <<'DS_EOF'
"""
PI-DeepONet dataset — returns 13-tuple including raw xyz coords and
raw SI material properties at query points (needed for physics loss).
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
HEATER_REGIONS = {f"heater_{i}" for i in range(1, 9)} | {"brick_heater"}
REGION_PROPERTIES = {
    "steel_cylinder": {"kappa": 60.0, "Cp": 450.0, "rho": 7800.0},
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
                self._simulations.append({
                    "T_set": T_set, "times": times, "n_times": n_t,
                    "total_cells": offset, "coords": coords_all,
                    "region_id": rid_all, "is_heater": heat_all,
                    "kappa": kappa_all, "Cp": Cp_all, "rho": rho_all,
                    "T_all": T_all, "weight": w,
                    "region_slices": region_slices,
                })

        if not self._simulations:
            raise RuntimeError(f"No cases found in {h5_path}")
        n_sims = len(self._simulations)

        rng = np.random.default_rng(42)
        perm = rng.permutation(n_sims)
        n_test  = max(1, int(round(cfg.test_fraction * n_sims)))
        n_val   = max(1, int(round(cfg.val_fraction  * n_sims)))
        test_idx  = perm[:n_test].tolist()
        val_idx   = perm[n_test:n_test + n_val].tolist()
        train_idx = perm[n_test + n_val:].tolist()
        self.sim_indices = {"train": train_idx, "val": val_idx,
                            "test": test_idx}[split]

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
        branch_scalars = np.array([Tset_norm, t_norm], dtype=np.float32)

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
    kw = dict(num_workers=2, pin_memory=True)
    train_ds = DeepONetDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds   = DeepONetDataset(cfg.dataset_path, cfg, "val",   "training")
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw),
        train_ds, val_ds,
    )


def get_deeponet_eval_dataset(cfg):
    return DeepONetDataset(cfg.dataset_path, cfg, "test", "evaluation")
DS_EOF
echo "  wrote: data/dataset.py"

# ===========================================================================
#  training/train.py  (full PI-DeepONet training loop)
# ===========================================================================
# This file is too long to embed; stage as separate step to keep this script
# under 500 lines. Actually let's embed it too.
cat > training/train.py <<'TRAIN_EOF'
"""PI-DeepONet training loop — autograd physics, matched to FNO/GNN."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, ".")
from configs.deeponet_config import CONFIG
from data.dataset import get_deeponet_dataloaders
from models.deeponet_model import HeatTreatmentDeepONet
from training.scheduler import build_scheduler
from training.loss import DeepONetLoss
from utils.checkpoint import CheckpointManager
from utils.metrics import compute_metrics
from utils.logging import setup_logging


def get_physics_lambda(epoch, n_epochs):
    p = min(1.0, epoch / max(n_epochs, 1))
    cosine = 0.5 * (1 + math.cos(math.pi * p))
    return 0.0005 + (0.003 - 0.0005) * cosine


def get_pushforward_weight(epoch, n_epochs):
    warmup_end = int(n_epochs * 0.10)
    if epoch <= warmup_end: return 0.0
    return 1.0 * (epoch - warmup_end) / (n_epochs - warmup_end)


def get_warmup_lr(epoch, base_lr, warmup_epochs=5):
    if epoch <= warmup_epochs:
        return base_lr * (0.1 + 0.9 * epoch / warmup_epochs)
    return base_lr


def _build_trunk_with_grad(xyz, region_id, is_heater, kappa, Cp, rho):
    static = torch.stack([
        region_id, is_heater,
        kappa / 100.0, Cp / 1000.0, rho / 10000.0,
    ], dim=-1)
    return torch.cat([xyz, static], dim=-1)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    T_mean, T_std, grad_clip, w2, lam, dt=10.0,
                    noise_std=0.03, t_total=3460.0):
    model.train()
    totals = {"loss":0,"data":0,"phys":0,"cond":0,"conv":0,"rad":0,"overshoot":0,"pf":0}
    n = 0
    for batch_ in loader:
        (branch, scalars, _trunk_old, y, T_cur_K, T_next_gt, w,
         xyz, rid, is_heat, kappa, Cp, rho) = [
            b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b
            for b in batch_]
        if noise_std > 0:
            branch = branch.clone()
            branch[:,0:1,:] = branch[:,0:1,:] + torch.randn_like(branch[:,0:1,:])*noise_std
        use_phys = lam > 1e-6
        xyz_grad = xyz.clone().detach().requires_grad_(use_phys)
        trunk = _build_trunk_with_grad(xyz_grad, rid, is_heat, kappa, Cp, rho)

        optimizer.zero_grad()
        pred1 = model(branch, scalars, trunk)
        pred_next = None
        if use_phys:
            scalars_next = scalars.clone()
            scalars_next[:,1] = scalars_next[:,1] + dt / t_total
            pred_next = model(branch, scalars_next, trunk)

        T_set_K = scalars[:,0] * T_std + T_mean
        loss1, parts = criterion(
            pred_norm=pred1, target_norm=y, weight=w,
            T_set=T_set_K, T_mean=T_mean, T_std=T_std,
            pred_next_norm=pred_next,
            xyz=xyz_grad if use_phys else None,
            T_cur_K=T_cur_K if use_phys else None,
            region_id=rid if use_phys else None,
            is_heater=is_heat if use_phys else None,
            kappa=kappa if use_phys else None,
            Cp=Cp if use_phys else None,
            rho=rho if use_phys else None,
            dt=dt,
        )

        loss = loss1
        pf_val = 0.0
        if w2 > 1e-6:
            with torch.no_grad():
                shift = pred1.mean(dim=1, keepdim=True).unsqueeze(1)
            branch_pf = branch.clone()
            branch_pf[:,0:1,:] = branch_pf[:,0:1,:] + shift
            trunk_pf = _build_trunk_with_grad(
                xyz_grad.detach(), rid, is_heat, kappa, Cp, rho)
            pred_pf = model(branch_pf, scalars, trunk_pf)
            loss2 = ((pred_pf - y).pow(2) * w).sum() / (w.sum() + 1e-8)
            loss = loss1 + w2 * loss2
            pf_val = float(loss2.detach())

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["data"] += parts["data"]
        totals["phys"] += parts.get("physics",0.0)
        totals["cond"] += parts.get("cond",0.0)
        totals["conv"] += parts.get("conv",0.0)
        totals["rad"]  += parts.get("rad",0.0)
        totals["overshoot"] += parts.get("overshoot",0.0)
        totals["pf"]   += pf_val
        n += 1
    for k in totals: totals[k] /= max(n,1)
    return totals


@torch.no_grad()
def evaluate(model, loader, criterion, device, T_mean, T_std):
    model.eval()
    losses, maes, r2s = [], [], []
    for batch_ in loader:
        (branch, scalars, trunk, y, _T_cur, T_gt, w, *_rest) = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch_]
        pred = model(branch, scalars, trunk)
        T_set_K = scalars[:,0] * T_std + T_mean
        loss, _ = criterion(
            pred_norm=pred, target_norm=y, weight=w,
            T_set=T_set_K, T_mean=T_mean, T_std=T_std)
        losses.append(float(loss))
        pred_K = pred.cpu().numpy() * T_std + T_mean
        true_K = T_gt.cpu().numpy()
        m = compute_metrics(pred_K.reshape(-1), true_K.reshape(-1))
        maes.append(m["mae"]); r2s.append(m["r2"])
    return float(np.mean(losses)), float(np.mean(maes)), float(np.mean(r2s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_pushforward", action="store_true")
    parser.add_argument("--no_physics", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    cfg = CONFIG
    if args.epochs: cfg.n_epochs = args.epochs
    if args.lr: cfg.learning_rate = args.lr
    if args.batch: cfg.batch_size = args.batch
    LAM_OVERRIDE = args.lam

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    sep = "=" * 72
    print(f"\n{sep}\n  PI-DeepONet — Heat Treatment (autograd physics)\n{sep}")
    print(f"  Device:  {device}   Batch: {cfg.batch_size}   Epochs: {cfg.n_epochs}")
    print(f"  Sensors: {cfg.sensor_grid_x}x{cfg.sensor_grid_y}x{cfg.sensor_grid_z} = {cfg.n_sensors}")
    print(f"  Query points/sample: {cfg.n_query_points}")
    print(f"  Latent dim: {cfg.latent_dim}")
    if args.no_physics:
        print(f"  Physics: OFF (--no_physics)")
    elif LAM_OVERRIDE is not None:
        print(f"  Physics: FIXED lam={LAM_OVERRIDE}")
    else:
        print(f"  Physics: cosine 0.003 -> 0.0005 (autograd Laplacian + Newton + SB)")
    print(f"  Pushforward: " + ("OFF" if args.no_pushforward else "w2 ramp 0 -> 1.0 (10% warmup)"))
    print(f"  LR warmup: linear 5 epochs")
    print(f"  Dataset: {cfg.dataset_path}\n{sep}\n")

    logger = setup_logging(cfg)
    train_loader, val_loader, train_ds, _ = get_deeponet_dataloaders(cfg)
    T_mean, T_std = train_ds.T_mean, train_ds.T_std
    print(f"  Stats: T_mean={T_mean:.1f} K   T_std={T_std:.1f} K\n")

    model = HeatTreatmentDeepONet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = DeepONetLoss(lambda_physics=cfg.lambda_physics)
    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)

    if args.test:
        print("=== SANITY TEST ===")
        batch_ = next(iter(train_loader))
        (br, sc, tr, y, Tc, Tn, w, xyz, rid, ish, k, cp, rho) = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch_]
        print(f"  branch: {tuple(br.shape)}  scalars: {tuple(sc.shape)}  trunk: {tuple(tr.shape)}")
        print(f"  xyz: {tuple(xyz.shape)}  (raw metres, range=[{xyz.min():.3f},{xyz.max():.3f}])")
        print(f"  kappa range=[{k.min():.1f},{k.max():.1f}]  Cp=[{cp.min():.1f},{cp.max():.1f}]  rho=[{rho.min():.1f},{rho.max():.1f}]")
        xyz_g = xyz.clone().detach().requires_grad_(True)
        trunk = _build_trunk_with_grad(xyz_g, rid, ish, k, cp, rho)
        pred1 = model(br, sc, trunk)
        sc_next = sc.clone(); sc_next[:,1] += 10.0/3460.0
        pred_next = model(br, sc_next, trunk)
        print(f"  pred1: {tuple(pred1.shape)}  range=[{pred1.min().item():.3f},{pred1.max().item():.3f}]")
        T_set_K = sc[:,0] * T_std + T_mean
        loss, parts = criterion(
            pred_norm=pred1, target_norm=y, weight=w,
            T_set=T_set_K, T_mean=T_mean, T_std=T_std,
            pred_next_norm=pred_next, xyz=xyz_g, T_cur_K=Tc,
            region_id=rid, is_heater=ish, kappa=k, Cp=cp, rho=rho, dt=10.0)
        print(f"\n  Loss: {float(loss):.4f}  Data: {parts['data']:.4f}  Physics: {parts.get('physics',0):.4f}")
        print(f"    cond={parts.get('cond',0):.4f}  conv={parts.get('conv',0):.4f}  "
              f"rad={parts.get('rad',0):.4f}  overshoot={parts.get('overshoot',0):.4f}")
        loss.backward()
        print(f"\n  Backward pass: OK")
        print(f"  xyz.grad range: [{xyz_g.grad.min().item():.3e},{xyz_g.grad.max().item():.3e}]")
        print("\n  Schedule preview (100 epochs):")
        for ep in [1, 10, 11, 50, 100]:
            print(f"    ep {ep:>3}: w2={get_pushforward_weight(ep,100):.3f}  "
                  f"lam={get_physics_lambda(ep,100):.4f}  lr={get_warmup_lr(ep,1.0):.3f}")
        print("=== OK ===")
        return

    history = {"train_loss":[],"val_loss":[],"val_mae":[],"val_r2":[],
               "lr":[],"w2":[],"lam":[],"pf_loss":[],
               "tr_data":[],"tr_phys":[],"L_cond":[],"L_conv":[],"L_rad":[]}

    print(f"  {'Ep':>4} | {'TrLoss':>9} | {'TrData':>9} | {'Cond':>8} | {'Conv':>8} | "
          f"{'Rad':>8} | {'VaLoss':>9} | {'MAE[K]':>8} | {'R2':>7} | "
          f"{'lam':>7} | {'w2':>6} | {'LR':>9} | {'t[s]':>6}")
    print("  " + "-" * 130)

    best_mae = float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        if args.no_physics: lam = 0.0
        elif LAM_OVERRIDE is not None: lam = LAM_OVERRIDE
        else: lam = get_physics_lambda(epoch, cfg.n_epochs)
        w2 = 0.0 if args.no_pushforward else get_pushforward_weight(epoch, cfg.n_epochs)
        lr = get_warmup_lr(epoch, cfg.learning_rate, warmup_epochs=max(5, cfg.n_epochs//10))
        for pg in optimizer.param_groups: pg["lr"] = lr
        criterion.lambda_physics = lam

        t_ep = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             T_mean, T_std, cfg.grad_clip, w2, lam,
                             dt=cfg.dt, t_total=cfg.t_total)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, criterion, device, T_mean, T_std)
        if epoch > 5: scheduler.step(val_loss)
        dt_ep = time.time() - t_ep
        lr_now = optimizer.param_groups[0]["lr"]

        for k, v in [("train_loss",tr["loss"]),("tr_data",tr["data"]),
                     ("tr_phys",tr["phys"]),("L_cond",tr["cond"]),
                     ("L_conv",tr["conv"]),("L_rad",tr["rad"]),
                     ("val_loss",val_loss),("val_mae",val_mae),
                     ("val_r2",val_r2),("lr",lr_now),("w2",w2),
                     ("lam",lam),("pf_loss",tr["pf"])]:
            history[k].append(v)

        print(f"  {epoch:>4d} | {tr['loss']:>9.4f} | {tr['data']:>9.4f} | "
              f"{tr['cond']:>8.4f} | {tr['conv']:>8.4f} | {tr['rad']:>8.4f} | "
              f"{val_loss:>9.4f} | {val_mae:>8.3f} | {val_r2:>7.4f} | "
              f"{lam:>7.4f} | {w2:>6.3f} | {lr_now:>9.2e} | {dt_ep:>6.1f}")

        if val_mae < best_mae:
            best_mae = val_mae
            ckpt_mgr.save_best(model, optimizer, scheduler, epoch, {"mae":val_mae,"r2":val_r2})
        if epoch % cfg.save_every_n_epochs == 0:
            ckpt_mgr.save_epoch(model, optimizer, scheduler, epoch, {"mae":val_mae,"r2":val_r2})

    total = time.time() - t0
    with open(f"{cfg.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training done in {total/60:.1f} min ({total/3600:.2f} hrs)")
    print(f"  Best val MAE: {best_mae:.3f} K")


if __name__ == "__main__":
    main()
TRAIN_EOF
echo "  wrote: training/train.py"

# ===========================================================================
#  models/rollout.py  (adapted to 13-tuple; forward unchanged)
# ===========================================================================
cat > models/rollout.py <<'RO_EOF'
"""Autoregressive rollout — PI-DeepONet. Dataset returns 13 items now."""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator


@torch.no_grad()
def rollout_deeponet(model, dataset, sim_i, device="cuda",
                     start_t=20, chunk_size=8192):
    model.eval()
    model.to(device)
    sim = dataset._simulations[sim_i]
    sens = dataset._static_sensors[sim_i]
    cfg = dataset.cfg

    T_mean = dataset.T_mean
    T_std  = dataset.T_std
    Tset_norm = (sim["T_set"] - dataset.Tset_mean) / dataset.Tset_std

    coords = sim["coords"]
    n_cells = sim["total_cells"]
    heater_cells = sim["is_heater"] > 0.5
    times = sim["times"]
    n_t = sim["n_times"]

    n_rollout = n_t - start_t
    T_pred = np.zeros((n_rollout, n_cells), dtype=np.float32)
    T_true = np.zeros((n_rollout, n_cells), dtype=np.float32)
    T_pred[0] = sim["T_all"][start_t]
    T_true[0] = sim["T_all"][start_t]
    for step in range(1, n_rollout):
        T_true[step] = sim["T_all"][start_t + step]

    trunk_static = np.stack([
        coords[:,0], coords[:,1], coords[:,2],
        sim["region_id"] / 11.0,
        sim["is_heater"],
        sim["kappa"]  / 100.0,
        sim["Cp"]    / 1000.0,
        sim["rho"]  / 10000.0,
    ], axis=1).astype(np.float32)
    trunk_all = torch.from_numpy(trunk_static).to(device)

    T_cur = T_pred[0].copy()
    for step in range(1, n_rollout):
        t_idx = start_t + step
        if t_idx >= n_t: break
        t_val = times[t_idx - 1]

        interp = NearestNDInterpolator(coords, T_cur)
        T_sens = interp(dataset.sensor_points).astype(np.float32)
        T_sens_norm = (T_sens - T_mean) / T_std

        branch = np.stack([
            T_sens_norm, sens["region_id"], sens["is_heater"],
            sens["kappa"], sens["Cp"], sens["rho"],
        ], axis=0).astype(np.float32)
        branch = torch.from_numpy(branch).unsqueeze(0).to(device)
        scalars = torch.tensor(
            [Tset_norm, t_val / cfg.t_total], dtype=torch.float32
        ).unsqueeze(0).to(device)

        preds = torch.zeros(n_cells, dtype=torch.float32, device=device)
        for s in range(0, n_cells, chunk_size):
            e = min(n_cells, s + chunk_size)
            y = trunk_all[s:e].unsqueeze(0)
            out = model(branch, scalars, y)
            preds[s:e] = out.squeeze(0)

        T_next_norm = preds.cpu().numpy()
        T_next = T_next_norm * T_std + T_mean
        T_next[heater_cells] = sim["T_set"]
        T_pred[step] = T_next.astype(np.float32)
        T_cur = T_next

    return T_pred, T_true
RO_EOF
echo "  wrote: models/rollout.py"

# ----- verify all files parse -----
echo ""
echo "Verifying Python syntax..."
ALL_OK=true
for f in training/loss.py data/dataset.py training/train.py models/rollout.py; do
    if python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        echo "  $f: OK"
    else
        echo "  $f: FAIL"
        ALL_OK=false
    fi
done

echo ""
echo "============================================================"
if [ "$ALL_OK" = true ]; then
    echo "  PI-DeepONet installation complete."
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Sanity test:"
    echo "     sbatch run_sanity_test_deeponet.sh"
    echo "     squeue -u \$USER"
    echo "     # wait 2-3 min, check log:"
    echo "     ls -t outputs/logs/test_*.log | head -1 | xargs tail -80"
    echo ""
    echo "  2. If OK, launch full training:"
    echo "     sbatch run_alvis_deeponet.sh"
    echo ""
    echo "To revert:"
    echo "  for f in training/loss.py training/train.py data/dataset.py models/rollout.py; do"
    echo "    [ -f \${f}.bak_option_a ] && cp \${f}.bak_option_a \$f"
    echo "  done"
else
    echo "  FAIL — some files have syntax errors. Check output above."
    echo "============================================================"
fi
