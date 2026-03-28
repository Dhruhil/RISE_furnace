#!/bin/bash
# ============================================================
# COMPLETE 3D FNO REWRITE
#
# Replaces the broken 1D per-region FNO with a proper 3D FNO
# that operates on a regular grid covering the entire furnace.
#
# All 12 regions interpolated onto ONE 3D grid.
# FFT in x,y,z has real spatial meaning.
# Inter-region coupling is automatic.
#
# cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# bash rewrite_3d_fno.sh
# ============================================================

set -euo pipefail
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "============================================"
echo "  COMPLETE 3D FNO REWRITE"
echo "============================================"

# Backup old files
mkdir -p _backup_1d
for f in configs/fno_config.py data/dataset.py models/fno_model.py \
         models/rollout.py train.py run_alvis_fno.sh; do
    [ -f "$f" ] && cp "$f" "_backup_1d/$(basename $f).bak" 2>/dev/null || true
done
echo "  Backed up old files to _backup_1d/"

# ── 1. CONFIG ─────────────────────────────────────────────────

echo ""
echo "  [1/6] Writing configs/fno_config.py..."

cat > configs/fno_config.py << 'PYEOF'
"""
3D FNO configuration.
All 12 regions interpolated onto one regular grid.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

_BASE = "/mimer/NOBACKUP/groups/revar"

@dataclass
class FNOConfig:
    # Paths
    dataset_path:   str = f"{_BASE}/GNN_PhysicsNeMo_Official/dataset_all_regions.h5"
    output_dir:     str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs"
    checkpoint_dir: str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/checkpoints"
    log_dir:        str = f"{_BASE}/FNO_PhysicsNeMo_Official/outputs/logs"

    # Regions
    all_regions: list = field(default_factory=lambda: [
        "steel_cylinder", "inner_box",
        "heater_1", "heater_2", "heater_3", "heater_4",
        "heater_5", "heater_6", "heater_7", "heater_8",
        "brick_heater", "outer_box",
    ])
    n_regions: int = 12

    # 3D grid resolution (furnace: x=0.206, y=0.36, z=0.39)
    grid_x: int = 24   # ~8.6mm resolution
    grid_y: int = 40   # ~9.0mm resolution
    grid_z: int = 44   # ~8.9mm resolution

    # Furnace bounds (from .geo file)
    x_min: float = 0.0;   x_max: float = 0.206
    y_min: float = 0.0;   y_max: float = 0.36
    z_min: float = 0.0;   z_max: float = 0.39

    # FNO architecture
    # Input channels: T_norm, T_set_norm, region_mask (12 binary), time,
    #                 is_heater, kappa, Cp, rho = 1+1+12+1+1+3 = 19
    fno_in_channels:  int = 19
    fno_out_channels: int = 1    # delta_T normalised
    fno_modes:        list = field(default_factory=lambda: [12, 16, 16])  # modes per dim
    fno_layers:       int = 4
    fno_latent:       int = 64
    fno_decoder_layers:     int = 2
    fno_decoder_layer_size: int = 64

    # Training
    batch_size:      int   = 4
    n_epochs:        int   = 100
    learning_rate:   float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience:     int   = 15
    weight_decay:    float = 1e-5
    grad_clip:       float = 1.0

    # Data splits
    val_fraction:  float = 0.14
    test_fraction: float = 0.10

    # Time
    dt:               float = 10.0
    t_total:          float = 4000.0
    train_time_end:   float = 3200.0
    predict_time_end: float = 4000.0

    # Logging
    log_every_n_epochs:  int  = 1
    save_every_n_epochs: int  = 10

    @property
    def n_train_steps(self) -> int:
        return int(self.train_time_end / self.dt)

    def __post_init__(self):
        for p in [self.checkpoint_dir, self.log_dir,
                  self.output_dir + "/evaluation"]:
            Path(p).mkdir(parents=True, exist_ok=True)

CONFIG = FNOConfig()
PYEOF

echo "    OK"

# ── 2. DATASET ────────────────────────────────────────────────

echo "  [2/6] Writing data/dataset.py..."

cat > data/dataset.py << 'PYEOF'
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

    Channels (19 total):
      [0]     T_norm           current temperature
      [1]     T_set_norm       furnace setpoint
      [2-13]  region_mask      12 binary channels (one-hot per region)
      [14]    time_norm        t / 4000
      [15]    is_heater        binary
      [16]    kappa_norm       thermal conductivity / 100
      [17]    Cp_norm          specific heat / 1000
      [18]    rho_norm         density / 10000
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
                })

        # Train/val/test split by simulation
        n_sims = len(self._simulations)
        n_test = max(1, int(n_sims * cfg.test_fraction))
        n_val = max(1, int(n_sims * cfg.val_fraction))
        n_train = n_sims - n_val - n_test

        if split == "train":
            self.sim_indices = list(range(n_train))
        elif split == "val":
            self.sim_indices = list(range(n_train, n_train + n_val))
        else:
            self.sim_indices = list(range(n_train + n_val, n_sims))

        # Compute stats from train sims
        all_T = []
        all_dT = []
        for si in range(n_train):
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
            for ch_name, ch_data in [
                ("region_onehot", sim["region_onehot"]),
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
            self._static_grids[sim_i] = {
                "interp_fields": interp_fields,
                "T_interp": NearestNDInterpolator(coords, np.zeros(sim["total_cells"])),
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
        dT = T_grid_tp1 - T_grid_t
        dT_norm = (dT - self.dT_mean) / self.dT_std

        # Build input: (19, Gx, Gy, Gz)
        Gx, Gy, Gz = self.grid_shape
        x = np.zeros((19, Gx, Gy, Gz), dtype=np.float32)
        x[0] = T_norm                                    # T_current
        x[1] = Tset_norm                                 # T_set (scalar broadcast)
        x[2:14] = fields["region_onehot"].transpose(3, 0, 1, 2)  # 12 region masks
        x[14] = t_norm                                   # time
        x[15] = fields["is_heater"].squeeze(-1)          # is_heater
        x[16] = fields["kappa"].squeeze(-1)              # kappa/100
        x[17] = fields["Cp"].squeeze(-1)                 # Cp/1000
        x[18] = fields["rho"].squeeze(-1)                # rho/10000

        # Target: (1, Gx, Gy, Gz)
        y = dT_norm[None, ...]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(T_grid_t, dtype=torch.float32),
            torch.tensor(T_grid_tp1, dtype=torch.float32),
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
PYEOF

echo "    OK"

# ── 3. MODEL ──────────────────────────────────────────────────

echo "  [3/6] Writing models/fno_model.py..."

cat > models/fno_model.py << 'PYEOF'
"""
3D Fourier Neural Operator for heat treatment.
Input:  (batch, 19, Gx, Gy, Gz)
Output: (batch, 1, Gx, Gy, Gz)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path

try:
    from physicsnemo.models.fno import FNO as _PhysicsNeMoFNO
    PHYSICSNEMO_FNO = True
except ImportError:
    PHYSICSNEMO_FNO = False
    print("[INFO] physicsnemo FNO not found — using fallback.")


class _SpectralConv3d(nn.Module):
    """3D Fourier layer."""
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes  # [mx, my, mz]
        scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, *modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, Nx, Ny, Nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        mx, my, mz = self.modes
        mx = min(mx, Nx // 2 + 1)
        my = min(my, Ny // 2 + 1)
        mz = min(mz, Nz // 2 + 1)
        out_ft = torch.zeros(B, self.weights.shape[1],
                             Nx, Ny, Nz // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bcxyz,coxyz->boxyz",
            x_ft[:, :, :mx, :my, :mz],
            self.weights[:, :, :mx, :my, :mz])
        return torch.fft.irfftn(out_ft, s=[Nx, Ny, Nz], dim=[-3, -2, -1])


class _FNOBlock3d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = _SpectralConv3d(width, width, modes)
        self.linear = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x):
        return nn.functional.gelu(
            self.norm(self.spectral(x) + self.linear(x)))


class _FallbackFNO3d(nn.Module):
    def __init__(self, in_ch, out_ch, modes, width, n_layers,
                 dec_layers, dec_size):
        super().__init__()
        self.lift = nn.Conv3d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [_FNOBlock3d(width, modes) for _ in range(n_layers)])
        layers = []
        prev = width
        for _ in range(dec_layers):
            layers += [nn.Conv3d(prev, dec_size, 1), nn.GELU()]
            prev = dec_size
        layers.append(nn.Conv3d(prev, out_ch, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x) + x
        return self.decoder(x)


class HeatTreatmentFNO3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if PHYSICSNEMO_FNO:
            self.fno = _PhysicsNeMoFNO(
                in_channels=cfg.fno_in_channels,
                out_channels=cfg.fno_out_channels,
                num_fno_modes=cfg.fno_modes,
                num_fno_layers=cfg.fno_layers,
                latent_channels=cfg.fno_latent,
                decoder_layers=cfg.fno_decoder_layers,
                decoder_layer_size=cfg.fno_decoder_layer_size,
                dimension=3,
                padding=4,
            )
            self._backend = "physicsnemo"
        else:
            self.fno = _FallbackFNO3d(
                cfg.fno_in_channels, cfg.fno_out_channels,
                cfg.fno_modes, cfg.fno_latent, cfg.fno_layers,
                cfg.fno_decoder_layers, cfg.fno_decoder_layer_size)
            self._backend = "fallback"
        n_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  HeatTreatmentFNO3D [{self._backend}]")
        print(f"    in={cfg.fno_in_channels} out={cfg.fno_out_channels} "
              f"modes={cfg.fno_modes} layers={cfg.fno_layers} latent={cfg.fno_latent}")
        print(f"    Grid: {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}")
        print(f"    Trainable parameters: {n_p:,}")

    def forward(self, x):
        return self.fno(x)

    def save(self, path, epoch, opt_state=None, sched_state=None, metrics=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model_state": self.state_dict(),
                     "optimizer_state": opt_state, "scheduler_state": sched_state,
                     "metrics": metrics or {}, "backend": self._backend}, path)

    @classmethod
    def load(cls, path, cfg, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        return model
PYEOF

echo "    OK"

# ── 4. TRAINING ───────────────────────────────────────────────

echo "  [4/6] Writing train.py..."

cat > train.py << 'PYEOF'
"""
3D FNO Training — all regions on one regular grid.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from configs.fno_config import CONFIG
from data.dataset import get_fno3d_dataloaders
from models.fno_model import HeatTreatmentFNO3D
from utils.metrics import compute_metrics, within_tolerance
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logging


def get_physics_lambda(epoch, n_epochs):
    warmup_end = int(n_epochs * 0.2)
    if epoch <= warmup_end:
        return 0.0
    return 0.1 * (epoch - warmup_end) / (n_epochs - warmup_end)


def physics_loss_3d(pred, x, dT_std, dT_mean):
    """3D FNO physics: convection + spectral smoothness + equilibrium."""
    device = pred.device
    dT_pred = pred.squeeze(1) * dT_std + dT_mean  # (B, Gx, Gy, Gz)
    T_norm = x[:, 0]  # already normalised
    Tset_norm = x[:, 1]
    is_heater = x[:, 15]
    non_heater = (1.0 - is_heater)

    # Convection: T_next should not exceed T_set
    # T_next_norm ≈ T_norm + dT_pred / T_std
    T_next_approx = T_norm + pred.squeeze(1)
    overshoot = F.relu(T_next_approx - Tset_norm) * non_heater
    L_conv = overshoot.pow(2).mean()

    # Spectral smoothness: penalise high frequencies in 3D FFT
    pred_fft = torch.fft.rfftn(pred.squeeze(1), dim=[-3, -2, -1])
    Nx, Ny, Nz_half = pred_fft.shape[-3], pred_fft.shape[-2], pred_fft.shape[-1]
    cx, cy, cz = max(Nx // 3, 1), max(Ny // 3, 1), max(Nz_half // 3, 1)
    high_freq = pred_fft[:, cx:, cy:, cz:].abs().pow(2)
    L_smooth = high_freq.mean()

    # Equilibrium: when T ≈ T_set, dT should be small
    gap = (Tset_norm - T_norm).abs()
    near_eq = torch.exp(-gap * 5.0) * non_heater
    L_eq = (pred.squeeze(1) * near_eq).pow(2).mean()

    return 0.5 * L_conv + 0.3 * L_smooth + 0.2 * L_eq


@torch.no_grad()
def evaluate(model, loader, device, train_ds, lam=0.0):
    model.eval()
    total_data, total_phys, n = 0.0, 0.0, 0
    all_pred, all_true = [], []

    for x, y, T_cur, T_next_gt in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        total_data += loss.item()
        if lam > 1e-6:
            lp = physics_loss_3d(pred, x, train_ds.dT_std, train_ds.dT_mean)
            total_phys += lp.item()
        n += 1

        dT_pred = pred.squeeze(1).cpu().numpy() * train_ds.dT_std + train_ds.dT_mean
        T_pred = T_cur.numpy() + dT_pred
        all_pred.append(T_pred.ravel())
        all_true.append(T_next_gt.numpy().ravel())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    m = compute_metrics(all_pred, all_true)
    avg_data = total_data / max(n, 1)
    avg_phys = total_phys / max(n, 1)
    m["loss_data"] = avg_data
    m["loss_phys"] = avg_phys
    m["loss_total"] = (1 - lam) * avg_data + lam * avg_phys if lam > 1e-6 else avg_data
    m["loss"] = avg_data
    m["within_5K"] = within_tolerance(all_pred, all_true, 5.0)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    cfg = CONFIG
    if args.epochs: cfg.n_epochs = args.epochs
    if args.batch:  cfg.batch_size = args.batch
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  3D FNO TRAINING — All Regions on Regular Grid")
    print(f"  Device: {device}  Batch: {cfg.batch_size}  Epochs: {cfg.n_epochs}")
    print(f"  Grid: {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}  "
          f"Modes: {cfg.fno_modes}  Layers: {cfg.fno_layers}")
    print(f"{sep}\n")

    train_loader, val_loader, train_ds, val_ds = get_fno3d_dataloaders(cfg)

    if args.test:
        print("  === SANITY TEST ===")
        batch = next(iter(train_loader))
        x, y, T_cur, T_next_gt = batch
        print(f"  x: {x.shape} (expected: batch, 19, {cfg.grid_x}, {cfg.grid_y}, {cfg.grid_z})")
        print(f"  y: {y.shape}")
        assert x.shape[1] == 19
        assert x.shape[2] == cfg.grid_x
        model = HeatTreatmentFNO3D(cfg).to(device)
        with torch.no_grad():
            out = model(x.to(device))
        print(f"  Forward OK: {out.shape}")
        loss = F.mse_loss(out, y.to(device))
        loss.backward()
        print(f"  Backward OK: loss={loss.item():.6f}")
        print(f"  === ALL CHECKS PASSED ===")
        return

    model = HeatTreatmentFNO3D(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)
    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)

    print(f"  {'Ep':>4} | {'TrLoss':>9} | {'TrData':>9} | {'TrPhys':>9} | "
          f"{'VaData':>9} | {'VaPhys':>9} | "
          f"{'MAE':>6} | {'R2':>7} | {'W5K':>5} | {'lam':>5}")
    print(f"  {'-'*95}")

    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        lam = get_physics_lambda(epoch, cfg.n_epochs)

        # Warmup
        if epoch <= 5:
            lr = args.lr * (0.1 + 0.9 * epoch / 5)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        model.train()
        total_loss, total_data, total_phys, nb = 0.0, 0.0, 0.0, 0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss_data = F.mse_loss(pred, y)

            if lam > 1e-6:
                L_phys = physics_loss_3d(pred, x, train_ds.dT_std, train_ds.dT_mean)
                loss = (1 - lam) * loss_data + lam * L_phys
                total_phys += L_phys.item()
            else:
                loss = loss_data

            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_data += loss_data.item()
            nb += 1

        tr_loss = total_loss / max(nb, 1)
        tr_data = total_data / max(nb, 1)
        tr_phys = total_phys / max(nb, 1)

        val_m = evaluate(model, val_loader, device, train_ds, lam=lam)
        if epoch > 5:
            scheduler.step(val_m["loss"])

        is_best = ckpt_mgr.update(model, optimizer, scheduler, epoch, val_m)
        if epoch % 10 == 0:
            ckpt_mgr.save_periodic(model, optimizer, scheduler, epoch, val_m)

        tag = " *" if is_best else ""
        print(f"  {epoch:>4} | {tr_loss:>9.5f} | {tr_data:>9.5f} | "
              f"{tr_phys:>9.5f} | {val_m['loss_data']:>9.5f} | "
              f"{val_m['loss_phys']:>9.5f} | "
              f"{val_m['mae']:>6.2f} | {val_m['r2']:>7.4f} | "
              f"{val_m['within_5K']:>5.1f} | {lam:>5.3f}{tag}")

    print(f"\n  Done in {(time.time()-t0)/60:.1f} min")
    print(f"  Best MAE: {ckpt_mgr.best_mae:.3f}K (epoch {ckpt_mgr.best_epoch})")


if __name__ == "__main__":
    main()
PYEOF

echo "    OK"

# ── 5. SLURM SCRIPTS ─────────────────────────────────────────

echo "  [5/6] Writing run scripts..."

cat > run_alvis_fno.sh << 'SHEOF'
#!/bin/bash
#SBATCH --job-name=heat_fno_3d
#SBATCH --account=NAISS2026-4-525
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno3d_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/fno3d_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs
mkdir -p /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/checkpoints
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== 3D FNO TRAINING ==="
echo "=== Start: $(date) ==="

apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --epochs 100 --lr 1e-3 --batch 4

echo "=== DONE: $(date) ==="
SHEOF

cat > run_sanity_test_fno.sh << 'SHEOF'
#!/bin/bash
#SBATCH --job-name=fno_test
#SBATCH --account=NAISS2026-4-525
#SBATCH --partition=alvis
#SBATCH --output=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/test_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official/outputs/logs/test_err_%j.log
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== 3D FNO SANITY TEST ==="
apptainer exec --nv \
  /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -u train.py --test --device cuda

echo "=== EXIT CODE: $? ==="
SHEOF

echo "    OK"

# ── 6. VERIFY ─────────────────────────────────────────────────

echo "  [6/6] Verification..."

for f in configs/fno_config.py data/dataset.py models/fno_model.py train.py \
         run_alvis_fno.sh run_sanity_test_fno.sh; do
    if [ -f "$f" ]; then
        echo "    OK  $f"
    else
        echo "    FAIL  $f missing"
    fi
done

# Syntax check
for f in configs/fno_config.py data/dataset.py models/fno_model.py train.py; do
    python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null && \
        echo "    OK  $f syntax" || echo "    FAIL  $f syntax"
done

echo ""
echo "============================================"
echo "  3D FNO REWRITE COMPLETE"
echo "============================================"
echo ""
echo "  Architecture:"
echo "    Input:  (batch, 19, 24, 40, 44)"
echo "    19 channels: T, T_set, 12 region masks,"
echo "    time, is_heater, kappa, Cp, rho"
echo "    FNO: 4 layers, 64 latent, modes [12,16,16]"
echo "    Output: (batch, 1, 24, 40, 44) = delta_T"
echo ""
echo "  Grid: 24x40x44 = 42,240 voxels"
echo "  Covers: x=[0,0.206] y=[0,0.36] z=[0,0.39]"
echo "  Resolution: ~9mm per voxel"
echo ""
echo "  Step 1: Sanity test (30 min):"
echo "    sbatch run_sanity_test_fno.sh"
echo "    cat outputs/logs/test_*.log"
echo ""
echo "  Step 2: Train (24 hrs):"
echo "    sbatch run_alvis_fno.sh"
