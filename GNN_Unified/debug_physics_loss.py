"""
Debug script: measure individual physics loss term magnitudes.
Shows which term is actually causing TrPhys = 115.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from configs.base_config import CONFIG as cfg
from configs.base_config import SIGMA_SB, EMISSIVITY_STEEL, H_CONV, CHAR_THICKNESS
from data.dataset_unified import UnifiedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load dataset
print("\n=== Loading dataset ===")
ds = UnifiedDataset(cfg.all_regions_dataset_path, cfg, "train", "training")
print(f"Samples: {len(ds)}")

# Get one batch
print("\n=== Getting one sample ===")
from torch_geometric.loader import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
batch = next(iter(loader)).to(device)

# Compute each term separately
T_std_ds = ds.T_std
T_mean = ds.T_mean
dt = 10.0

print(f"\nT_mean={T_mean:.2f} K, T_std={T_std_ds:.2f} K")

# Create dummy prediction (use T_current + small noise)
pred = batch.x[:, 0:1].clone()
pred_denorm = pred.squeeze(-1) * T_std_ds + T_mean
T_now = batch.T_current
dT_dt = (pred_denorm - T_now) / dt

# Node properties
T_set = batch.T_set_raw
kappa = batch.kappa_raw
Cp = batch.Cp_raw
rho = batch.rho_raw
non_heater = (~batch.is_heater.bool()).float()

alpha = kappa / (rho * Cp + 1e-8)
rho_cp = rho * Cp
delta = CHAR_THICKNESS

# 1. CONDUCTION
src_i, dst_i = batch.edge_index[0], batch.edge_index[1]
N = T_now.shape[0]
T_diff = T_now[dst_i] - T_now[src_i]
lap_T = torch.zeros(N, device=device, dtype=T_now.dtype)
degree = torch.zeros(N, device=device, dtype=T_now.dtype)
lap_T.scatter_add_(0, dst_i, T_diff)
degree.scatter_add_(0, dst_i, torch.ones_like(T_diff))
lap_T = lap_T / degree.clamp(min=1.0)
dT_dt_cond = alpha * lap_T
cond_res_raw = (dT_dt - dT_dt_cond) * non_heater
cond_res = (dT_dt - dT_dt_cond) / 10.0 * non_heater
L_cond_noNorm = (cond_res_raw).pow(2).mean()
L_cond = (cond_res).pow(2).mean()

# 2. CONVECTION
dT_dt_conv = H_CONV * (T_set - T_now) / (rho_cp * delta + 1e-8)
conv_res = (dT_dt - dT_dt_conv) / 100.0
L_conv_match = (conv_res * non_heater).pow(2).mean()
overshoot = F.relu(pred_denorm - T_set) * non_heater
L_overshoot = (overshoot / T_set.clamp(min=300)).pow(2).mean()
L_conv = 0.5 * L_conv_match + 0.5 * L_overshoot

# 3. RADIATION
dT_dt_rad = (EMISSIVITY_STEEL * SIGMA_SB *
             (T_set.pow(4) - T_now.pow(4)) / (rho_cp * delta + 1e-8))
rad_res = (dT_dt - dT_dt_rad) / 1000.0
L_rad = (rad_res * non_heater).pow(2).mean()

# 4. ENERGY
dT_dt_total = dT_dt_cond + dT_dt_conv + dT_dt_rad
L_eng = ((dT_dt - dT_dt_total) * non_heater).pow(2).mean() / (T_std_ds ** 2 + 1e-8)

# Print magnitudes
print("\n" + "=" * 60)
print("PHYSICS LOSS TERM MAGNITUDES")
print("=" * 60)
print(f"L_cond (WITHOUT /10):  {L_cond_noNorm.item():.4f}")
print(f"L_cond (WITH /10):     {L_cond.item():.4f}  ← normalized")
print(f"L_conv (WITH /100):     {L_conv.item():.4f}")
print(f"L_conv_match alone:    {L_conv_match.item():.4f}")
print(f"L_overshoot alone:     {L_overshoot.item():.4f}")
print(f"L_rad (WITH /1000):     {L_rad.item():.6f}")
print(f"L_eng (WITH /T_std²):  {L_eng.item():.6f}")
print()
print("=" * 60)
print("WEIGHTED SUM")
print("=" * 60)
weighted = 0.4 * L_cond + 0.3 * L_conv + 0.2 * L_rad + 0.1 * L_eng
print(f"0.4 × L_cond = {0.4 * L_cond.item():.4f}")
print(f"0.3 × L_conv = {0.3 * L_conv.item():.4f}")
print(f"0.2 × L_rad  = {0.2 * L_rad.item():.6f}")
print(f"0.1 × L_eng  = {0.1 * L_eng.item():.6f}")
print(f"TOTAL        = {weighted.item():.4f}")
print()
print("Expected TrPhys in training log: around", weighted.item())

# Raw dT_dt magnitude
print()
print("=" * 60)
print("dT/dt magnitude analysis")
print("=" * 60)
print(f"dT_dt (target):     mean={dT_dt.abs().mean().item():.4f}, max={dT_dt.abs().max().item():.4f}")
print(f"dT_dt_cond:         mean={dT_dt_cond.abs().mean().item():.4f}, max={dT_dt_cond.abs().max().item():.4f}")
print(f"dT_dt_conv:         mean={dT_dt_conv.abs().mean().item():.4f}, max={dT_dt_conv.abs().max().item():.4f}")
print(f"dT_dt_rad:          mean={dT_dt_rad.abs().mean().item():.4f}, max={dT_dt_rad.abs().max().item():.4f}")
