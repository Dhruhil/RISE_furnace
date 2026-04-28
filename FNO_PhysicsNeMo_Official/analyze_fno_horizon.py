"""
Per-step rollout error analysis for the trained FNO model.
Tells you EXACTLY how many seconds the FNO predicts accurately.
No retraining required - uses existing checkpoint.
"""
import sys, os
sys.path.insert(0, ".")

import numpy as np
import torch
import json
from pathlib import Path
from configs.fno_config import CONFIG
from data.dataset import HeatTreatmentDataset
from models.fno_model import HeatTreatmentFNO3D
from utils.checkpoint import CheckpointManager

# ─── Setup ─────────────────────────────────────────────────────────
import glob

# Find latest v5 FIX checkpoint
TAG = open(".retrain_tag").read().strip()
CKPT_DIR = f"outputs/{TAG}/checkpoints"
ckpts = sorted(glob.glob(f"{CKPT_DIR}/best_*.pt"))
if not ckpts:
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/*.pt"))
ckpt_path = ckpts[-1]
print(f"Using checkpoint: {ckpt_path}")

device = torch.device("cuda")
cfg = CONFIG

# Load test dataset
ds = HeatTreatmentDataset(
    cfg.dataset_path, cfg, "test", "evaluation",
)

# Load model
model = HeatTreatmentFNO3D(
    in_channels=8, out_channels=1,
    modes=cfg.fno_modes, n_layers=cfg.fno_layers,
    latent_dim=cfg.fno_latent,
    grid_shape=cfg.grid_shape,
).to(device)

checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ─── Per-step rollout ──────────────────────────────────────────────
print("="*70)
print("  PER-STEP ROLLOUT ERROR ANALYSIS")
print("="*70)

results = {}  # sim_idx -> list of (t, steel_mae, air_mae)

for sim_idx in ds.sim_indices:
    print(f"\nProcessing sim {sim_idx}...")
    
    # Get full ground truth trajectory
    sim_data = ds._get_full_sim(sim_idx)
    times = sim_data['times']  # (N_t,)
    T_gt = sim_data['T_gridded']  # (N_t, 30, 36, 54)
    region_grid = sim_data['region_grid']  # (30, 36, 54)
    static_features = sim_data['static_features']  # (8, 30, 36, 54)
    
    # Find start step (t=200s)
    start_idx = np.argmin(np.abs(times - 200.0))
    end_idx = len(times) - 1
    
    # Steel and air masks
    steel_mask = (region_grid == 0)  # steel_cylinder region_id=0
    air_mask = (region_grid == 1)    # inner_box region_id=1
    
    # Initialize with ground truth
    T_norm = (T_gt[start_idx] - cfg.T_mean) / cfg.T_std
    
    per_step_steel = []
    per_step_air = []
    per_step_t = []
    
    with torch.no_grad():
        for k in range(start_idx + 1, end_idx + 1):
            t_current = times[k - 1]
            t_next = times[k]
            
            # Build input
            input_grid = static_features.copy()
            input_grid[0] = T_norm
            input_grid[3] = t_current / cfg.t_total  # time channel
            
            x = torch.from_numpy(input_grid[None]).float().to(device)
            
            # Forward pass
            T_next_norm = model(x).squeeze().cpu().numpy()
            
            # Reset heaters to ground truth (Dirichlet)
            heater_mask = (region_grid >= 2) & (region_grid <= 9)
            T_next = T_next_norm * cfg.T_std + cfg.T_mean
            T_next[heater_mask] = T_gt[k][heater_mask]
            
            # Compute per-region MAE for THIS step
            steel_mae = np.mean(np.abs(T_next[steel_mask] - T_gt[k][steel_mask]))
            air_mae = np.mean(np.abs(T_next[air_mask] - T_gt[k][air_mask]))
            
            per_step_steel.append(steel_mae)
            per_step_air.append(air_mae)
            per_step_t.append(t_next)
            
            # Feed back
            T_norm = (T_next - cfg.T_mean) / cfg.T_std
    
    results[sim_idx] = {
        't': per_step_t,
        'steel_mae': per_step_steel,
        'air_mae': per_step_air,
    }

# ─── Aggregate ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  HORIZON ANALYSIS — WHEN DOES FNO BREAK?")
print("="*70)

# Common time grid
all_times = sorted(set(t for r in results.values() for t in r['t']))

# Mean across sims at each time
mean_steel_mae = []
mean_air_mae = []
for t in all_times:
    steel_vals = [r['steel_mae'][r['t'].index(t)] for r in results.values() if t in r['t']]
    air_vals = [r['air_mae'][r['t'].index(t)] for r in results.values() if t in r['t']]
    mean_steel_mae.append(np.mean(steel_vals))
    mean_air_mae.append(np.mean(air_vals))

# ─── Find "good prediction" horizons ────────────────────────────────
def find_horizon(times, errors, threshold):
    """Find the latest time at which mean error <= threshold."""
    for i, e in enumerate(errors):
        if e > threshold:
            return times[i] - 200.0  # seconds from start
    return times[-1] - 200.0

print("\n📊 STEEL CYLINDER:")
for thresh in [5, 10, 20, 30, 50]:
    horizon = find_horizon(all_times, mean_steel_mae, thresh)
    print(f"  MAE < {thresh:>3} K up to: {horizon:>5.0f} s into rollout"
          f"  ({horizon/60:.1f} minutes)")

print("\n📊 INNER CAVITY (AIR):")
for thresh in [5, 10, 20, 30, 50]:
    horizon = find_horizon(all_times, mean_air_mae, thresh)
    print(f"  MAE < {thresh:>3} K up to: {horizon:>5.0f} s into rollout"
          f"  ({horizon/60:.1f} minutes)")

# ─── Save results ────────────────────────────────────────────────────
out = {
    'times': all_times,
    'mean_steel_mae': mean_steel_mae,
    'mean_air_mae': mean_air_mae,
    'per_sim': {str(k): v for k, v in results.items()},
}
out_path = f"outputs/{TAG}/per_step_horizon_analysis.json"
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n💾 Saved detailed analysis: {out_path}")

# ─── Print key findings ──────────────────────────────────────────────
print("\n" + "="*70)
print("  KEY FINDINGS")
print("="*70)
print(f"\n  At step    1 (t=210s):  Steel MAE = {mean_steel_mae[0]:.2f} K")
print(f"  At step   10 (t=300s):  Steel MAE = {mean_steel_mae[9]:.2f} K")
print(f"  At step   50 (t=700s):  Steel MAE = {mean_steel_mae[49]:.2f} K")
print(f"  At step  100 (t=1200s): Steel MAE = {mean_steel_mae[99]:.2f} K")
print(f"  At step  200 (t=2200s): Steel MAE = {mean_steel_mae[199]:.2f} K")
print(f"  At step  256 (t=2760s): Steel MAE = {mean_steel_mae[255]:.2f} K"
      f"  ← end of in-horizon")
print(f"  At step  326 (t=3460s): Steel MAE = {mean_steel_mae[-1]:.2f} K"
      f"  ← end of extrapolation")
