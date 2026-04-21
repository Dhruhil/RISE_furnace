"""
Measure physics loss magnitudes on v2 dataset.
Runs 30 batches, logs MSE + each of 4 physics terms separately.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from configs.base_config import CONFIG as cfg, SIGMA_SB, EMISSIVITY_STEEL, H_CONV, CHAR_THICKNESS
from data.dataset_unified import UnifiedGNNDataset
from models.gnn_unified import HeatTreatmentGNN


def measure_physics_terms(pred, batch, T_std_ds, T_mean, dt=10.0, device='cuda'):
    """Return dict with MSE + each of 4 physics terms."""
    target_norm = batch.y.to(device)
    mse = F.mse_loss(pred.squeeze(-1), target_norm).item()
    
    T_next = pred.squeeze(-1) * T_std_ds + T_mean
    T_now = batch.T_current.to(device)
    dT_dt = (T_next - T_now) / dt
    
    T_set = batch.T_set_raw.to(device)
    kappa = batch.kappa_raw.to(device)
    Cp = batch.Cp_raw.to(device)
    rho = batch.rho_raw.to(device)
    non_heater = (~batch.is_heater.bool()).float()
    
    alpha = kappa / (rho * Cp + 1e-8)
    rho_cp = rho * Cp
    delta = CHAR_THICKNESS
    
    # Conduction (Fourier)
    src_i, dst_i = batch.edge_index[0], batch.edge_index[1]
    N = T_now.shape[0]
    T_diff = T_now[dst_i] - T_now[src_i]
    lap_T = torch.zeros(N, device=device, dtype=T_now.dtype)
    degree = torch.zeros(N, device=device, dtype=T_now.dtype)
    lap_T.scatter_add_(0, dst_i, T_diff)
    degree.scatter_add_(0, dst_i, torch.ones_like(T_diff))
    lap_T = lap_T / degree.clamp(min=1.0)
    dT_dt_cond = alpha * lap_T
    L_cond = ((dT_dt - dT_dt_cond) * non_heater).pow(2).mean().item()
    
    # Convection (Newton)
    dT_dt_conv = H_CONV * (T_set - T_now) / (rho_cp * delta + 1e-8)
    conv_res = (dT_dt - dT_dt_conv) / 10.0
    L_conv_match = (conv_res * non_heater).pow(2).mean().item()
    overshoot = F.relu(T_next - T_set) * non_heater
    L_overshoot = (overshoot / T_set.clamp(min=300)).pow(2).mean().item()
    L_conv = 0.5 * L_conv_match + 0.5 * L_overshoot
    
    # Radiation (Stefan-Boltzmann)
    dT_dt_rad = (EMISSIVITY_STEEL * SIGMA_SB *
                 (T_set.pow(4) - T_now.pow(4)) / (rho_cp * delta + 1e-8))
    rad_res = (dT_dt - dT_dt_rad) / 100.0
    L_rad = (rad_res * non_heater).pow(2).mean().item()
    
    # Energy balance
    dT_dt_total = dT_dt_cond + dT_dt_conv + dT_dt_rad
    L_eng = (((dT_dt - dT_dt_total) * non_heater).pow(2).mean() /
             (T_std_ds ** 2 + 1e-8)).item()
    
    return {
        'mse': mse,
        'L_cond': L_cond,
        'L_conv': L_conv,
        'L_rad': L_rad,
        'L_eng': L_eng,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {cfg.all_regions_dataset_path}")
    print()
    
    print("Loading dataset...")
    train_ds = UnifiedGNNDataset(cfg, split="train")
    print(f"  Train samples: {len(train_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                             num_workers=4, pin_memory=True)
    
    print("\nBuilding model...")
    model = HeatTreatmentGNN(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    T_mean = train_ds.T_mean
    T_std_ds = train_ds.T_std
    print(f"  T_mean: {T_mean:.2f} K")
    print(f"  T_std:  {T_std_ds:.2f} K")
    print()
    
    print("=" * 70)
    print("MEASURING LOSS MAGNITUDES (30 batches)")
    print("=" * 70)
    
    all_measurements = {k: [] for k in ['mse', 'L_cond', 'L_conv', 'L_rad', 'L_eng']}
    
    model.train()
    t0 = time.time()
    
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        
        m = measure_physics_terms(pred, batch, T_std_ds, T_mean, device=device)
        for k, v in m.items():
            all_measurements[k].append(v)
        
        target_norm = batch.y.to(device)
        loss = F.mse_loss(pred.squeeze(-1), target_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Step {i+1:>4d}  |  MSE={m['mse']:.6f}  |  "
                  f"L_cond={m['L_cond']:.6f}  |  L_conv={m['L_conv']:.6f}  |  "
                  f"L_rad={m['L_rad']:.6f}  |  L_eng={m['L_eng']:.6f}  "
                  f"[{elapsed:.1f}s]")
        
        if i >= 29:
            break
    
    print()
    print("=" * 70)
    print("SUMMARY — LOSS MAGNITUDES (averaged over 30 batches)")
    print("=" * 70)
    
    stats = {}
    for k in ['mse', 'L_cond', 'L_conv', 'L_rad', 'L_eng']:
        arr = np.array(all_measurements[k])
        stats[k] = {'mean': arr.mean(), 'std': arr.std(),
                    'min': arr.min(), 'max': arr.max()}
        print(f"  {k:8s}: mean={stats[k]['mean']:.6f}  "
              f"std={stats[k]['std']:.6f}  min={stats[k]['min']:.6f}  "
              f"max={stats[k]['max']:.6f}")
    
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    mse = stats['mse']['mean']
    print(f"\nMSE magnitude: {mse:.6f}")
    
    total_phys = sum(stats[k]['mean'] for k in ['L_cond', 'L_conv', 'L_rad', 'L_eng'])
    print(f"\nPhysics term magnitudes:")
    for k in ['L_cond', 'L_conv', 'L_rad', 'L_eng']:
        v = stats[k]['mean']
        frac = v / total_phys if total_phys > 0 else 0
        print(f"  {k}: {v:.6f}  ({frac*100:>5.1f}% of total)")
    
    # Current weighted physics loss
    L_phys_weighted = (0.4 * stats['L_cond']['mean'] +
                       0.3 * stats['L_conv']['mean'] +
                       0.2 * stats['L_rad']['mean'] +
                       0.1 * stats['L_eng']['mean'])
    print(f"\nWeighted physics loss (current 0.4/0.3/0.2/0.1): {L_phys_weighted:.6f}")
    print(f"Ratio MSE / Physics: {mse / max(L_phys_weighted, 1e-12):.2f}")
    
    print()
    print("LAMBDA RECOMMENDATIONS (physics as X% of MSE contribution):")
    for target_ratio in [0.01, 0.05, 0.1, 0.25, 0.5]:
        lam_rec = target_ratio * mse / max(L_phys_weighted, 1e-12)
        print(f"  Physics as {target_ratio*100:.0f}% of MSE: lambda = {lam_rec:.6f}")
    
    print()
    print("EQUAL-CONTRIBUTION WEIGHTS (each term equally scaled):")
    terms = {k: stats[k]['mean'] for k in ['L_cond', 'L_conv', 'L_rad', 'L_eng']}
    weights_raw = {k: 1.0 / max(v, 1e-12) for k, v in terms.items()}
    total_w = sum(weights_raw.values())
    for k, w in weights_raw.items():
        norm_w = w / total_w
        print(f"  {k}: {norm_w:.3f}")
    
    print()
    print("=" * 70)
    print("DONE — Use these values to configure final hyperparameters")
    print("=" * 70)


if __name__ == "__main__":
    main()
