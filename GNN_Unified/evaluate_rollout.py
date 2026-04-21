"""
Rollout evaluation on TEST dataset.

Runs autoregressive rollout on 7 test cases (never seen during training).
This is REAL deployment accuracy for digital twin thesis.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from configs.base_config import CONFIG as cfg
from data.dataset_unified import UnifiedDataset
from models.gnn_unified import HeatTreatmentGNN


CHECKPOINT_PATH = Path("/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/checkpoints_unified/best_model.pt")
OUTPUT_DIR = Path("/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/rollout_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rollout_one_case(model, test_ds, sim_idx, device):
    sim = test_ds._simulations[sim_idx]
    case_name = sim["name"]
    
    T_gt = torch.tensor(sim["T"], dtype=torch.float32).to(device)
    n_timesteps = T_gt.shape[0]
    
    T_mean = test_ds.T_mean
    T_std = test_ds.T_std
    
    batch = test_ds.get_graph_at_timestep(sim_idx, 0)
    batch = batch.to(device)
    is_heater = batch.is_heater.bool()
    
    predictions = [T_gt[0].cpu().numpy()]
    
    model.eval()
    with torch.no_grad():
        for t in range(n_timesteps - 1):
            T_current_norm = (batch.T_current - T_mean) / T_std
            batch.x[:, 0] = T_current_norm
            
            pred_norm = model(batch).squeeze(-1)
            T_next = pred_norm * T_std + T_mean
            T_next = torch.where(is_heater, T_gt[t+1], T_next)
            
            batch.T_current = T_next
            predictions.append(T_next.cpu().numpy())
    
    pred_traj = np.array(predictions)
    gt_traj = T_gt.cpu().numpy()
    
    heater_mask = is_heater.cpu().numpy().astype(bool)
    err = pred_traj[:, ~heater_mask] - gt_traj[:, ~heater_mask]
    
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    max_err = float(np.abs(err).max())
    mae_per_step = np.abs(err).mean(axis=1)
    
    return {
        'case': case_name.split('_')[0],
        'full_name': case_name,
        'n_steps': int(n_timesteps),
        'mae': mae,
        'rmse': rmse,
        'max_err': max_err,
        'final_mae': float(mae_per_step[-1]),
        'mae_at_50': float(mae_per_step[min(49, n_timesteps-1)]),
        'mae_at_200': float(mae_per_step[min(199, n_timesteps-1)]),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 72)
    print("ROLLOUT EVALUATION ON TEST DATASET")
    print("=" * 72)
    print(f"Device:     {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()
    
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: best_model.pt not found at {CHECKPOINT_PATH}")
        print("Run training first!")
        sys.exit(1)
    
    print("Loading TEST dataset (cases never seen during training)...")
    test_ds = UnifiedDataset(cfg, split="test")
    print(f"  Test cases: {len(test_ds.sim_indices)}")
    print(f"  T_mean:     {test_ds.T_mean:.2f} K")
    print(f"  T_std:      {test_ds.T_std:.2f} K")
    print()
    
    print("Loading best model from training...")
    model = HeatTreatmentGNN(cfg).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded epoch: {ckpt.get('epoch', '?')}")
        print(f"  Best val MAE: {ckpt.get('best_mae', '?')}")
    else:
        model.load_state_dict(ckpt)
    print(f"  Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    print("=" * 72)
    print("PER-CASE ROLLOUT RESULTS (autoregressive, 400 steps each)")
    print("=" * 72)
    print(f"{'Case':<10} {'Steps':>6} {'MAE(K)':>8} {'RMSE(K)':>9} "
          f"{'Max(K)':>8} {'End(K)':>8} {'@50':>7} {'@200':>7}")
    print("-" * 72)
    
    per_case = []
    for sim_idx in test_ds.sim_indices:
        r = rollout_one_case(model, test_ds, sim_idx, device)
        per_case.append(r)
        print(f"{r['case']:<10} {r['n_steps']:>6} {r['mae']:>8.2f} {r['rmse']:>9.2f} "
              f"{r['max_err']:>8.2f} {r['final_mae']:>8.2f} "
              f"{r['mae_at_50']:>7.2f} {r['mae_at_200']:>7.2f}")
    
    all_mae = [r['mae'] for r in per_case]
    all_rmse = [r['rmse'] for r in per_case]
    all_max = [r['max_err'] for r in per_case]
    all_final = [r['final_mae'] for r in per_case]
    
    print()
    print("=" * 72)
    print("SUMMARY — REAL DEPLOYMENT ACCURACY (test set, unseen during training)")
    print("=" * 72)
    print(f"Test cases:        {len(per_case)}")
    print()
    print(f"Mean Rollout MAE:  {np.mean(all_mae):.2f} K   (std: {np.std(all_mae):.2f})")
    print(f"Mean Rollout RMSE: {np.mean(all_rmse):.2f} K")
    print(f"Mean Max Error:    {np.mean(all_max):.2f} K")
    print(f"Mean End MAE:      {np.mean(all_final):.2f} K")
    print()
    print(f"Best case MAE:     {min(all_mae):.2f} K")
    print(f"Worst case MAE:    {max(all_mae):.2f} K")
    print()
    
    target_5K = sum(1 for m in all_mae if m < 5)
    print("=" * 72)
    print("THESIS RESULTS")
    print("=" * 72)
    print(f"Cases with Rollout MAE < 5K:  {target_5K}/{len(per_case)}  "
          f"({100*target_5K/len(per_case):.0f}%)")
    print()
    print("Summary for thesis Results section:")
    print(f"  Per-step MAE (val, teacher-forced):    0.XX K (from training log)")
    print(f"  Rollout MAE (test, autoregressive):   {np.mean(all_mae):.2f} K")
    print(f"  Rollout RMSE:                         {np.mean(all_rmse):.2f} K")
    print(f"  Cases meeting 5K accuracy:            {target_5K}/{len(per_case)}")
    print()
    
    summary = {
        'mean_mae': float(np.mean(all_mae)),
        'mean_rmse': float(np.mean(all_rmse)),
        'mean_max_err': float(np.mean(all_max)),
        'mean_final_mae': float(np.mean(all_final)),
        'best_mae': float(min(all_mae)),
        'worst_mae': float(max(all_mae)),
        'cases_under_5K': int(target_5K),
        'total_cases': len(per_case),
        'per_case': per_case,
    }
    
    output_file = OUTPUT_DIR / 'rollout_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved: {output_file}")
    print()


if __name__ == "__main__":
    main()
