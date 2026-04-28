"""
Per-step rollout error analysis for the trained FNO model.
Tells you EXACTLY how many seconds the FNO predicts accurately.

Uses your existing rollout_fno3d() function from models/rollout.py.
"""
import sys, os, glob, json
sys.path.insert(0, ".")

import numpy as np
import torch

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, REGION_IDS
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_fno3d


def per_step_mae(T_pred, T_true, region_mask):
    """Compute MAE per timestep for one region."""
    n_steps = T_pred.shape[0]
    out = np.zeros(n_steps)
    for s in range(n_steps):
        diff = np.abs(T_pred[s][region_mask] - T_true[s][region_mask])
        out[s] = np.mean(diff)
    return out


def find_horizon(times, mae, threshold):
    """First time where mean error exceeds threshold (relative to start)."""
    for i, e in enumerate(mae):
        if e > threshold:
            return times[i]
    return times[-1]


# ─── Setup ─────────────────────────────────────────────────────────
TAG = open(".retrain_tag").read().strip()
ckpt_path = f"outputs/{TAG}/checkpoints/best_model.pt"
print(f"Checkpoint: {ckpt_path}")
assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = CONFIG

# ─── Load test dataset ─────────────────────────────────────────────
print("\nLoading test dataset...")
dataset = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
sim_indices = dataset.sim_indices
print(f"  {len(sim_indices)} test sims: {sim_indices}")

# ─── Load model ────────────────────────────────────────────────────
print("\nLoading FNO model...")
model = HeatTreatmentFNO3D(cfg).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt["model_state"]
model.load_state_dict(state)
model.eval()

# ─── Per-step rollout for every test sim ───────────────────────────
print("\n" + "="*70)
print("  PER-STEP ROLLOUT ERROR — STEEL CYLINDER & INNER CAVITY")
print("="*70)

results_per_sim = {}

for sim_i in sim_indices:
    sim = dataset._simulations[sim_i]
    static = dataset._static_grids[sim_i]
    region_grid = static["interp_fields"]["region_id"].squeeze(-1)

    steel_id = REGION_IDS["steel_cylinder"]
    air_id   = REGION_IDS["inner_box"]
    steel_mask = (region_grid == steel_id)
    air_mask   = (region_grid == air_id)

    print(f"\nRolling out sim {sim_i} (T_set={sim['T_set']:.0f}K)...")
    T_pred, T_true = rollout_fno3d(model, dataset, sim_i, device=device, start_t=20)

    # Per-step time axis (relative to rollout start at t=200s)
    times_full = sim["times"]
    rollout_times = times_full[20:20 + T_pred.shape[0]]
    rollout_dt = rollout_times - rollout_times[0]   # 0, 10, 20, ...

    steel_mae_traj = per_step_mae(T_pred, T_true, steel_mask)
    air_mae_traj   = per_step_mae(T_pred, T_true, air_mask)

    results_per_sim[sim_i] = {
        "T_set": float(sim["T_set"]),
        "rollout_dt": rollout_dt.tolist(),
        "steel_mae": steel_mae_traj.tolist(),
        "air_mae":   air_mae_traj.tolist(),
    }

    # Print key checkpoints
    n = len(rollout_dt)
    pts = [1, min(10, n-1), min(30, n-1), min(60, n-1),
           min(120, n-1), min(200, n-1), n-1]
    pts = sorted(set(pts))
    print(f"  Step    Time    Steel MAE   Air MAE")
    for p in pts:
        print(f"  {p:>4}  {rollout_dt[p]:>5.0f}s    "
              f"{steel_mae_traj[p]:>6.2f} K   {air_mae_traj[p]:>6.2f} K")

# ─── Aggregate across sims ──────────────────────────────────────────
print("\n" + "="*70)
print("  HORIZON ANALYSIS — MEAN ACROSS ALL TEST SIMS")
print("="*70)

# Find the shortest rollout length (some sims may differ)
min_len = min(len(r["rollout_dt"]) for r in results_per_sim.values())
common_times = np.array(list(results_per_sim.values())[0]["rollout_dt"][:min_len])

steel_stack = np.stack([
    np.array(r["steel_mae"][:min_len]) for r in results_per_sim.values()
])  # (n_sims, n_steps)
air_stack = np.stack([
    np.array(r["air_mae"][:min_len]) for r in results_per_sim.values()
])

mean_steel = steel_stack.mean(axis=0)
mean_air   = air_stack.mean(axis=0)
std_steel  = steel_stack.std(axis=0)

print(f"\n📊 STEEL CYLINDER — when does mean MAE exceed threshold?")
for thresh in [5, 10, 20, 30, 50, 75, 100]:
    h = find_horizon(common_times, mean_steel, thresh)
    print(f"  MAE < {thresh:>3} K  up to  {h:>5.0f}s into rollout"
          f"  ({h/60:.1f} min)  →  absolute t = {200+h:.0f}s")

print(f"\n📊 INNER CAVITY — when does mean MAE exceed threshold?")
for thresh in [5, 10, 20, 30, 50, 75, 100]:
    h = find_horizon(common_times, mean_air, thresh)
    print(f"  MAE < {thresh:>3} K  up to  {h:>5.0f}s into rollout"
          f"  ({h/60:.1f} min)  →  absolute t = {200+h:.0f}s")

# ─── Key findings table ─────────────────────────────────────────────
print("\n" + "="*70)
print("  KEY FINDINGS — PER-STEP ERROR PROGRESSION")
print("="*70)
print(f"  {'Step':>4}  {'dt':>6}  {'abs_t':>6}  {'Steel MAE':>14}  {'Air MAE':>10}")
print(f"  {'─'*52}")
for s in [1, 5, 10, 20, 30, 60, 100, 150, 200, 256]:
    if s < min_len:
        print(f"  {s:>4}  {common_times[s]:>5.0f}s  "
              f"{200+common_times[s]:>5.0f}s  "
              f"{mean_steel[s]:>6.2f} ± {std_steel[s]:>4.2f} K  "
              f"{mean_air[s]:>6.2f} K")

# ─── Save full data ────────────────────────────────────────────────
out = {
    "checkpoint": ckpt_path,
    "n_sims": len(sim_indices),
    "common_times": common_times.tolist(),
    "mean_steel_mae": mean_steel.tolist(),
    "std_steel_mae":  std_steel.tolist(),
    "mean_air_mae":   mean_air.tolist(),
    "per_sim": {str(k): v for k, v in results_per_sim.items()},
}
out_path = f"outputs/{TAG}/per_step_horizon.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n💾 Detailed data saved: {out_path}")

print("\n✅ Analysis complete.")
