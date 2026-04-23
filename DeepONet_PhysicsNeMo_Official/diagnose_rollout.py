"""
Compare rollout step-by-step vs training data loader output.
"""
import sys, torch, numpy as np
sys.path.insert(0, ".")
from configs.deeponet_config import CONFIG
from data.dataset import DeepONetDataset, get_deeponet_eval_dataset
from models.deeponet_model import HeatTreatmentDeepONet
from scipy.interpolate import NearestNDInterpolator

device = torch.device("cuda")

# Build eval dataset (test split)
ds = get_deeponet_eval_dataset(CONFIG)
model = HeatTreatmentDeepONet(CONFIG).to(device)
ckpt = torch.load("outputs/checkpoints/best.pt", map_location=device, weights_only=False)
state = ckpt["model"]
model.load_state_dict(state)
model.eval()

sim_i = ds.sim_indices[0]
sim = ds._simulations[sim_i]
sens = ds._static_sensors[sim_i]
T_mean, T_std = ds.T_mean, ds.T_std
Tset_norm_val = (sim["T_set"] - ds.Tset_mean) / ds.Tset_std
t_i = 100
t_val = sim["times"][t_i]

print(f"\n=== DIAGNOSTIC ===")
print(f"sim_i={sim_i}, t_i={t_i}, T_set={sim['T_set']:.1f} K")
print(f"T_mean={T_mean:.2f}, T_std={T_std:.2f}")
print(f"Tset_mean={ds.Tset_mean:.2f}, Tset_std={ds.Tset_std:.2f}")
print(f"Tset_norm = {Tset_norm_val:.4f}")
print(f"t_norm    = {t_val/CONFIG.t_total:.4f}  (t_val={t_val:.1f}, t_total={CONFIG.t_total})")

# --- PATH A: What the dataset produces for training sample at (sim_i, t_i) ---
# We need to find the sample index for (sim_i, t_i) — NOTE: eval split doesn't match train split
# So we'll build a fresh dataset in "train" split mode just to see __getitem__ output for this case.
# But test case is NOT in train split. So: directly call __getitem__ logic ourselves.

T_t_gt = sim["T_all"][t_i]
T_tp1_gt = sim["T_all"][t_i + 1]

# This mimics dataset.__getitem__ exactly (EVAL mode: no noise)
interp_T = NearestNDInterpolator(sim["coords"], T_t_gt)
T_sens_path_A = interp_T(ds.sensor_points).astype(np.float32)
T_sens_norm_A = (T_sens_path_A - T_mean) / T_std

branch_A = np.stack([
    T_sens_norm_A, sens["region_id"], sens["is_heater"],
    sens["kappa"], sens["Cp"], sens["rho"],
], axis=0).astype(np.float32)
scalars_A = np.array([Tset_norm_val, t_val/CONFIG.t_total], dtype=np.float32)

# All-cells trunk
trunk_full = np.stack([
    sim["coords"][:,0], sim["coords"][:,1], sim["coords"][:,2],
    sim["region_id"]/11.0, sim["is_heater"],
    sim["kappa"]/100.0, sim["Cp"]/1000.0, sim["rho"]/10000.0,
], axis=1).astype(np.float32)

# Forward
branch_t = torch.from_numpy(branch_A).unsqueeze(0).to(device)
scalars_t = torch.from_numpy(scalars_A).unsqueeze(0).to(device)
trunk_t = torch.from_numpy(trunk_full).unsqueeze(0).to(device)

with torch.no_grad():
    pred_A = model(branch_t, scalars_t, trunk_t).squeeze().cpu().numpy()
T_pred_A_K = pred_A * T_std + T_mean

# Per-region MAE
heater_mask = sim["is_heater"] > 0.5
for region, (a,b) in sim["region_slices"].items():
    true_reg = T_tp1_gt[a:b]
    pred_reg = T_pred_A_K[a:b]
    mae = np.abs(pred_reg - true_reg).mean()
    print(f"  {region:>20}:  pred=[{pred_reg.min():.1f},{pred_reg.max():.1f}]  "
          f"true=[{true_reg.min():.1f},{true_reg.max():.1f}]  "
          f"MAE={mae:.2f} K")

non_heat = ~heater_mask
overall_mae = np.abs(T_pred_A_K[non_heat] - T_tp1_gt[non_heat]).mean()
print(f"\n  SINGLE-STEP MAE (non-heater, GT input): {overall_mae:.2f} K")
print(f"  Training val MAE was 6.62 K")
print(f"  If single-step > 50 K → input format bug (normalization mismatch)")
print(f"  If single-step ~ 6-10 K → rollout drift bug")
