"""
Deeper diagnostic: test single-step MAE across multiple times and all test sims.
"""
import sys, torch, numpy as np
sys.path.insert(0, ".")
from configs.deeponet_config import CONFIG
from data.dataset import get_deeponet_eval_dataset
from models.deeponet_model import HeatTreatmentDeepONet
from scipy.interpolate import NearestNDInterpolator

device = torch.device("cuda")
ds = get_deeponet_eval_dataset(CONFIG)
model = HeatTreatmentDeepONet(CONFIG).to(device)
ckpt = torch.load("outputs/checkpoints/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

T_mean, T_std = ds.T_mean, ds.T_std
print(f"\nDataset stats: T_mean={T_mean:.2f}, T_std={T_std:.2f}")
print(f"Test sims: {ds.sim_indices}")
print(f"Number of training cases: {len(ds._simulations)}")

# Build trunk for a given sim
def build_trunk(sim):
    t = np.stack([
        sim["coords"][:,0], sim["coords"][:,1], sim["coords"][:,2],
        sim["region_id"]/11.0, sim["is_heater"],
        sim["kappa"]/100.0, sim["Cp"]/1000.0, sim["rho"]/10000.0,
    ], axis=1).astype(np.float32)
    return torch.from_numpy(t).unsqueeze(0).to(device)

def build_branch(sim, sens, T_t):
    interp_T = NearestNDInterpolator(sim["coords"], T_t)
    T_sens = interp_T(ds.sensor_points).astype(np.float32)
    T_sens_norm = (T_sens - T_mean) / T_std
    branch = np.stack([
        T_sens_norm, sens["region_id"], sens["is_heater"],
        sens["kappa"], sens["Cp"], sens["rho"],
    ], axis=0).astype(np.float32)
    return torch.from_numpy(branch).unsqueeze(0).to(device)

print(f"\n{'='*80}")
print(f"{'sim':>5} | {'t_i':>4} | {'T_set':>6} | {'steel':>7} | {'air':>7} | "
      f"{'outer':>7} | {'brick':>7} | {'non-heat MAE':>13}")
print('-'*80)

for sim_i in ds.sim_indices:
    sim = ds._simulations[sim_i]
    sens = ds._static_sensors[sim_i]
    Tset_norm = (sim["T_set"] - ds.Tset_mean) / ds.Tset_std
    trunk = build_trunk(sim)
    is_heat = sim["is_heater"] > 0.5
    
    for t_i in [30, 100, 200, 276]:
        if t_i + 1 >= sim["n_times"]:
            continue
        T_t = sim["T_all"][t_i]
        T_tp1 = sim["T_all"][t_i+1]
        t_val = sim["times"][t_i]
        branch = build_branch(sim, sens, T_t)
        scalars = torch.tensor([Tset_norm, t_val/CONFIG.t_total], 
                                dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(branch, scalars, trunk).squeeze().cpu().numpy()
        T_pred_K = pred * T_std + T_mean
        
        mae_per = {}
        for region, (a,b) in sim["region_slices"].items():
            mae_per[region] = float(np.abs(T_pred_K[a:b] - T_tp1[a:b]).mean())
        
        overall = float(np.abs(T_pred_K[~is_heat] - T_tp1[~is_heat]).mean())
        print(f"{sim_i:>5} | {t_i:>4} | {sim['T_set']:>6.0f} | "
              f"{mae_per.get('steel_cylinder',0):>7.2f} | "
              f"{mae_per.get('inner_box',0):>7.2f} | "
              f"{mae_per.get('outer_box',0):>7.2f} | "
              f"{mae_per.get('brick_heater',0):>7.2f} | "
              f"{overall:>13.2f}")

# --- Now test on a TRAIN simulation to compare ---
print(f"\n{'='*80}")
print(f"Now same test on TRAIN simulations (from val split):")
print('-'*80)
import json
import h5py
# Reload a train-split dataset to access train sims
from data.dataset import DeepONetDataset
train_ds = DeepONetDataset(CONFIG.dataset_path, CONFIG, split="val", split_mode="training")
print(f"Val sim_indices: {train_ds.sim_indices}")

for sim_i in train_ds.sim_indices[:3]:
    sim = train_ds._simulations[sim_i]
    sens = train_ds._static_sensors[sim_i]
    Tset_norm = (sim["T_set"] - train_ds.Tset_mean) / train_ds.Tset_std
    trunk = build_trunk(sim)
    is_heat = sim["is_heater"] > 0.5
    
    t_i = 100
    T_t = sim["T_all"][t_i]
    T_tp1 = sim["T_all"][t_i+1]
    t_val = sim["times"][t_i]
    branch = build_branch(sim, sens, T_t)
    scalars = torch.tensor([Tset_norm, t_val/CONFIG.t_total],
                            dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(branch, scalars, trunk).squeeze().cpu().numpy()
    T_pred_K = pred * T_std + T_mean
    
    mae_per = {}
    for region, (a,b) in sim["region_slices"].items():
        mae_per[region] = float(np.abs(T_pred_K[a:b] - T_tp1[a:b]).mean())
    
    overall = float(np.abs(T_pred_K[~is_heat] - T_tp1[~is_heat]).mean())
    print(f"{sim_i:>5} | {t_i:>4} | {sim['T_set']:>6.0f} | "
          f"{mae_per.get('steel_cylinder',0):>7.2f} | "
          f"{mae_per.get('inner_box',0):>7.2f} | "
          f"{mae_per.get('outer_box',0):>7.2f} | "
          f"{mae_per.get('brick_heater',0):>7.2f} | "
          f"{overall:>13.2f}")
