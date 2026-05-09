"""Save per-step T(t) arrays for thesis Figure RQ3 — true vs prediction."""
import sys, json
from pathlib import Path
import numpy as np
import torch
import h5py

sys.path.insert(0, ".")

from configs.base_config import CONFIG
from data.dataset import HeatTreatmentDataset
from models.gnn_model import HeatTreatmentGNN
from models.rollout import rollout_unified  # adjust if name differs

OUT_DIR = Path("outputs/GNN_v5_FIX_150ep_20260425_0948/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "rollout_temps.h5"

CKPT = "outputs/GNN_v5_FIX_150ep_20260425_0948/checkpoints/best_model.pt"

cfg = CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
dataset = HeatTreatmentDataset(cfg.dataset_path, cfg, "test")
sim_indices = dataset.sim_indices

# Load model
model = HeatTreatmentGNN.load(CKPT, cfg, device)
model.eval()

with h5py.File(OUT_PATH, "w") as f:
    f.attrs["model"] = "GNN"
    f.attrs["checkpoint"] = CKPT
    for sim_i in sim_indices:
        sim = dataset._simulations[sim_i]
        print(f"Rolling out sim_{sim_i} (T_set={sim['T_set']:.0f}K)...")
        T_pred, T_true = rollout_unified(model, dataset, sim_i, device=device, start_t=20)
        # T_pred / T_true: (n_steps, n_cells) on the original mesh
        times = sim["times"][20:20 + T_pred.shape[0]]
        region_id = sim["region_id"]  # (n_cells,)
        is_steel = (region_id == 0)   # adjust if steel id differs
        is_air   = (region_id == 1)
        grp = f.create_group(f"sim_{sim_i}")
        grp.attrs["T_set"] = float(sim["T_set"])
        grp.create_dataset("times",     data=times.astype(np.float32))
        grp.create_dataset("T_pred",    data=T_pred.astype(np.float32))
        grp.create_dataset("T_true",    data=T_true.astype(np.float32))
        grp.create_dataset("is_steel",  data=is_steel)
        grp.create_dataset("is_air",    data=is_air)
print(f"\nSaved: {OUT_PATH}")
