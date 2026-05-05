"""Quick inspection of trained GNN model and its input features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch_geometric.loader import DataLoader

from configs.base_config import CONFIG
from models.meshgraphnet import HeatTreatmentGNN
from data.dataset_unified import UnifiedDataset

CKPT = "outputs/checkpoints_unified/best_model.pt"

print("=" * 70)
print("  GNN MODEL INSPECTION")
print("=" * 70)

# --- 1. Load checkpoint ---
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"\n  Checkpoint: {CKPT}")
print(f"  Best epoch: {ckpt['epoch']}")
print(f"  Steel MAE:  {ckpt['metrics']['steel_mae']:.3f} K")
print(f"  R2:         {ckpt['metrics']['r2']:.6f}")

# --- 2. Load one batch (validation split = no noise) ---
cfg = CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Loading val dataset (this takes 1-2 min)...")
val_ds = UnifiedDataset(cfg.all_regions_dataset_path, cfg, "val", "training")
loader = DataLoader(val_ds, batch_size=1, shuffle=False)
batch = next(iter(loader))

print(f"\n  batch.x shape: {tuple(batch.x.shape)}")
print(f"  batch.edge_index shape: {tuple(batch.edge_index.shape)}")

# --- 3. Inspect all 16 feature columns ---
print(f"\n  === 16 NODE FEATURE COLUMNS ===")
names = ["x", "y", "z", "T_current", "T_set", "region_id", "time", "is_heater",
         "cx", "cy", "cz", "radius", "height", "kappa", "Cp", "rho"]
for i, name in enumerate(names):
    col = batch.x[:, i]
    n_unique = len(col.unique())
    marker = "  <-- CYLINDER POSITION" if name in ("cx", "cy", "cz") else ""
    print(f"  [{i:2d}] {name:10s}  "
          f"min={col.min().item():+.4f}  max={col.max().item():+.4f}  "
          f"mean={col.mean().item():+.4f}  n_unique={n_unique}{marker}")

# --- 4. Decode the normalized cx, cy, cz back to physical units ---
cx_norm = batch.x[0, 8].item()
cy_norm = batch.x[0, 9].item()
cz_norm = batch.x[0, 10].item()
print(f"\n  === CYLINDER POSITION (physical) ===")
print(f"  cx = {cx_norm * 0.206:.4f} m  (normalized: {cx_norm:.4f})")
print(f"  cy = {cy_norm * 0.36:.4f} m  (normalized: {cy_norm:.4f})")
print(f"  cz = {cz_norm * 0.39:.4f} m  (normalized: {cz_norm:.4f})")

# --- 5. Load trained weights and do one forward pass ---
print(f"\n  === FORWARD PASS WITH TRAINED MODEL ===")
model = HeatTreatmentGNN(cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
batch = batch.to(device)
with torch.no_grad():
    pred = model(batch)
T_pred = pred.squeeze(-1) * val_ds.T_std + val_ds.T_mean
T_true = batch.T_next
mae = (T_pred - T_true).abs().mean()
print(f"  Single-step MAE on this batch: {mae.item():.3f} K")
print(f"  Pred range: [{T_pred.min().item():.1f}, {T_pred.max().item():.1f}] K")
print(f"  True range: [{T_true.min().item():.1f}, {T_true.max().item():.1f}] K")

print(f"\n{'=' * 70}")
print(f"  DONE")
print(f"{'=' * 70}")
