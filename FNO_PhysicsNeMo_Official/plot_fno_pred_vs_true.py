"""
FNO predicted vs ground-truth temperature time-series plots.
Picks representative cells from each region and plots T(t).
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '.')

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_fno3d

# ─── Settings ───
PLOT_DIR = 'outputs/FNO_v5_FIX_150ep_20260425_1040/evaluation/plots_thesis'
CKPT_PATH = 'outputs/FNO_v5_FIX_150ep_20260425_1040/checkpoints/best_model.pt'
TEST_SIMS = ['sim_28', 'sim_31']  # one easy + one easy-to-medium

os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'figure.dpi': 100, 'savefig.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
})

cfg = CONFIG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = HeatTreatmentFNO3D(cfg).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model_state = ckpt.get('model_state', ckpt.get('model', ckpt.get('model_state_dict', ckpt)))
model.load_state_dict(model_state)
model.eval()
print(f"[INFO] Loaded FNO checkpoint from {CKPT_PATH}")

# Load eval dataset
ds = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
print(f"Test sims: {ds.sim_indices}")

# ─── Pick representative cell indices for each region ───
def pick_cells(sim, region_slices):
    """Pick 2 cells per region: one near center, one near edge of region."""
    picked = {}
    coords = sim['coords']
    for region in ['steel_cylinder', 'inner_box', 'outer_box']:
        if region not in region_slices:
            continue
        a, b = region_slices[region]
        region_coords = coords[a:b]
        # Center: closest to centroid
        centroid = region_coords.mean(axis=0)
        dist_centroid = np.linalg.norm(region_coords - centroid, axis=1)
        idx_center = a + int(np.argmin(dist_centroid))
        # Edge: farthest from centroid (typically a corner/boundary)
        idx_edge = a + int(np.argmax(dist_centroid))
        picked[region] = {'center': idx_center, 'edge': idx_edge}
    return picked

# ─── Run rollout for selected sims ───
for sim_label in TEST_SIMS:
    sim_i = int(sim_label.split('_')[1])
    if sim_i not in ds.sim_indices:
        print(f"  SKIP: {sim_label} not in test set {ds.sim_indices}")
        continue

    sim = ds._simulations[sim_i]
    print(f"\nRolling out {sim_label} (T_set={sim['T_set']:.0f}K) ...")
    T_pred, T_true = rollout_fno3d(model, ds, sim_i, device=device, start_t=20)

    region_slices = sim['region_slices']
    cells = pick_cells(sim, region_slices)

    times = sim['times']
    n_rollout = T_pred.shape[0]
    t_axis = times[20:20 + n_rollout]

    # ─── Plot ───
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    regions = ['steel_cylinder', 'inner_box', 'outer_box']
    cell_types = ['center', 'edge']

    for col, region in enumerate(regions):
        if region not in cells:
            continue
        for row, cell_type in enumerate(cell_types):
            ax = axes[row, col]
            cell_idx = cells[region][cell_type]

            T_t = T_true[:, cell_idx]
            T_p = T_pred[:, cell_idx]

            # Plot
            ax.plot(t_axis, T_t, color='black', lw=2.0, label='Ground truth (OpenFOAM)')
            ax.plot(t_axis, T_p, color='#d62728', lw=1.5, ls='--', label='FNO prediction')

            # Phase shading
            ax.axvspan(t_axis[0], 2760, alpha=0.05, color='#1f77b4')
            ax.axvspan(2760, t_axis[-1], alpha=0.05, color='#d62728')
            ax.axvline(2760, color='gray', ls=':', lw=0.8, alpha=0.7)

            # Compute MAE for this cell
            mae = np.mean(np.abs(T_p - T_t))
            ax.set_title(f"{region} ({cell_type}) — MAE={mae:.1f} K", fontsize=10)

            if row == 1:
                ax.set_xlabel('Time [s]')
            if col == 0:
                ax.set_ylabel('Temperature [K]')

            # Legend on first panel only
            if col == 0 and row == 0:
                ax.legend(loc='upper left', fontsize=8)

    fig.suptitle(f'FNO predicted vs ground-truth temperature — {sim_label} '
                 f'(T_set={sim["T_set"]:.0f}K)',
                 fontsize=13, y=1.0)

    plt.tight_layout()
    out = f'{PLOT_DIR}/05_pred_vs_true_{sim_label}.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

print(f"\n═══ All plots saved to: {PLOT_DIR}/")
