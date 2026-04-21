"""
FNO vs GNN Rollout Comparison — Same test simulation.
Generates thesis-quality side-by-side comparison plots.
"""
import sys, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator

sys.path.insert(0, ".")
sys.path.insert(0, "/mimer/NOBACKUP/groups/revar/GNN_Unified")

OUT = "outputs/plots"
Path(OUT).mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.5,
})

def get_fno_rollout():
    from configs.fno_config import CONFIG as fno_cfg
    from data.dataset import FNO3DDataset
    from models.fno_model import HeatTreatmentFNO3D
    from models.rollout import rollout_fno3d

    device = 'cpu'
    test_ds = FNO3DDataset(fno_cfg.dataset_path, fno_cfg, 'test', 'evaluation')
    model = HeatTreatmentFNO3D.load(
        f'{fno_cfg.checkpoint_dir}/best_model.pt', fno_cfg, device)

    sim_i = test_ds.sim_indices[0]
    sim = test_ds._simulations[sim_i]
    start_t = 20

    T_pred_grids, T_true_grids = rollout_fno3d(model, test_ds, sim_i, device, start_t)
    n_steps = T_pred_grids.shape[0]
    times = sim['times'][start_t:start_t + n_steps]
    grid_points = test_ds.grid_points

    results = {}
    for rname in ['steel_cylinder', 'inner_box', 'outer_box']:
        s, e = sim['region_slices'][rname]
        coords = sim['coords'][s:e]
        pred = np.zeros((n_steps, coords.shape[0]), dtype=np.float32)
        true = np.zeros((n_steps, coords.shape[0]), dtype=np.float32)
        for step in range(n_steps):
            interp = NearestNDInterpolator(grid_points, T_pred_grids[step].ravel())
            pred[step] = interp(coords)
            t_idx = start_t + step
            if t_idx < sim['n_times']:
                true[step] = sim['T_all'][t_idx, s:e]
        results[rname] = (pred, true)

    return times, results, sim_i, sim['T_set']


def get_gnn_rollout():
    gnn_path = "/mimer/NOBACKUP/groups/revar/GNN_Unified"
    sys.path.insert(0, gnn_path)
    from configs.base_config import CONFIG as gnn_cfg
    from data.dataset_unified import UnifiedDataset, REGION_IDS, HEATER_REGIONS
    from models.meshgraphnet import HeatTreatmentGNN
    from evaluation.evaluate_unified import rollout_gnn

    gnn_cfg.node_in_features = 16
    device = 'cpu'

    test_ds = UnifiedDataset(gnn_cfg.all_regions_dataset_path, gnn_cfg, 'test', 'evaluation')
    ckpt_path = f'{gnn_path}/outputs/checkpoints_unified/best_model.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeatTreatmentGNN(gnn_cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    sim_i = test_ds.sim_indices[0]
    sim = test_ds._simulations[sim_i]
    start_t = 20

    T_pred, T_true = rollout_gnn(model, test_ds, sim_i, device, start_t)
    times = sim['times'][start_t:start_t + T_pred.shape[0]]

    results = {}
    for rname in ['steel_cylinder', 'inner_box', 'outer_box']:
        o = sim['region_data'][rname]['offset']
        n = sim['region_data'][rname]['n_cells']
        results[rname] = (T_pred[:, o:o+n], T_true[:, o:o+n])

    return times, results


def plot_comparison(fno_times, fno_results, gnn_times, gnn_results, sim_i, T_set):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Rollout Comparison — Test Sim {sim_i} (T_set={T_set:.0f}K)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Steel temperature
    ax = axes[0, 0]
    fp, ft = fno_results['steel_cylinder']
    gp, gt = gnn_results['steel_cylinder']
    ax.plot(fno_times[:len(ft)], ft.mean(axis=1), '#3266ad', linewidth=2.5, label='OpenFOAM')
    ax.plot(gnn_times[:len(gp)], gp.mean(axis=1), '#1D9E75', linewidth=2, linestyle='--', label='GNN')
    ax.plot(fno_times[:len(fp)], fp.mean(axis=1), '#E24B4A', linewidth=2, linestyle='--', label='FNO')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('(a) Steel cylinder')
    ax.legend(framealpha=0.9)

    # Steel error
    ax = axes[0, 1]
    gnn_err = gp.mean(axis=1) - gt.mean(axis=1)
    fno_err = fp.mean(axis=1) - ft.mean(axis=1)
    ax.plot(gnn_times[:len(gnn_err)], gnn_err, '#1D9E75', linewidth=2, label='GNN error')
    ax.plot(fno_times[:len(fno_err)], fno_err, '#E24B4A', linewidth=2, label='FNO error')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(y=-10, color='gray', linestyle='--', alpha=0.4)
    ax.fill_between(gnn_times[:len(gnn_err)], -10, 10, alpha=0.04, color='green')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [K]')
    ax.set_title('(b) Steel cylinder — error')
    ax.legend(framealpha=0.9)

    # Air temperature
    ax = axes[1, 0]
    fp_a, ft_a = fno_results['inner_box']
    gp_a, gt_a = gnn_results['inner_box']
    ax.plot(fno_times[:len(ft_a)], ft_a.mean(axis=1), '#3266ad', linewidth=2.5, label='OpenFOAM')
    ax.plot(gnn_times[:len(gp_a)], gp_a.mean(axis=1), '#1D9E75', linewidth=2, linestyle='--', label='GNN')
    ax.plot(fno_times[:len(fp_a)], fp_a.mean(axis=1), '#E24B4A', linewidth=2, linestyle='--', label='FNO')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('(c) Inner box (air)')
    ax.legend(framealpha=0.9)

    # Air error
    ax = axes[1, 1]
    gnn_err_a = gp_a.mean(axis=1) - gt_a.mean(axis=1)
    fno_err_a = fp_a.mean(axis=1) - ft_a.mean(axis=1)
    ax.plot(gnn_times[:len(gnn_err_a)], gnn_err_a, '#1D9E75', linewidth=2, label='GNN error')
    ax.plot(fno_times[:len(fno_err_a)], fno_err_a, '#E24B4A', linewidth=2, label='FNO error')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [K]')
    ax.set_title('(d) Inner box (air) — error')
    ax.legend(framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f'{OUT}/fno_vs_gnn_rollout_comparison.png'
    plt.savefig(path)
    print(f'  Saved: {path}')
    plt.close()


if __name__ == '__main__':
    print('  Loading GNN rollout...')
    gnn_times, gnn_results = get_gnn_rollout()
    print('  Loading FNO rollout...')
    fno_times, fno_results, sim_i, T_set = get_fno_rollout()
    print('  Generating comparison plot...')
    plot_comparison(fno_times, fno_results, gnn_times, gnn_results, sim_i, T_set)
    print('  Done!')
