"""
FNO Thesis Plots — Clean, Professional Figures.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Generates:
  1. Training curves (4-panel: data loss, MAE, curriculum, train-val gap)
  2. Rollout temperature comparison (predicted vs ground truth)
  3. Per-region MAE bar chart (from rollout evaluation)
  4. Inference speed comparison (OpenFOAM vs FNO vs GNN)
"""
import sys, json, re, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, ".")

OUT = "outputs/plots"
Path(OUT).mkdir(parents=True, exist_ok=True)

# Thesis-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})


def parse_training_log(log_path):
    """Parse FNO training log — captures all columns including Steel MAE."""
    epochs, tr_loss, tr_data, tr_phys = [], [], [], []
    va_wloss, va_data, mae, steel_mae, r2, lam, w2 = [], [], [], [], [], [], []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            # Match: "  1 | 0.00922 | 0.00866 | 1.13748 | 0.00182 | 0.00232 | 13.03 | 2.89 | 0.9907 | 0.0005 | 0.00 | 838s"
            m = re.match(
                r'\s*(\d+)\s*\|'           # epoch
                r'\s*([\d.]+)\s*\|'        # TrWLoss
                r'\s*([\d.]+)\s*\|'        # TrData
                r'\s*([\d.]+)\s*\|'        # TrPhys
                r'\s*([\d.]+)\s*\|'        # VaWLoss
                r'\s*([\d.]+)\s*\|'        # VaData
                r'\s*([\d.]+)\s*\|'        # MAE
                r'\s*([\d.]+)\s*\|'        # Steel MAE
                r'\s*([\d.]+)\s*\|'        # R2
                r'\s*([\d.]+)\s*\|'        # lam
                r'\s*([\d.]+)',            # w2
                line
            )
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_data.append(float(m.group(3)))
                tr_phys.append(float(m.group(4)))
                va_wloss.append(float(m.group(5)))
                va_data.append(float(m.group(6)))
                mae.append(float(m.group(7)))
                steel_mae.append(float(m.group(8)))
                r2.append(float(m.group(9)))
                lam.append(float(m.group(10)))
                w2.append(float(m.group(11)))

    return {
        "epoch": np.array(epochs),
        "tr_loss": np.array(tr_loss), "tr_data": np.array(tr_data),
        "tr_phys": np.array(tr_phys),
        "va_wloss": np.array(va_wloss), "va_data": np.array(va_data),
        "mae": np.array(mae), "steel_mae": np.array(steel_mae),
        "r2": np.array(r2), "lam": np.array(lam), "w2": np.array(w2),
    }


def plot_training_curves(d):
    """Plot 1: 4-panel training curves — thesis quality."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("3D FNO Training Curves", fontsize=16, fontweight='bold', y=0.98)

    best_epoch = int(d["epoch"][np.argmin(d["mae"])])
    best_mae = float(np.min(d["mae"]))
    best_steel = float(d["steel_mae"][np.argmin(d["mae"])])

    # ── 1a: Data loss (train vs val) ──
    ax = axes[0, 0]
    ax.plot(d["epoch"], d["tr_data"], color='#3266ad', alpha=0.8, label='Train data loss')
    ax.plot(d["epoch"], d["va_data"], color='#E24B4A', alpha=0.8, label='Val data loss')
    # Mark pushforward start
    pf_start = d["epoch"][d["w2"] > 0]
    if len(pf_start) > 0:
        ax.axvline(x=pf_start[0], color='#7F77DD', linestyle='--', alpha=0.5,
                   label=f'Pushforward starts (ep {pf_start[0]})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data loss (MSE)")
    ax.set_title("(a) Train vs validation loss")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 1b: MAE and Steel MAE ──
    ax = axes[0, 1]
    ax.plot(d["epoch"], d["mae"], color='#1D9E75', alpha=0.8, label='Overall MAE')
    ax.plot(d["epoch"], d["steel_mae"], color='#D85A30', alpha=0.6, label='Steel MAE')
    ax.axhline(y=best_mae, color='#1D9E75', linestyle=':', alpha=0.4)
    ax.annotate(f'Best: {best_mae:.1f}K (ep {best_epoch})',
                xy=(best_epoch, best_mae), fontsize=9,
                xytext=(best_epoch + len(d["epoch"])//10, best_mae + 1),
                arrowprops=dict(arrowstyle='->', color='#1D9E75', alpha=0.6),
                color='#1D9E75')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE [K]")
    ax.set_title("(b) Validation MAE")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(d["mae"].max(), d["steel_mae"].max()) * 1.1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 1c: Training curriculum ──
    ax = axes[1, 0]
    ax.plot(d["epoch"], d["w2"], color='#7F77DD', linewidth=2, label='Pushforward $w_2$')
    ax.plot(d["epoch"], d["lam"] * 1000, color='#BA7517', linewidth=2,
            linestyle='--', label=r'Physics $\lambda$ (×1000)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title("(c) Training curriculum schedule")
    ax.legend(loc='center right', framealpha=0.9)
    ax.set_ylim(-0.05, max(d["w2"].max(), d["lam"].max() * 1000) * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 1d: Train-val gap ──
    ax = axes[1, 1]
    gap = ((d["va_data"] - d["tr_data"]) / (d["tr_data"] + 1e-10)) * 100
    ax.plot(d["epoch"], gap, color='#3266ad', alpha=0.8)
    ax.fill_between(d["epoch"], 0, gap, color='#3266ad', alpha=0.08)
    ax.axhline(y=np.mean(gap[-10:]), color='#3266ad', linestyle=':', alpha=0.4,
               label=f'Last 10 avg: {np.mean(gap[-10:]):.0f}%')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap [%]")
    ax.set_title("(d) Train-validation gap")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(gap) * 1.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{OUT}/fno_training_curves.png"
    plt.savefig(path)
    print(f"  Saved: {path}")
    plt.close()


def plot_r2_curve(d):
    """Plot 2: R² over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d["epoch"], d["r2"], color='#1D9E75', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Validation R² over training")
    ax.set_ylim(min(d["r2"].min() - 0.002, 0.985), 1.001)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    best_r2_epoch = int(d["epoch"][np.argmax(d["r2"])])
    best_r2 = float(np.max(d["r2"]))
    ax.annotate(f'Best: {best_r2:.4f} (ep {best_r2_epoch})',
                xy=(best_r2_epoch, best_r2), fontsize=10,
                xytext=(best_r2_epoch + len(d["epoch"])//10, best_r2 - 0.003),
                arrowprops=dict(arrowstyle='->', color='#1D9E75'),
                color='#1D9E75')

    plt.tight_layout()
    path = f"{OUT}/fno_r2_curve.png"
    plt.savefig(path)
    print(f"  Saved: {path}")
    plt.close()


def plot_rollout_comparison():
    """Plot 3: Rollout — predicted vs ground truth temperature."""
    from configs.fno_config import CONFIG
    from data.dataset import FNO3DDataset, REGION_IDS, HEATER_REGIONS
    import torch
    from models.fno_model import HeatTreatmentFNO3D
    from models.rollout import rollout_fno3d
    from scipy.interpolate import NearestNDInterpolator

    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
    model = HeatTreatmentFNO3D.load(f"{cfg.checkpoint_dir}/best_model.pt", cfg, device)

    sim_i = test_ds.sim_indices[0]
    sim = test_ds._simulations[sim_i]
    start_t = 20
    n_times = sim["n_times"]

    print(f"  Running rollout for sim {sim_i} (T_set={sim['T_set']:.0f}K)...")
    T_pred_grids, T_true_grids = rollout_fno3d(model, test_ds, sim_i, device, start_t)
    n_steps = T_pred_grids.shape[0]
    times = sim["times"][start_t:start_t + n_steps]
    grid_points = test_ds.grid_points

    def get_region_temps(region_name):
        slc = sim["region_slices"][region_name]
        s, e = slc
        coords = sim["coords"][s:e]
        n_cells = coords.shape[0]
        T_pred = np.zeros((n_steps, n_cells), dtype=np.float32)
        T_true = np.zeros((n_steps, n_cells), dtype=np.float32)
        for step in range(n_steps):
            interp_p = NearestNDInterpolator(grid_points, T_pred_grids[step].ravel())
            T_pred[step] = interp_p(coords)
            t_idx = start_t + step
            if t_idx < n_times:
                T_true[step] = sim["T_all"][t_idx, s:e]
        return T_pred, T_true

    steel_pred, steel_true = get_region_temps("steel_cylinder")
    air_pred, air_true = get_region_temps("inner_box")
    outer_pred, outer_true = get_region_temps("outer_box")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"3D FNO Autoregressive Rollout — Test Sim {sim_i}", fontsize=14, fontweight='bold', y=0.98)

    ax = axes[0, 0]
    s_pred_mean = steel_pred.mean(axis=1)
    s_true_mean = steel_true.mean(axis=1)
    ax.fill_between(times, steel_true.min(axis=1), steel_true.max(axis=1), alpha=0.1, color='#3266ad')
    ax.fill_between(times, steel_pred.min(axis=1), steel_pred.max(axis=1), alpha=0.1, color='#E24B4A')
    ax.plot(times, s_true_mean, '#3266ad', linewidth=2, label='OpenFOAM')
    ax.plot(times, s_pred_mean, '#E24B4A', linewidth=2, linestyle='--', label='FNO prediction')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("(a) Steel cylinder")
    ax.legend(framealpha=0.9)

    ax = axes[0, 1]
    error = s_pred_mean - s_true_mean
    ax.plot(times, error, '#E24B4A', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.4, label='+-10K')
    ax.axhline(y=-10, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(times, -10, 10, alpha=0.04, color='green')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [K]")
    ax.set_title("(b) Steel cylinder — error")
    ax.legend(framealpha=0.9)

    ax = axes[1, 0]
    ax.plot(times, air_true.mean(axis=1), '#3266ad', linewidth=2, label='OpenFOAM')
    ax.plot(times, air_pred.mean(axis=1), '#E24B4A', linewidth=2, linestyle='--', label='FNO')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("(c) Inner box (air)")
    ax.legend(framealpha=0.9)

    ax = axes[1, 1]
    ax.plot(times, outer_true.mean(axis=1), '#3266ad', linewidth=2, label='OpenFOAM')
    ax.plot(times, outer_pred.mean(axis=1), '#E24B4A', linewidth=2, linestyle='--', label='FNO')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("(d) Outer box (walls)")
    ax.legend(framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{OUT}/fno_rollout_comparison.png"
    plt.savefig(path)
    print(f"  Saved: {path}")
    plt.close()

def plot_per_region_mae():
    """Plot 4: Per-region MAE bar chart from rollout evaluation JSON."""
    json_path = "outputs/evaluation/fno3d_rollout_results.json"
    if not Path(json_path).exists():
        print(f"  Skipping per-region plot: {json_path} not found")
        return

    with open(json_path) as f:
        results = json.load(f)

    # Aggregate across test sims
    region_p1 = {}
    region_p2 = {}
    for sim_key, sim_data in results.items():
        for region, metrics in sim_data.items():
            if region not in region_p1:
                region_p1[region] = []
                region_p2[region] = []
            region_p1[region].append(metrics["mae_p1"])
            if metrics["mae_p2"] and not (isinstance(metrics["mae_p2"], float) and
                                           np.isnan(metrics["mae_p2"])):
                region_p2[region].append(metrics["mae_p2"])

    # Only non-heater regions
    plot_regions = ["steel_cylinder", "inner_box", "outer_box"]
    labels = ["Steel\ncylinder", "Inner box\n(air)", "Outer box\n(walls)"]
    p1_means = [np.mean(region_p1.get(r, [0])) for r in plot_regions]
    p2_means = [np.mean(region_p2.get(r, [0])) for r in plot_regions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_regions))
    width = 0.35

    bars1 = ax.bar(x - width/2, p1_means, width, label='Phase 1 (0-3200s)',
                   color='#3266ad', alpha=0.85)
    bars2 = ax.bar(x + width/2, p2_means, width, label='Phase 2 (3200-4000s)',
                   color='#D85A30', alpha=0.85)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}K',
                ha='center', fontsize=10, color='#3266ad')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}K',
                ha='center', fontsize=10, color='#D85A30')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE [K]")
    ax.set_title("Autoregressive Rollout MAE by Region")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(max(p1_means), max(p2_means)) * 1.3)

    plt.tight_layout()
    path = f"{OUT}/fno_per_region_mae.png"
    plt.savefig(path)
    print(f"  Saved: {path}")
    plt.close()


def plot_speed_comparison():
    """Plot 5: Speed comparison — OpenFOAM vs FNO vs GNN."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["OpenFOAM\n(CFD)", "GNN\n(MeshGraphNet)", "3D FNO\n(Neural Operator)"]
    times_sec = [3 * 3600, 71.3, 1.73]
    colors = ['#73726c', '#3266ad', '#1D9E75']
    speedups = [1, 3*3600/71.3, 3*3600/1.73]

    bars = ax.bar(methods, times_sec, color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel("Time per full simulation [seconds]")
    ax.set_title("Simulation Speed Comparison")
    ax.set_yscale('log')

    for i, (bar, t, s) in enumerate(zip(bars, times_sec, speedups)):
        if t > 3600:
            label = f"{t/3600:.0f}h"
        elif t > 60:
            label = f"{t:.0f}s"
        else:
            label = f"{t:.1f}s"
        ax.text(bar.get_x() + bar.get_width()/2, t * 2.5, label,
                ha='center', fontsize=12, fontweight='500')
        if s > 1:
            ax.text(bar.get_x() + bar.get_width()/2, t * 0.3,
                    f'{s:.0f}× faster', ha='center', fontsize=9, color='white',
                    fontweight='500')

    plt.tight_layout()
    path = f"{OUT}/fno_speed_comparison.png"
    plt.savefig(path)
    print(f"  Saved: {path}")
    plt.close()


if __name__ == "__main__":
    print("\n  === FNO THESIS PLOTS ===\n")

    # Find latest training log
    logs = sorted([l for l in glob.glob("outputs/logs/fno3d_*.log") if "err" not in l])
    if logs:
        log_path = logs[-1]
        print(f"  Parsing: {log_path}")
        d = parse_training_log(log_path)
        if len(d["epoch"]) > 0:
            print(f"  Found {len(d['epoch'])} epochs")
            print(f"  Best MAE: {np.min(d['mae']):.2f}K (epoch {d['epoch'][np.argmin(d['mae'])]})")
            print(f"  Best Steel MAE: {np.min(d['steel_mae']):.2f}K")
            print(f"  Best R²: {np.max(d['r2']):.4f}")
            print()
            plot_training_curves(d)
            plot_r2_curve(d)
        else:
            print("  WARNING: No epoch data found in log")
    else:
        print("  WARNING: No training log found")

    # Speed comparison (always works)
    plot_speed_comparison()

    # Per-region MAE from evaluation JSON
    plot_per_region_mae()

    # Rollout comparison (needs GPU + trained model)
    try:
        plot_rollout_comparison()
    except Exception as e:
        print(f"  Rollout plot skipped: {e}")
        print(f"  (Run on GPU: sbatch run_plots.sh)")

    print(f"\n  All plots saved to: {OUT}/")
