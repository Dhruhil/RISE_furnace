"""
FNO Thesis Plots — clean, professional figures for the Master's Thesis.

Generates:
  1. Training curves       → fno_training_curves.png
  2. Validation R²         → fno_r2_curve.png
  3. Rollout T(t) vs GT    → fno_rollout_comparison.png  (steel + air only)
  4. Per-region MAE        → fno_per_region_mae.png      (steel + air only)
  5. Speed comparison      → fno_speed_comparison.png    (OpenFOAM vs FNO vs GNN)

Focus regions: steel_cylinder, inner_box (primary engineering interest).
Terminology: "In-horizon" (200-2760s, training time range) and
             "Extrapolation" (2760-3460s, beyond training horizon).
"""
import sys, json, re, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, ".")
from configs.fno_config import CONFIG

OUT = "outputs/plots"
Path(OUT).mkdir(parents=True, exist_ok=True)

# Thesis-quality rcParams
plt.rcParams.update({
    'font.size':        12,
    'axes.labelsize':   13,
    'axes.titlesize':   14,
    'legend.fontsize':  10,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'lines.linewidth':  1.8,
})

# Consistent colour palette
C_TRAIN  = '#3266ad'  # blue   — OpenFOAM / training / truth
C_VAL    = '#E24B4A'  # red    — validation / predictions
C_STEEL  = '#1D9E75'  # green  — steel cylinder
C_AIR    = '#D85A30'  # orange — inner_box air
C_ACCENT = '#7F77DD'  # purple — accents / curriculum

TRAIN_END = CONFIG.train_time_end      # 2760
PRED_END  = CONFIG.predict_time_end    # 3460


# ──────────────────────────────────────────────────────────────────────────
#  Training log parser
# ──────────────────────────────────────────────────────────────────────────
def parse_training_log(log_path):
    """Extract per-epoch metrics from FNO training log."""
    rows = {
        "epoch": [], "tr_data": [], "va_data": [],
        "mae": [], "steel_mae": [], "r2": [], "w2": [],
    }
    pat = re.compile(
        r'\s*(\d+)\s*\|'           # epoch
        r'\s*([\d.]+)\s*\|'        # TrWLoss
        r'\s*([\d.]+)\s*\|'        # TrData
        r'\s*[\d.]+\s*\|'          # TrPhys (skip)
        r'\s*[\d.]+\s*\|'          # VaWLoss (skip)
        r'\s*([\d.]+)\s*\|'        # VaData
        r'\s*([\d.]+)\s*\|'        # MAE
        r'\s*([\d.]+)\s*\|'        # Steel MAE
        r'\s*([\d.]+)\s*\|'        # R2
        r'\s*[\d.]+\s*\|'          # lam (skip)
        r'\s*([\d.]+)'             # w2
    )
    with open(log_path) as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            rows["epoch"].append(int(m.group(1)))
            rows["tr_data"].append(float(m.group(3)))
            rows["va_data"].append(float(m.group(4)))
            rows["mae"].append(float(m.group(5)))
            rows["steel_mae"].append(float(m.group(6)))
            rows["r2"].append(float(m.group(7)))
            rows["w2"].append(float(m.group(8)))
    return {k: np.array(v) for k, v in rows.items()}


def find_training_log():
    """Locate the FNO training log — tries known locations."""
    candidates = (
        glob.glob("outputs/FINAL_RUN_v4_*/logs/fno_v4_*.log")
        + glob.glob("outputs/logs/fno_v4_*.log")
        + glob.glob("outputs/logs/fno3d_*.log")
    )
    # Filter out error logs
    candidates = [c for c in candidates if "err" not in Path(c).name]
    if not candidates:
        return None
    # Most recent
    return max(candidates, key=lambda p: Path(p).stat().st_mtime)


# ──────────────────────────────────────────────────────────────────────────
#  Plot 1: Training curves (2 panels)
# ──────────────────────────────────────────────────────────────────────────
def plot_training_curves(d):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    best_ep    = int(d["epoch"][np.argmin(d["mae"])])
    best_mae   = float(np.min(d["mae"]))
    best_steel = float(d["steel_mae"][np.argmin(d["mae"])])

    # ── (a) Train vs Val data loss (log scale) ─────────────────
    ax = axes[0]
    ax.semilogy(d["epoch"], d["tr_data"], color=C_TRAIN, alpha=0.9, label='Training')
    ax.semilogy(d["epoch"], d["va_data"], color=C_VAL,   alpha=0.9, label='Validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data loss (MSE, normalised)")
    ax.set_title("(a) Training vs validation loss")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── (b) Validation MAE — overall + steel ──────────────────
    ax = axes[1]
    ax.plot(d["epoch"], d["mae"],       color=C_VAL,   alpha=0.9, label='Overall MAE')
    ax.plot(d["epoch"], d["steel_mae"], color=C_STEEL, alpha=0.9, label='Steel cylinder MAE')
    ax.axvline(best_ep, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'Best: overall {best_mae:.2f} K, steel {best_steel:.2f} K\n(epoch {best_ep})',
                xy=(best_ep, best_mae),
                xytext=(best_ep + len(d["epoch"]) * 0.1, best_mae + 3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='#444')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE [K]")
    ax.set_title("(b) Validation MAE over training")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = f"{OUT}/fno_training_curves.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
#  Plot 2: R² curve
# ──────────────────────────────────────────────────────────────────────────
def plot_r2_curve(d):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d["epoch"], d["r2"], color=C_STEEL, linewidth=2)
    best_ep = int(d["epoch"][np.argmax(d["r2"])])
    best_r2 = float(np.max(d["r2"]))
    ax.annotate(f'Best: {best_r2:.4f} (epoch {best_ep})',
                xy=(best_ep, best_r2),
                xytext=(best_ep + len(d["epoch"]) * 0.1, best_r2 - 0.003),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=11, color=C_STEEL)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Validation R² over training")
    ax.set_ylim(min(d["r2"].min() - 0.002, 0.985), 1.001)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = f"{OUT}/fno_r2_curve.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
#  Plot 3: Rollout T(t) — FNO vs OpenFOAM (steel + air only)
# ──────────────────────────────────────────────────────────────────────────
def plot_rollout_comparison():
    import torch
    from data.dataset import FNO3DDataset
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

    print(f"  Running rollout for test sim {sim_i} (T_set={sim['T_set']:.0f} K)...")
    T_pred_grids, _ = rollout_fno3d(model, test_ds, sim_i, device, start_t)
    n_steps = T_pred_grids.shape[0]
    times = sim["times"][start_t:start_t + n_steps]
    grid_points = test_ds.grid_points

    def get_region_temps(region_name):
        s, e = sim["region_slices"][region_name]
        coords = sim["coords"][s:e]
        T_pred = np.zeros((n_steps, coords.shape[0]), dtype=np.float32)
        T_true = np.zeros_like(T_pred)
        for step in range(n_steps):
            t_idx = start_t + step
            if t_idx >= sim["n_times"]:
                break
            T_true[step] = sim["T_all"][t_idx, s:e]
            interp = NearestNDInterpolator(grid_points, T_pred_grids[step].ravel())
            T_pred[step] = interp(coords)
        return T_pred, T_true

    steel_pred, steel_true = get_region_temps("steel_cylinder")
    air_pred,   air_true   = get_region_temps("inner_box")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Autoregressive Rollout — Test Simulation {sim_i} "
                 f"(T$_{{set}}$ = {sim['T_set']:.0f} K)",
                 fontsize=15, fontweight='bold', y=0.995)

    def _plot_region(ax_T, ax_err, T_pred, T_true, label, color):
        pred_mean = T_pred.mean(axis=1)
        true_mean = T_true.mean(axis=1)

        ax_T.fill_between(times, T_true.min(axis=1), T_true.max(axis=1),
                          alpha=0.10, color=C_TRAIN)
        ax_T.fill_between(times, T_pred.min(axis=1), T_pred.max(axis=1),
                          alpha=0.10, color=C_VAL)
        ax_T.plot(times, true_mean, color=C_TRAIN, linewidth=2, label='OpenFOAM (truth)')
        ax_T.plot(times, pred_mean, color=C_VAL,   linewidth=2, linestyle='--', label='FNO prediction')
        ax_T.axvline(TRAIN_END, color='gray', linestyle=':', alpha=0.6,
                     label=f'End of training horizon ({TRAIN_END:.0f}s)')
        ax_T.set_xlabel("Time [s]")
        ax_T.set_ylabel("Temperature [K]")
        ax_T.set_title(f"(a) {label} — T(t)" if label == "Steel cylinder" else f"(c) {label} — T(t)")
        ax_T.legend(loc='lower right', framealpha=0.9, fontsize=9)

        err = pred_mean - true_mean
        ax_err.plot(times, err, color=color, linewidth=1.6)
        ax_err.axhline(0, color='black', linewidth=0.5)
        ax_err.axhline( 10, color='gray', linestyle='--', alpha=0.4, label='±10 K')
        ax_err.axhline(-10, color='gray', linestyle='--', alpha=0.4)
        ax_err.fill_between(times, -10, 10, alpha=0.05, color='green')
        ax_err.axvline(TRAIN_END, color='gray', linestyle=':', alpha=0.6)
        ax_err.set_xlabel("Time [s]")
        ax_err.set_ylabel("Error [K]")
        ax_err.set_title(f"(b) {label} — prediction error" if label == "Steel cylinder"
                         else f"(d) {label} — prediction error")
        ax_err.legend(loc='upper left', framealpha=0.9, fontsize=9)

    _plot_region(axes[0, 0], axes[0, 1], steel_pred, steel_true, "Steel cylinder", C_STEEL)
    _plot_region(axes[1, 0], axes[1, 1], air_pred,   air_true,   "Inner box (air)", C_AIR)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"{OUT}/fno_rollout_comparison.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
#  Plot 4: Per-region MAE bar chart (in-horizon vs extrapolation)
# ──────────────────────────────────────────────────────────────────────────
def plot_per_region_mae():
    json_path = "outputs/evaluation/fno_rollout_results.json"
    if not Path(json_path).exists():
        json_path = "outputs/evaluation/fno3d_rollout_results.json"  # fallback
    if not Path(json_path).exists():
        print(f"  Skipping per-region MAE — no evaluation JSON found")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Handle both old and new JSON shapes
    per_sim = data.get("per_sim", data)

    regions = ["steel_cylinder", "inner_box"]
    labels = ["Steel cylinder", "Inner box (air)"]

    mae_in, mae_ext = {r: [] for r in regions}, {r: [] for r in regions}
    for sim_key, sim_data in per_sim.items():
        if not isinstance(sim_data, dict):
            continue
        for region in regions:
            if region in sim_data:
                mae_in[region].append(sim_data[region]["mae_p1"])
                p2 = sim_data[region].get("mae_p2")
                if p2 is not None and not (isinstance(p2, float) and np.isnan(p2)):
                    mae_ext[region].append(p2)

    in_means  = [np.mean(mae_in[r])  if mae_in[r]  else 0 for r in regions]
    in_stds   = [np.std(mae_in[r])   if mae_in[r]  else 0 for r in regions]
    ext_means = [np.mean(mae_ext[r]) if mae_ext[r] else 0 for r in regions]
    ext_stds  = [np.std(mae_ext[r])  if mae_ext[r] else 0 for r in regions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(regions))
    width = 0.35

    ax.bar(x - width/2, in_means,  width, yerr=in_stds,
           label=f'In-horizon (200–{TRAIN_END:.0f} s)',
           color=C_TRAIN, alpha=0.85, capsize=4)
    ax.bar(x + width/2, ext_means, width, yerr=ext_stds,
           label=f'Extrapolation ({TRAIN_END:.0f}–{PRED_END:.0f} s)',
           color=C_VAL,   alpha=0.85, capsize=4)

    for i, (m, s) in enumerate(zip(in_means, in_stds)):
        ax.text(i - width/2, m + s + 2, f'{m:.1f} K', ha='center', fontsize=10, color=C_TRAIN)
    for i, (m, s) in enumerate(zip(ext_means, ext_stds)):
        ax.text(i + width/2, m + s + 2, f'{m:.1f} K', ha='center', fontsize=10, color=C_VAL)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rollout MAE [K]")
    ax.set_title("Autoregressive rollout accuracy by region")
    ax.legend(framealpha=0.9, loc='upper left')
    ymax = max(max(in_means), max(ext_means)) + max(max(in_stds), max(ext_stds))
    ax.set_ylim(0, ymax * 1.3 if ymax > 0 else 1)

    plt.tight_layout()
    path = f"{OUT}/fno_per_region_mae.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
#  Plot 5: Speed comparison
# ──────────────────────────────────────────────────────────────────────────
def plot_speed_comparison():
    methods    = ["OpenFOAM\n(CFD reference)",
                  "GNN\n(MeshGraphNet)",
                  "3D FNO\n(this work)"]
    times_sec  = [3 * 3600, 71.3, 2.66]        # full simulation time
    colors     = ['#73726c', '#3266ad', '#1D9E75']
    speedups   = [1, 3*3600/71.3, 3*3600/2.66]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, times_sec, color=colors, width=0.5, alpha=0.9)
    ax.set_ylabel("Wall-clock time per full simulation [s]")
    ax.set_title("Inference speed — OpenFOAM vs neural surrogates")
    ax.set_yscale('log')

    for bar, t, s in zip(bars, times_sec, speedups):
        if t >= 3600:
            label = f"{t/3600:.1f} h"
        elif t >= 60:
            label = f"{t:.0f} s"
        else:
            label = f"{t:.2f} s"
        ax.text(bar.get_x() + bar.get_width()/2, t * 1.6, label,
                ha='center', fontsize=12, fontweight='500')
        if s > 1:
            ax.text(bar.get_x() + bar.get_width()/2, t * 0.35,
                    f'{s:,.0f}× faster', ha='center', fontsize=10, color='white',
                    fontweight='600')

    plt.tight_layout()
    path = f"{OUT}/fno_speed_comparison.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  === FNO THESIS PLOTS ===\n")

    log_path = find_training_log()
    if log_path:
        print(f"  Training log: {log_path}")
        d = parse_training_log(log_path)
        if len(d["epoch"]) > 0:
            print(f"  Epochs parsed:     {len(d['epoch'])}")
            print(f"  Best overall MAE:  {np.min(d['mae']):.2f} K "
                  f"(epoch {d['epoch'][np.argmin(d['mae'])]})")
            print(f"  Best steel MAE:    {np.min(d['steel_mae']):.2f} K")
            print(f"  Best R²:           {np.max(d['r2']):.4f}\n")
            plot_training_curves(d)
            plot_r2_curve(d)
        else:
            print("  WARNING: no epoch data parsed from training log\n")
    else:
        print("  WARNING: no training log found — skipping training curves\n")

    # Speed comparison (no data needed)
    plot_speed_comparison()

    # Per-region MAE from evaluation JSON
    plot_per_region_mae()

    # Rollout comparison (needs GPU)
    try:
        plot_rollout_comparison()
    except Exception as e:
        print(f"  Rollout plot skipped: {e}")
        print(f"  (Make sure you run this on a GPU with best_model.pt available)")

    print(f"\n  All plots saved to: {OUT}/\n")
