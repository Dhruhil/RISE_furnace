"""
FNO Thesis Plots:
  1. Training curves (loss, physics, pushforward)
  2. Rollout temperature comparison (predicted vs ground truth)
  3. Per-region MAE bar chart
  4. Inference speed comparison
"""
import sys, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

OUT = "outputs/plots"
Path(OUT).mkdir(parents=True, exist_ok=True)


def parse_training_log(log_path):
    """Parse FNO training log into arrays."""
    epochs, tr_loss, tr_data, tr_phys = [], [], [], []
    va_data, va_phys, mae, r2, lam, w2 = [], [], [], [], [], []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            # Match epoch lines: "     1 | 0.00021 | ..."
            m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_data.append(float(m.group(3)))
                tr_phys.append(float(m.group(4)))
                va_data.append(float(m.group(5)))
                va_phys.append(float(m.group(6)))
                mae.append(float(m.group(7)))
                r2.append(float(m.group(8)))
                lam.append(float(m.group(10)))
                w2.append(float(m.group(11)))

    return {
        "epoch": np.array(epochs), "tr_loss": np.array(tr_loss),
        "tr_data": np.array(tr_data), "tr_phys": np.array(tr_phys),
        "va_data": np.array(va_data), "va_phys": np.array(va_phys),
        "mae": np.array(mae), "r2": np.array(r2),
        "lam": np.array(lam), "w2": np.array(w2),
    }


def plot_training_curves(d):
    """Plot 1: Training loss curves with curriculum annotations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("3D FNO Training Curves — Heat Treatment Digital Twin", fontsize=14, fontweight='bold')

    # 1a: Data loss (train vs val)
    ax = axes[0, 0]
    ax.semilogy(d["epoch"], d["tr_data"], 'b-', alpha=0.7, label='Train data loss')
    ax.semilogy(d["epoch"], d["va_data"], 'r-', alpha=0.7, label='Val data loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Data loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1b: Physics loss
    ax = axes[0, 1]
    mask = d["tr_phys"] > 0
    if mask.any():
        ax.plot(d["epoch"][mask], d["tr_phys"][mask], 'b-', alpha=0.7, label='Train physics')
        ax.plot(d["epoch"][mask], d["va_phys"][mask], 'r-', alpha=0.7, label='Val physics')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Physics Loss")
    ax.set_title("Physics loss (conduction + convection + equilibrium)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Physics starts')

    # 1c: Total loss with curriculum
    ax = axes[1, 0]
    ax.semilogy(d["epoch"], d["tr_loss"], 'b-', alpha=0.7, label='Train total')
    ax.axvline(x=15, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=20, color='green', linestyle='--', alpha=0.5)
    ax.text(15, ax.get_ylim()[1]*0.5, 'Pushforward\nstarts', fontsize=8, ha='center', color='orange')
    ax.text(20, ax.get_ylim()[1]*0.3, 'Physics\nstarts', fontsize=8, ha='center', color='green')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss (log)")
    ax.set_title("Total loss = (1-λ)·data + λ·physics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1d: Curriculum schedule
    ax = axes[1, 1]
    ax.plot(d["epoch"], d["lam"], 'g-', linewidth=2, label='λ (physics weight)')
    ax.plot(d["epoch"], d["w2"], 'orange', linewidth=2, label='w₂ (pushforward weight)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title("Training curriculum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.55)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fno_training_curves.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUT}/fno_training_curves.png")
    plt.close()


def plot_rollout_comparison():
    """Plot 2: Rollout — predicted vs ground truth temperature for steel."""
    # Load rollout data from the evaluation
    from configs.fno_config import CONFIG
    from data.dataset import FNO3DDataset, REGION_IDS
    import torch
    from models.fno_model import HeatTreatmentFNO3D
    from evaluation.evaluate_fno3d import rollout_fno3d

    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
    model = HeatTreatmentFNO3D.load(f"{cfg.checkpoint_dir}/best_model.pt", cfg, device)

    # Use first test sim
    sim_i = test_ds.sim_indices[0]
    sim = test_ds._simulations[sim_i]
    start_t = 20

    print(f"  Running rollout for sim {sim_i} (T_set={sim['T_set']:.0f}K)...")
    T_pred, T_true = rollout_fno3d(model, test_ds, sim_i, device, start_t)

    times = sim["times"][start_t:start_t + T_pred.shape[0]]
    region_onehot = sim["region_onehot"]
    region_ids = np.argmax(region_onehot, axis=1)

    # Steel cylinder average temperature over time
    steel_mask = region_ids == REGION_IDS["steel_cylinder"]
    inner_mask = region_ids == REGION_IDS["inner_box"]

    steel_pred_mean = T_pred[:, steel_mask].mean(axis=1)
    steel_true_mean = T_true[:, steel_mask].mean(axis=1)
    steel_pred_min = T_pred[:, steel_mask].min(axis=1)
    steel_pred_max = T_pred[:, steel_mask].max(axis=1)
    steel_true_min = T_true[:, steel_mask].min(axis=1)
    steel_true_max = T_true[:, steel_mask].max(axis=1)

    air_pred_mean = T_pred[:, inner_mask].mean(axis=1)
    air_true_mean = T_true[:, inner_mask].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"3D FNO Rollout — Sim {sim_i} (T_set={sim['T_set']:.0f}K)", fontsize=14, fontweight='bold')

    # 2a: Steel mean temperature
    ax = axes[0, 0]
    ax.plot(times, steel_true_mean, 'b-', linewidth=2, label='OpenFOAM (ground truth)')
    ax.plot(times, steel_pred_mean, 'r--', linewidth=2, label='3D FNO prediction')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.text(3200, ax.get_ylim()[0] if len(ax.get_ylim()) > 0 else 300, ' Phase 2→', fontsize=9, color='gray')
    ax.fill_between(times, steel_true_min, steel_true_max, alpha=0.1, color='blue', label='GT range')
    ax.fill_between(times, steel_pred_min, steel_pred_max, alpha=0.1, color='red', label='Pred range')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Steel cylinder — mean temperature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2b: Steel prediction error
    ax = axes[0, 1]
    error = steel_pred_mean - steel_true_mean
    ax.plot(times, error, 'r-', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='±5K')
    ax.axhline(y=-5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [K]")
    ax.set_title("Steel cylinder — prediction error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2c: Air (inner_box) mean temperature
    ax = axes[1, 0]
    ax.plot(times, air_true_mean, 'b-', linewidth=2, label='OpenFOAM')
    ax.plot(times, air_pred_mean, 'r--', linewidth=2, label='3D FNO')
    ax.axvline(x=3200, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Inner box (air) — mean temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2d: Per-region MAE bar chart
    ax = axes[1, 1]
    regions = ["steel_cylinder", "inner_box"]
    p1_maes = []
    p2_maes = []
    n_train = cfg.n_train_steps - start_t
    for rname, mask in [("steel_cylinder", steel_mask), ("inner_box", inner_mask)]:
        p1 = np.mean(np.abs(T_pred[1:n_train, mask] - T_true[1:n_train, mask]))
        p2 = np.mean(np.abs(T_pred[n_train:, mask] - T_true[n_train:, mask]))
        p1_maes.append(p1)
        p2_maes.append(p2)

    x_pos = np.arange(len(regions))
    width = 0.35
    ax.bar(x_pos - width/2, p1_maes, width, label='Phase 1 (0-3200s)', color='steelblue')
    ax.bar(x_pos + width/2, p2_maes, width, label='Phase 2 (3200-4000s)', color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Steel\ncylinder", "Inner box\n(air)"])
    ax.set_ylabel("MAE [K]")
    ax.set_title("Rollout MAE by region and phase")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{OUT}/fno_rollout_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUT}/fno_rollout_comparison.png")
    plt.close()


def plot_speed_comparison():
    """Plot 3: Speed comparison bar chart for thesis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["OpenFOAM\n(CFD)", "3D FNO\n(AI)"]
    times_sec = [3 * 3600, 0.78]  # 3 hours vs 0.78 seconds
    colors = ['steelblue', 'coral']

    ax.bar(methods, times_sec, color=colors, width=0.5)
    ax.set_ylabel("Time per simulation [seconds]")
    ax.set_title("Simulation Speed: OpenFOAM vs 3D FNO")
    ax.set_yscale('log')

    # Add value labels
    for i, (m, t) in enumerate(zip(methods, times_sec)):
        if t > 3600:
            label = f"{t/3600:.0f} hours"
        else:
            label = f"{t:.2f}s"
        ax.text(i, t * 1.5, label, ha='center', fontsize=12, fontweight='bold')

    ax.text(0.5, 0.02, f"Speedup: ~{times_sec[0]/times_sec[1]:.0f}x",
            transform=ax.transAxes, ha='center', fontsize=14,
            fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{OUT}/fno_speed_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUT}/fno_speed_comparison.png")
    plt.close()


if __name__ == "__main__":
    import glob

    # Find training log
    logs = sorted(glob.glob("outputs/logs/fno3d_*.log"))
    if logs:
        print(f"  Parsing training log: {logs[-1]}")
        d = parse_training_log(logs[-1])
        if len(d["epoch"]) > 0:
            print(f"  Found {len(d['epoch'])} epochs")
            plot_training_curves(d)
        else:
            print("  WARNING: No epoch data found in log")
    else:
        print("  WARNING: No training log found")

    # Speed comparison (always)
    plot_speed_comparison()

    # Rollout plots (needs GPU + model)
    try:
        plot_rollout_comparison()
    except Exception as e:
        print(f"  Rollout plot skipped: {e}")
        print(f"  (Run with GPU: apptainer exec --nv ... python plots_fno.py)")

    print(f"\n  All plots saved to {OUT}/")
