"""
Plot FNO evaluation results from the existing JSON.
Generates 3 thesis-ready figures from already-computed metrics.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── Settings ───
JSON_PATH = 'outputs/FNO_v5_FIX_150ep_20260425_1040/evaluation/fno_rollout_results.json'
PLOT_DIR = 'outputs/FNO_v5_FIX_150ep_20260425_1040/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

with open(JSON_PATH) as f:
    data = json.load(f)

# ─── Plot 1: Per-step MAE trajectory for steel_cylinder (all sims) ───
print('Plot 1: Per-step Steel MAE trajectory ...')
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(data['per_sim'])))
for i, (sim_key, sim_data) in enumerate(data['per_sim'].items()):
    if 'steel_cylinder' in sim_data:
        steel = sim_data['steel_cylinder']
        if 'step_mae' in steel:
            mae_traj = np.array(steel['step_mae'])
            # Time axis assumes start_t=20 and dt=10s (FNO settings)
            t_axis = 200 + np.arange(len(mae_traj)) * 10
            plt.plot(t_axis, mae_traj, color=colors[i], lw=1.0,
                     alpha=0.7, label=sim_key)

plt.axvline(2760, color='k', ls='--', lw=1, alpha=0.6, label='Train cutoff (2760s)')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Steel cylinder MAE [K]', fontsize=12)
plt.title('FNO rollout: per-step Steel MAE per test simulation', fontsize=13)
plt.legend(fontsize=9, loc='upper left', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = f'{PLOT_DIR}/01_steel_mae_per_sim.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'  Saved: {out}')

# ─── Plot 2: Phase 2 MAE trajectory (averaged) ───
print('Plot 2: Phase 2 averaged MAE trajectory ...')
if 'phase2_per_step' in data:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, region in zip(axes, ['steel_cylinder', 'inner_box']):
        if region in data['phase2_per_step']:
            p2 = data['phase2_per_step'][region]
            t = np.array(p2['abs_t'])
            mae_mean = np.array(p2['mae_mean'])
            mae_std = np.array(p2.get('mae_std', [0.0]*len(mae_mean)))
            
            ax.plot(t, mae_mean, lw=2, color='C3', label='Mean MAE')
            ax.fill_between(t, mae_mean - mae_std, mae_mean + mae_std,
                            alpha=0.25, color='C3', label='±1 std')
            ax.set_xlabel('Time [s]', fontsize=12)
            ax.set_ylabel(f'{region} MAE [K]', fontsize=12)
            ax.set_title(f'Phase 2 — {region}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
    plt.tight_layout()
    out = f'{PLOT_DIR}/02_phase2_avg_mae.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')

# ─── Plot 3: Per-sim aggregate MAE bar chart ───
print('Plot 3: Per-sim aggregate MAE bar chart ...')
sims = list(data['per_sim'].keys())
mae_p1 = [data['per_sim'][s]['steel_cylinder']['mae_p1'] for s in sims]
mae_p2 = [data['per_sim'][s]['steel_cylinder']['mae_p2'] for s in sims]

x = np.arange(len(sims))
width = 0.35
plt.figure(figsize=(10, 5.5))
plt.bar(x - width/2, mae_p1, width, label='Phase 1 (in-horizon)', color='C0')
plt.bar(x + width/2, mae_p2, width, label='Phase 2 (extrapolation)', color='C3')
plt.xticks(x, sims, rotation=45)
plt.ylabel('Steel cylinder MAE [K]', fontsize=12)
plt.title('FNO Steel MAE per test simulation', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
out = f'{PLOT_DIR}/03_per_sim_steel_mae.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'  Saved: {out}')

print()
print('═══ All plots saved ═══')
print(f'Directory: {PLOT_DIR}/')
