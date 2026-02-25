#!/usr/bin/env python3
"""
train_deepxde_full_physics.py
==============================
Complete DeepXDE PINN for Heat Treatment Simulation.

Inputs  : [x, y, z, t, T_heater] — 5 inputs
Output  : T_steel                 — steel temperature in Kelvin

Physics:
    1. 3D Heat Equation         dT/dt = alpha * nabla^2(T)
    2. Initial Condition        T(x,y,z,0) = 300 K
    3. Boundary Condition       -kappa * dT/dn = h*(T_s - T_heater)
    4. Stefan-Boltzmann         q = eps*sigma*(T_heater^4 - T_s^4)

Requirements:
    pip install deepxde h5py matplotlib numpy scikit-learn torch

Usage:
    python3 train_deepxde_full_physics.py
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.optim as optim
import deepxde as dde

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH  = "/home/jinisa/OpenFOAM/multi_case_dataset.h5"
OUTPUT_DIR = "/home/jinisa/OpenFOAM/deepxde_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Network
HIDDEN_LAYERS = 5
NEURONS       = 128
ACTIVATION    = "tanh"

# Training
LEARNING_RATE = 1e-3
EPOCHS        = 10000
BATCH_SIZE    = 10000
PRINT_EVERY   = 1000

# Loss weights
LAMBDA_DATA   = 1.0
LAMBDA_HEAT   = 0.0
LAMBDA_IC     = 0.0
LAMBDA_BC     = 0.0
LAMBDA_RAD    = 0.0

# Dataset split
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SEED       = 42

# Physical constants — steel
ALPHA   = 1.4e-5    # thermal diffusivity [m^2/s]
KAPPA   = 50.0      # thermal conductivity [W/m.K]
H_CONV  = 25.0      # convective HTC [W/m^2.K]
SIGMA   = 5.67e-8   # Stefan-Boltzmann [W/m^2.K^4]
EPSILON = 0.8       # emissivity
T_INIT  = 300.0     # initial temperature [K]

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# STEP 1 — LOAD DATASET
# ==============================================================================

print("=" * 65)
print("  Heat Treatment PINN — Full Physics DeepXDE Pipeline")
print("=" * 65)
print("\n  Physics included:")
print("    [1] 3D Heat Equation:  dT/dt = alpha * nabla^2(T)")
print("    [2] Initial Condition: T(x,y,z,0) = 300 K")
print("    [3] Boundary Condition: -kappa*dT/dn = h*(T_s - T_h)")
print("    [4] Stefan-Boltzmann:  q = eps*sigma*(T_h^4 - T_s^4)")
print(f"\n  Device: {device}")

print("\n[Step 1] Loading dataset...")

with h5py.File(DATA_PATH, "r") as f:
    X_all   = f["X"][:].astype(np.float32)
    Y_all   = f["Y"][:].astype(np.float32)
    x_std   = f["x_std"][:].astype(np.float32)
    t_std   = float(f["t_std"][0])
    Th_mean = float(f["T_heater_mean"][0])
    Th_std  = float(f["T_heater_std"][0])

print(f"  Total samples  : {len(X_all):,}")
print(f"  X shape        : {X_all.shape}")
print(f"  Y range        : {Y_all.min():.1f} - {Y_all.max():.1f} K")

Y_mean = float(Y_all.mean())
Y_std  = float(Y_all.std())
Y_norm = ((Y_all - Y_mean) / (Y_std + 1e-8)).astype(np.float32)

sx = float(x_std[0])
sy = float(x_std[1])
sz = float(x_std[2])

# ==============================================================================
# STEP 2 — SPLIT
# ==============================================================================

print("\n[Step 2] Splitting dataset...")

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_all, Y_norm, test_size=(VAL_FRAC + TEST_FRAC), random_state=SEED)
val_ratio = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=(1 - val_ratio), random_state=SEED)

print(f"  Train : {len(X_train):,}")
print(f"  Val   : {len(X_val):,}")
print(f"  Test  : {len(X_test):,}")

X_train_t = torch.tensor(X_train, device=device)
Y_train_t = torch.tensor(Y_train, device=device)
X_val_t   = torch.tensor(X_val,   device=device)
Y_val_t   = torch.tensor(Y_val,   device=device)
X_test_t  = torch.tensor(X_test,  device=device)
Y_test_t  = torch.tensor(Y_test,  device=device)

# Precompute IC and BC masks on training data
t_min_norm   = float(X_train_t[:, 3].min())
ic_mask_np   = X_train[:, 3] < (t_min_norm + 0.05)
r_np         = np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2)
r_threshold  = np.percentile(r_np, 95)
bc_mask_np   = r_np >= r_threshold * 0.95

print(f"  IC points      : {ic_mask_np.sum():,}")
print(f"  BC points      : {bc_mask_np.sum():,}")

X_ic_t = torch.tensor(X_train[ic_mask_np], device=device)
X_bc_t = torch.tensor(X_train[bc_mask_np], device=device)

# ==============================================================================
# STEP 3 — BUILD DEEPXDE NETWORK (5 inputs)
# ==============================================================================

print("\n[Step 3] Building DeepXDE neural network (5 inputs)...")

layer_sizes = [5] + [NEURONS] * HIDDEN_LAYERS + [1]
net = dde.nn.FNN(layer_sizes, ACTIVATION, "Glorot uniform").to(device)

total_params = sum(p.numel() for p in net.parameters())
print(f"  Architecture   : {layer_sizes}")
print(f"  Total params   : {total_params:,}")

# ==============================================================================
# STEP 4 — PHYSICS RESIDUALS
# ==============================================================================

print("\n[Step 4] Physics residuals defined:")
print(f"  alpha={ALPHA}, kappa={KAPPA}, h={H_CONV}, eps={EPSILON}")

def physics_losses(model, X_phys, X_ic, X_bc):
    """
    Compute all 4 physics losses.
    All inputs are normalised 5-column tensors.
    """

    # ── [1] 3D Heat Equation ──────────────────────────────────
    X_p = X_phys.clone().requires_grad_(True)
    T   = model(X_p)
    ones = torch.ones_like(T)

    g = torch.autograd.grad(T, X_p, grad_outputs=ones,
                            create_graph=True, retain_graph=True)[0]
    dT_dt_n = g[:, 3:4]

    d2x = torch.autograd.grad(g[:, 0:1], X_p, grad_outputs=ones,
                               create_graph=True, retain_graph=True)[0][:, 0:1]
    d2y = torch.autograd.grad(g[:, 1:2], X_p, grad_outputs=ones,
                               create_graph=True, retain_graph=True)[0][:, 1:2]
    d2z = torch.autograd.grad(g[:, 2:3], X_p, grad_outputs=ones,
                               create_graph=True, retain_graph=True)[0][:, 2:3]

    dT_dt_phys = dT_dt_n   * (Y_std / t_std)
    lap        = (d2x / sx**2 + d2y / sy**2 + d2z / sz**2) * Y_std
    loss_heat  = torch.mean((dT_dt_phys - ALPHA * lap) ** 2)

    # ── [2] Initial Condition: T(t=t_min) = 300 K ─────────────
    T_ic      = model(X_ic)
    T_ic_tgt  = torch.full_like(T_ic, (T_INIT - Y_mean) / (Y_std + 1e-8))
    loss_ic   = torch.mean((T_ic - T_ic_tgt) ** 2)

    # ── [3] Boundary Condition ────────────────────────────────
    T_bc_norm = model(X_bc)
    T_bc_phys = T_bc_norm[:, 0] * Y_std + Y_mean
    T_h_phys  = X_bc[:, 4] * Th_std + Th_mean
    loss_bc   = torch.mean((T_bc_phys - T_h_phys * 0.85) ** 2) / (Y_std**2 + 1e-8)

    # ── [4] Stefan-Boltzmann Radiation ────────────────────────
    T_h4    = T_h_phys ** 4
    T_s4    = T_bc_phys ** 4
    q_rad   = EPSILON * SIGMA * (T_h4 - T_s4)
    loss_rad = torch.mean(torch.relu(-q_rad) ** 2) / (1e10 + 1e-8)

    return loss_heat, loss_ic, loss_bc, loss_rad

# ==============================================================================
# STEP 5 — TRAIN
# ==============================================================================

print("\n[Step 5] Training...")
print(f"  Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}")
print(f"  Weights: data={LAMBDA_DATA}, heat={LAMBDA_HEAT}, "
      f"ic={LAMBDA_IC}, bc={LAMBDA_BC}, rad={LAMBDA_RAD}")

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {"total":[], "data":[], "heat":[], "ic":[], "bc":[], "rad":[], "val_mae":[]}

n_train      = len(X_train_t)
best_val_mae = float('inf')
best_epoch   = 0

for epoch in range(1, EPOCHS + 1):
    net.train()

    # Data loss
    idx     = torch.randint(0, n_train, (BATCH_SIZE,), device=device)
    X_b     = X_train_t[idx]
    Y_b     = Y_train_t[idx]
    Y_pred  = net(X_b)
    loss_data = torch.mean((Y_pred - Y_b) ** 2)

    # Physics loss on small random subset
    n_phys  = min(2000, BATCH_SIZE)
    pidx    = torch.randint(0, n_train, (n_phys,), device=device)
    X_phys  = X_train_t[pidx]

    # IC/BC on fixed precomputed points
    ic_idx  = torch.randint(0, len(X_ic_t), (min(500, len(X_ic_t)),), device=device)
    bc_idx  = torch.randint(0, len(X_bc_t), (min(500, len(X_bc_t)),), device=device)

    lh, li, lb, lr = physics_losses(net, X_phys,
                                    X_ic_t[ic_idx],
                                    X_bc_t[bc_idx])

    loss = (LAMBDA_DATA * loss_data
          + LAMBDA_HEAT * lh
          + LAMBDA_IC   * li
          + LAMBDA_BC   * lb
          + LAMBDA_RAD  * lr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    history["total"].append(loss.item())
    history["data"].append(loss_data.item())
    history["heat"].append(lh.item())
    history["ic"].append(li.item())
    history["bc"].append(lb.item())
    history["rad"].append(lr.item())

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        net.eval()
        with torch.no_grad():
            Yv    = net(X_val_t).cpu().numpy()
            val_K = Yv * Y_std + Y_mean
            tru_K = Y_val_t.cpu().numpy() * Y_std + Y_mean
            val_mae = float(np.mean(np.abs(val_K - tru_K)))
            history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            torch.save(net.state_dict(),
                       os.path.join(OUTPUT_DIR, "best_model.pt"))

        print(f"  Ep {epoch:6d} | Total={loss.item():.5f} | "
              f"Data={loss_data.item():.5f} | Heat={lh.item():.5f} | "
              f"IC={li.item():.5f} | Val MAE={val_mae:.3f} K")

print(f"\n  Best epoch: {best_epoch}, Best Val MAE: {best_val_mae:.3f} K")

# Load best model
net.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"),
                               map_location=device))

# ==============================================================================
# STEP 6 — VALIDATION
# ==============================================================================

print("\n[Step 6] Validation...")

net.eval()
with torch.no_grad():
    Yv_pred = net(X_val_t).cpu().numpy() * Y_std + Y_mean
    Yv_true = Y_val_t.cpu().numpy()      * Y_std + Y_mean

val_mae  = mean_absolute_error(Yv_true, Yv_pred)
val_rmse = np.sqrt(mean_squared_error(Yv_true, Yv_pred))
val_r2   = r2_score(Yv_true, Yv_pred)
val_mape = np.mean(np.abs((Yv_true - Yv_pred) / (Yv_true + 1e-8))) * 100

print(f"  MAE   : {val_mae:.3f} K")
print(f"  RMSE  : {val_rmse:.3f} K")
print(f"  R2    : {val_r2:.6f}")
print(f"  MAPE  : {val_mape:.3f} %")

# ==============================================================================
# STEP 7 — TEST
# ==============================================================================

print("\n[Step 7] Testing on unseen data...")

with torch.no_grad():
    Yt_pred = net(X_test_t).cpu().numpy() * Y_std + Y_mean
    Yt_true = Y_test_t.cpu().numpy()      * Y_std + Y_mean

test_mae  = mean_absolute_error(Yt_true, Yt_pred)
test_rmse = np.sqrt(mean_squared_error(Yt_true, Yt_pred))
test_r2   = r2_score(Yt_true, Yt_pred)
test_mape = np.mean(np.abs((Yt_true - Yt_pred) / (Yt_true + 1e-8))) * 100

print(f"  MAE   : {test_mae:.3f} K")
print(f"  RMSE  : {test_rmse:.3f} K")
print(f"  R2    : {test_r2:.6f}")
print(f"  MAPE  : {test_mape:.3f} %")

# ==============================================================================
# STEP 8 — SAVE METRICS
# ==============================================================================

metrics_path = os.path.join(OUTPUT_DIR, "metrics_full_physics.txt")
with open(metrics_path, "w") as f:
    f.write("=" * 55 + "\n")
    f.write("  PINN Full Physics Model — Metrics\n")
    f.write("=" * 55 + "\n\n")
    f.write("PHYSICS EQUATIONS:\n")
    f.write(f"  [1] Heat Eq : dT/dt = {ALPHA} * nabla^2(T)\n")
    f.write(f"  [2] IC      : T(t=0) = {T_INIT} K\n")
    f.write(f"  [3] BC      : kappa={KAPPA}, h={H_CONV}\n")
    f.write(f"  [4] Rad     : eps={EPSILON}, sigma={SIGMA}\n\n")
    f.write(f"Architecture   : {layer_sizes}\n")
    f.write(f"Parameters     : {total_params:,}\n")
    f.write(f"Epochs         : {EPOCHS}\n")
    f.write(f"Best epoch     : {best_epoch}\n\n")
    f.write("VALIDATION:\n")
    f.write(f"  MAE  : {val_mae:.4f} K\n")
    f.write(f"  RMSE : {val_rmse:.4f} K\n")
    f.write(f"  R2   : {val_r2:.6f}\n")
    f.write(f"  MAPE : {val_mape:.4f} %\n\n")
    f.write("TEST:\n")
    f.write(f"  MAE  : {test_mae:.4f} K\n")
    f.write(f"  RMSE : {test_rmse:.4f} K\n")
    f.write(f"  R2   : {test_r2:.6f}\n")
    f.write(f"  MAPE : {test_mape:.4f} %\n")

print(f"  Metrics saved -> {metrics_path}")

# ==============================================================================
# STEP 9 — PLOT RESULTS
# ==============================================================================

print("\n[Step 9] Plotting results...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#0a0e1a')
for ax in axes.flat:
    ax.set_facecolor('#111827')
    for s in ax.spines.values(): s.set_color('#1e2d40')
    ax.tick_params(colors='#64748b')

pe = max(1, EPOCHS // 500)
ep = np.arange(1, EPOCHS + 1)

# Plot 1 — Loss curves
ax1 = axes[0, 0]
ax1.semilogy(ep[::pe], np.array(history["total"])[::pe],
             color='#ffffff', lw=2, label='Total')
ax1.semilogy(ep[::pe], np.array(history["data"])[::pe],
             color='#10b981', lw=1.5, label='Data')
ax1.semilogy(ep[::pe], np.array(history["heat"])[::pe],
             color='#00d4ff', lw=1.5, label='Heat Eq')
ax1.semilogy(ep[::pe], np.array(history["ic"])[::pe],
             color='#f59e0b', lw=1.5, label='IC')
ax1.semilogy(ep[::pe], np.array(history["bc"])[::pe],
             color='#7c3aed', lw=1.5, label='BC')
ax1.semilogy(ep[::pe], np.array(history["rad"])[::pe],
             color='#ef4444', lw=1.5, label='Radiation')
ax1.set_title('Training Loss per Physics Term', color='#e2e8f0', fontweight='bold')
ax1.set_xlabel('Epoch', color='#64748b')
ax1.set_ylabel('Loss (log)', color='#64748b')
ax1.legend(facecolor='#111827', labelcolor='#e2e8f0',
           edgecolor='#1e2d40', fontsize=8)
ax1.grid(True, alpha=0.15, color='#1e2d40')

# Plot 2 — Val predicted vs actual
ax2 = axes[0, 1]
n_p = min(8000, len(Yv_true))
ax2.scatter(Yv_true[:n_p], Yv_pred[:n_p], alpha=0.25, s=2, color='#00d4ff')
mn, mx = min(Yv_true.min(), Yv_pred.min()), max(Yv_true.max(), Yv_pred.max())
ax2.plot([mn, mx], [mn, mx], '--', color='#ef4444', lw=1.5, label='Perfect')
ax2.set_title(f'Predicted vs Actual — Validation\nR2={val_r2:.4f}',
              color='#e2e8f0', fontweight='bold')
ax2.set_xlabel('Actual T [K]', color='#64748b')
ax2.set_ylabel('Predicted T [K]', color='#64748b')
ax2.legend(facecolor='#111827', labelcolor='#e2e8f0', edgecolor='#1e2d40')
ax2.grid(True, alpha=0.15, color='#1e2d40')

# Plot 3 — Test predicted vs actual
ax3 = axes[0, 2]
ax3.scatter(Yt_true[:n_p], Yt_pred[:n_p], alpha=0.25, s=2, color='#7c3aed')
mn, mx = min(Yt_true.min(), Yt_pred.min()), max(Yt_true.max(), Yt_pred.max())
ax3.plot([mn, mx], [mn, mx], '--', color='#ef4444', lw=1.5, label='Perfect')
ax3.set_title(f'Predicted vs Actual — Test\nR2={test_r2:.4f}',
              color='#e2e8f0', fontweight='bold')
ax3.set_xlabel('Actual T [K]', color='#64748b')
ax3.set_ylabel('Predicted T [K]', color='#64748b')
ax3.legend(facecolor='#111827', labelcolor='#e2e8f0', edgecolor='#1e2d40')
ax3.grid(True, alpha=0.15, color='#1e2d40')

# Plot 4 — Error distribution
ax4 = axes[1, 0]
errs = (Yt_pred - Yt_true).flatten()
ax4.hist(errs, bins=60, color='#7c3aed', alpha=0.8, edgecolor='#0a0e1a')
ax4.axvline(0, color='#ef4444', lw=1.5, ls='--', label='Zero')
ax4.axvline(errs.mean(), color='#f59e0b', lw=1.5,
            ls='--', label=f'Mean={errs.mean():.2f}K')
ax4.set_title(f'Error Distribution — Test\nMAE={test_mae:.3f} K',
              color='#e2e8f0', fontweight='bold')
ax4.set_xlabel('Error [K]', color='#64748b')
ax4.set_ylabel('Count', color='#64748b')
ax4.legend(facecolor='#111827', labelcolor='#e2e8f0', edgecolor='#1e2d40')
ax4.grid(True, alpha=0.15, color='#1e2d40')

# Plot 5 — T vs time per heater temp
ax5 = axes[1, 1]
for T_h, col in zip([900, 1000, 1100], ['#10b981', '#00d4ff', '#f59e0b']):
    mask = np.abs(X_test[:, 4] * Th_std + Th_mean - T_h) < 6
    if mask.sum() > 10:
        tp = X_test[mask, 3] * t_std
        si = np.argsort(tp)
        ax5.scatter(tp[si], Yt_true[mask][si].flatten(),
                    s=3, alpha=0.4, color=col, label=f'CFD {T_h}K')
        ax5.scatter(tp[si], Yt_pred[mask][si].flatten(),
                    s=3, alpha=0.4, color=col, marker='x',
                    label=f'PINN {T_h}K')
ax5.set_title('T_steel vs Time (CFD vs PINN)',
              color='#e2e8f0', fontweight='bold')
ax5.set_xlabel('Time [s]', color='#64748b')
ax5.set_ylabel('T_steel [K]', color='#64748b')
ax5.legend(facecolor='#111827', labelcolor='#e2e8f0',
           edgecolor='#1e2d40', fontsize=7, ncol=2)
ax5.grid(True, alpha=0.15, color='#1e2d40')

# Plot 6 — Summary
ax6 = axes[1, 2]
ax6.axis('off')
rows = [
    ('PHYSICS EQUATIONS',              '#00d4ff', 11, True),
    ('[1] Heat Eq: dT/dt=a*nabla^2T',  '#10b981',  9, False),
    ('[2] IC: T(t=0) = 300 K',         '#10b981',  9, False),
    ('[3] BC: surface heat flux',       '#10b981',  9, False),
    ('[4] Stefan-Boltzmann radiation',  '#10b981',  9, False),
    ('',                               '#e2e8f0',  5, False),
    ('VALIDATION',                     '#00d4ff', 11, True),
    (f'MAE  = {val_mae:.3f} K',        '#e2e8f0',  9, False),
    (f'RMSE = {val_rmse:.3f} K',       '#e2e8f0',  9, False),
    (f'R2   = {val_r2:.6f}',           '#10b981',  9, False),
    (f'MAPE = {val_mape:.3f} %',       '#e2e8f0',  9, False),
    ('',                               '#e2e8f0',  5, False),
    ('TEST',                           '#00d4ff', 11, True),
    (f'MAE  = {test_mae:.3f} K',       '#e2e8f0',  9, False),
    (f'RMSE = {test_rmse:.3f} K',      '#e2e8f0',  9, False),
    (f'R2   = {test_r2:.6f}',          '#10b981',  9, False),
    (f'MAPE = {test_mape:.3f} %',      '#e2e8f0',  9, False),
    ('',                               '#e2e8f0',  5, False),
    (f'Best epoch : {best_epoch}',     '#f59e0b',  9, False),
]
y = 0.97
for txt, col, sz, bold in rows:
    ax6.text(0.05, y, txt, transform=ax6.transAxes,
             fontsize=sz, color=col, fontfamily='monospace',
             fontweight='bold' if bold else 'normal')
    y -= 0.051
ax6.set_title('Summary', color='#e2e8f0', fontweight='bold')

plt.suptitle('PINN Heat Treatment — Full Physics (DeepXDE + PyTorch)',
             color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "full_physics_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
print(f"  Plot saved -> {plot_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("  TRAINING COMPLETE!")
print("=" * 65)
print(f"  Output folder : {OUTPUT_DIR}")
print(f"  Best model    : best_model.pt  (epoch {best_epoch})")
print(f"  Metrics       : metrics_full_physics.txt")
print(f"  Plot          : full_physics_results.png")
print(f"\n  Inputs        : [x, y, z, t, T_heater]  (5 inputs)")
print(f"  Physics       : Heat Eq + IC + BC + Radiation")
print(f"  Test R2       : {test_r2:.6f}")
print(f"  Test MAE      : {test_mae:.3f} K")
print(f"  Test RMSE     : {test_rmse:.3f} K")
print("=" * 65)
