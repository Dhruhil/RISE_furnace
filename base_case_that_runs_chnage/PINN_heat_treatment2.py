# ============================================================
# Physics-Informed Neural Network (PINN) — Version 2
# Heat Treatment of Steel Cylinder
# IMPROVED: Better normalization, pretraining, loss balancing
# ============================================================

import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================
# 1. Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_default_dtype(torch.float32)

# ============================================================
# 2. Load HDF5 Data
# ============================================================
with h5py.File("steel_cylinder_T_timeseries.h5", "r") as f:
    coords = torch.tensor(f["coords"][:], dtype=torch.float32)
    times  = torch.tensor(f["times"][:],  dtype=torch.float32)
    Tdata  = torch.tensor(f["T"][:],      dtype=torch.float32)

Ns = coords.shape[0]
Nt = times.shape[0]

if Tdata.shape == (Nt, Ns):
    Tdata = Tdata.T

print(f"Spatial points: {Ns}")
print(f"Time steps    : {Nt}")
print(f"Time range    : {times[0].item():.2f} to {times[-1].item():.2f} s")
print(f"Temp range    : {Tdata.min().item():.1f} to {Tdata.max().item():.1f} K")

# ============================================================
# 3. Normalization using mean/std (much better for PINNs)
# ============================================================
x_raw = coords[:, 0].repeat_interleave(Nt)
y_raw = coords[:, 1].repeat_interleave(Nt)
z_raw = coords[:, 2].repeat_interleave(Nt)
t_raw = times.repeat(Ns)
T_raw = Tdata.reshape(-1)

# Compute statistics
x_mean, x_std = x_raw.mean(), x_raw.std() + 1e-8
y_mean, y_std = y_raw.mean(), y_raw.std() + 1e-8
z_mean, z_std = z_raw.mean(), z_raw.std() + 1e-8
t_mean, t_std = t_raw.mean(), t_raw.std() + 1e-8
T_mean, T_std = T_raw.mean(), T_raw.std() + 1e-8

print(f"\nNormalization stats:")
print(f"  x: mean={x_mean:.4f}, std={x_std:.4f}")
print(f"  y: mean={y_mean:.4f}, std={y_std:.4f}")
print(f"  z: mean={z_mean:.4f}, std={z_std:.4f}")
print(f"  t: mean={t_mean:.4f}, std={t_std:.4f}")
print(f"  T: mean={T_mean:.1f}, std={T_std:.1f}")

# Normalize
x_n = (x_raw - x_mean) / x_std
y_n = (y_raw - y_mean) / y_std
z_n = (z_raw - z_mean) / z_std
t_n = (t_raw - t_mean) / t_std
T_n = (T_raw - T_mean) / T_std

inputs  = torch.stack([x_n, y_n, z_n, t_n], dim=1).to(device)
targets = T_n.unsqueeze(1).to(device)

def denormalize_temperature(T_norm):
    return T_norm * T_std + T_mean

print(f"Total training points: {inputs.shape[0]}")
print(f"Normalized T range: [{T_n.min():.2f}, {T_n.max():.2f}]")

# ============================================================
# 4. Network with Residual Connections
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.Tanh()

    def forward(self, x):
        return x + self.act(self.fc2(self.act(self.fc1(x))))


class PINN(nn.Module):
    def __init__(self, width=256, n_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(4, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(width, 1)
        self.act = nn.Tanh()

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


model = PINN(width=256, n_blocks=4).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ============================================================
# 5. Physics Residual with Correct Chain Rule
# ============================================================
kappa = 80.0
rho   = 7800.0
cp    = 450.0
alpha_phys = kappa / (rho * cp)
print(f"Thermal diffusivity α = {alpha_phys:.3e} m²/s")

def heat_equation_residual(model, inp):
    inp = inp.requires_grad_(True)
    T_pred = model(inp)

    grad = torch.autograd.grad(
        T_pred, inp, torch.ones_like(T_pred), create_graph=True
    )[0]

    dT_dxn = grad[:, 0:1]
    dT_dyn = grad[:, 1:2]
    dT_dzn = grad[:, 2:3]
    dT_dtn = grad[:, 3:4]

    d2T_dxn2 = torch.autograd.grad(
        dT_dxn, inp, torch.ones_like(dT_dxn), create_graph=True
    )[0][:, 0:1]

    d2T_dyn2 = torch.autograd.grad(
        dT_dyn, inp, torch.ones_like(dT_dyn), create_graph=True
    )[0][:, 1:2]

    d2T_dzn2 = torch.autograd.grad(
        dT_dzn, inp, torch.ones_like(dT_dzn), create_graph=True
    )[0][:, 2:3]

    # Chain rule: T_phys = T_n * T_std + T_mean
    # dT_phys/dt = T_std/t_std * dT_n/dt_n
    # d²T_phys/dx² = T_std/x_std² * d²T_n/dx_n²

    dT_dt = (T_std / t_std) * dT_dtn
    laplacian = (T_std / x_std**2) * d2T_dxn2 + \
                (T_std / y_std**2) * d2T_dyn2 + \
                (T_std / z_std**2) * d2T_dzn2

    residual = dT_dt - alpha_phys * laplacian

    # Normalize residual to make it O(1)
    residual_normalized = residual / (T_std / t_std + 1e-8)

    return T_pred, residual_normalized

# ============================================================
# 6. Training Setup
# ============================================================
batch_size = 4096
os.makedirs("results", exist_ok=True)

# ============================================================
# 7. PHASE 1: Data-Only Pretraining
# ============================================================
print("\n" + "="*70)
print("PHASE 1: DATA-ONLY PRETRAINING")
print("="*70)

pretrain_epochs = 2000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=pretrain_epochs, eta_min=1e-4
)

history = {"total": [], "data": [], "physics": [], "phase": []}
start_time = time.time()

for epoch in range(pretrain_epochs):
    idx = torch.randperm(inputs.shape[0], device=device)[:batch_size]
    inp_batch = inputs[idx]
    tgt_batch = targets[idx]

    optimizer.zero_grad()
    T_pred = model(inp_batch)
    data_loss = torch.mean((T_pred - tgt_batch) ** 2)
    data_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    history["total"].append(data_loss.item())
    history["data"].append(data_loss.item())
    history["physics"].append(0.0)
    history["phase"].append(1)

    if epoch % 500 == 0 or epoch == pretrain_epochs - 1:
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:5d}/{pretrain_epochs} | "
            f"Data Loss: {data_loss.item():.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

# ============================================================
# 8. PHASE 2: Physics-Informed Fine-Tuning
# ============================================================
print("\n" + "="*70)
print("PHASE 2: PHYSICS-INFORMED FINE-TUNING")
print("="*70)

finetune_epochs = 3000
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=finetune_epochs, eta_min=1e-6
)

for epoch in range(finetune_epochs):
    # Gradual physics weight ramp-up
    progress = min(epoch / 1000.0, 1.0)
    physics_weight = 0.001 + 0.099 * progress  # 0.001 → 0.1

    idx = torch.randperm(inputs.shape[0], device=device)[:batch_size]
    inp_batch = inputs[idx]
    tgt_batch = targets[idx]

    optimizer.zero_grad()

    T_pred, residual = heat_equation_residual(model, inp_batch)

    data_loss = torch.mean((T_pred - tgt_batch) ** 2)
    physics_loss = torch.mean(residual ** 2)
    loss = data_loss + physics_weight * physics_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    history["total"].append(loss.item())
    history["data"].append(data_loss.item())
    history["physics"].append(physics_loss.item())
    history["phase"].append(2)

    if epoch % 500 == 0 or epoch == finetune_epochs - 1:
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:5d}/{finetune_epochs} | "
            f"Total: {loss.item():.3e} | "
            f"Data: {data_loss.item():.6f} | "
            f"Physics: {physics_loss.item():.3e} | "
            f"λ: {physics_weight:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

total_time = time.time() - start_time
print(f"\nTotal training time: {total_time:.1f} seconds")

# ============================================================
# 9. Save Model
# ============================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "normalization": {
        "x_mean": x_mean.item(), "x_std": x_std.item(),
        "y_mean": y_mean.item(), "y_std": y_std.item(),
        "z_mean": z_mean.item(), "z_std": z_std.item(),
        "t_mean": t_mean.item(), "t_std": t_std.item(),
        "T_mean": T_mean.item(), "T_std": T_std.item(),
    },
    "history": history,
}, "results/pinn_heat_treatment_v2.pt")
print("Model saved.")

# ============================================================
# 10. Evaluation & Plots
# ============================================================
model.eval()
with torch.no_grad():
    # Full dataset prediction
    T_pred_all = model(inputs).cpu().squeeze()
    T_true_all = targets.cpu().squeeze()

    T_pred_phys = denormalize_temperature(T_pred_all)
    T_true_phys = denormalize_temperature(T_true_all)

    mae = torch.mean(torch.abs(T_pred_phys - T_true_phys)).item()
    rmse = torch.sqrt(torch.mean((T_pred_phys - T_true_phys)**2)).item()
    max_err = torch.max(torch.abs(T_pred_phys - T_true_phys)).item()
    rel_err = (rmse / T_std.item()) * 100

print("\n" + "="*50)
print("ERROR ANALYSIS")
print("="*50)
print(f"MAE      : {mae:.2f} K")
print(f"RMSE     : {rmse:.2f} K")
print(f"Max Error: {max_err:.2f} K")
print(f"Relative : {rel_err:.2f} %")
print("="*50)

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Loss curves
axes[0, 0].semilogy(history["data"], label="Data", alpha=0.6)
axes[0, 0].semilogy(history["total"], label="Total", alpha=0.6)
axes[0, 0].axvline(x=pretrain_epochs, color='red', linestyle='--', label='Phase 2 start')
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training Losses")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Physics loss (phase 2 only)
phys_losses = [h for h, p in zip(history["physics"], history["phase"]) if p == 2]
axes[0, 1].semilogy(phys_losses, alpha=0.6, color="green")
axes[0, 1].set_xlabel("Fine-tune Epoch")
axes[0, 1].set_ylabel("Physics Loss")
axes[0, 1].set_title("Physics Loss (Phase 2)")
axes[0, 1].grid(True)

# Parity plot at last time step
last_t_idx = Nt - 1
idx_last = torch.arange(last_t_idx, Ns * Nt, Nt)
T_pred_last = denormalize_temperature(T_pred_all[idx_last])
T_true_last = denormalize_temperature(T_true_all[idx_last])

axes[1, 0].scatter(T_true_last, T_pred_last, s=2, alpha=0.3)
t_range_plot = [T_true_last.min(), T_true_last.max()]
axes[1, 0].plot(t_range_plot, t_range_plot, "r--", linewidth=2, label="Perfect")
axes[1, 0].set_xlabel("OpenFOAM T [K]")
axes[1, 0].set_ylabel("PINN T [K]")
axes[1, 0].set_title(f"Last Time Step (t={times[-1].item():.1f}s)")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Error per time step
errors_per_t = []
for ti in range(Nt):
    idx_t = torch.arange(ti, Ns * Nt, Nt)
    err = torch.mean(torch.abs(T_pred_phys[idx_t] - T_true_phys[idx_t])).item()
    errors_per_t.append(err)

axes[1, 1].bar(range(Nt), errors_per_t, color="steelblue")
axes[1, 1].set_xlabel("Time Step Index")
axes[1, 1].set_ylabel("MAE [K]")
axes[1, 1].set_title("Error per Time Step")
axes[1, 1].grid(True, axis='y')

plt.tight_layout()
plt.savefig("results/training_results_v2.png", dpi=150)
print("Plot saved to results/training_results_v2.png")