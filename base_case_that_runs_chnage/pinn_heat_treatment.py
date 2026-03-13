# ============================================================
# Physics-Informed Neural Network (PINN)
# Heat Treatment of Steel Cylinder
# Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI
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
    Tdata = Tdata.T  # shape: (Ns, Nt)

print(f"Spatial points: {Ns}")
print(f"Time steps    : {Nt}")
print(f"Time range    : {times[0].item():.2f} to {times[-1].item():.2f} s")
print(f"Temp range    : {Tdata.min().item():.1f} to {Tdata.max().item():.1f} K")

# ============================================================
# 3. Normalization (CRITICAL for PINN training)
# ============================================================
# Spatial normalization
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
t_min, t_max = times.min(), times.max()

# Temperature normalization
T_min_val = Tdata.min()
T_max_val = Tdata.max()
T_range = T_max_val - T_min_val

def normalize_inputs(x, y, z, t):
    """Normalize to [-1, 1] range"""
    x_n = 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0
    y_n = 2.0 * (y - y_min) / (y_max - y_min + 1e-8) - 1.0
    z_n = 2.0 * (z - z_min) / (z_max - z_min + 1e-8) - 1.0
    t_n = 2.0 * (t - t_min) / (t_max - t_min + 1e-8) - 1.0
    return x_n, y_n, z_n, t_n

def normalize_temperature(T):
    """Normalize temperature to [0, 1]"""
    return (T - T_min_val) / (T_range + 1e-8)

def denormalize_temperature(T_norm):
    """Convert back to physical temperature"""
    return T_norm * T_range + T_min_val

# ============================================================
# 4. Space-Time Expansion
# ============================================================
x_raw = coords[:, 0].repeat_interleave(Nt)
y_raw = coords[:, 1].repeat_interleave(Nt)
z_raw = coords[:, 2].repeat_interleave(Nt)
t_raw = times.repeat(Ns)
T_raw = Tdata.reshape(-1)

# Normalize
x_n, y_n, z_n, t_n = normalize_inputs(x_raw, y_raw, z_raw, t_raw)
T_n = normalize_temperature(T_raw)

inputs  = torch.stack([x_n, y_n, z_n, t_n], dim=1).to(device)
targets = T_n.unsqueeze(1).to(device)

print(f"Total training points: {inputs.shape[0]}")

# ============================================================
# 5. Network Architecture
# ============================================================
class SirenLayer(nn.Module):
    """Sinusoidal activation — better for PINNs than tanh"""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_features) / omega_0,
                     np.sqrt(6 / in_features) / omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class PINN(nn.Module):
    def __init__(self, layer_sizes, activation="siren"):
        super().__init__()

        if activation == "siren":
            layers = []
            for i in range(len(layer_sizes) - 2):
                layers.append(
                    SirenLayer(layer_sizes[i], layer_sizes[i+1], is_first=(i == 0))
                )
            layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.net = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(len(layer_sizes) - 2):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = PINN([4, 256, 256, 256, 256, 256, 1], activation="siren").to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ============================================================
# 6. Manual Physics Residual (most reliable approach)
# ============================================================
kappa = 80.0     # W/(m·K)
rho   = 7800.0   # kg/m³
cp    = 450.0    # J/(kg·K)
alpha = kappa / (rho * cp)
print(f"Thermal diffusivity α = {alpha:.3e} m²/s")

def heat_equation_residual(model, inp):
    """
    Compute residual: dT/dt - α(d²T/dx² + d²T/dy² + d²T/dz²) = 0

    Works on NORMALIZED inputs but applies chain rule
    to account for the coordinate scaling.
    """
    inp = inp.requires_grad_(True)
    T_pred = model(inp)

    # First derivatives w.r.t. normalized coordinates
    grad = torch.autograd.grad(
        T_pred, inp, torch.ones_like(T_pred), create_graph=True
    )[0]

    dT_dxn = grad[:, 0:1]
    dT_dyn = grad[:, 1:2]
    dT_dzn = grad[:, 2:3]
    dT_dtn = grad[:, 3:4]

    # Second derivatives
    d2T_dxn2 = torch.autograd.grad(
        dT_dxn, inp, torch.ones_like(dT_dxn), create_graph=True
    )[0][:, 0:1]

    d2T_dyn2 = torch.autograd.grad(
        dT_dyn, inp, torch.ones_like(dT_dyn), create_graph=True
    )[0][:, 1:2]

    d2T_dzn2 = torch.autograd.grad(
        dT_dzn, inp, torch.ones_like(dT_dzn), create_graph=True
    )[0][:, 2:3]

    # Chain rule: d/dx = d/dx_n * (2 / (x_max - x_min))
    # So d²/dx² = d²/dx_n² * (2 / (x_max - x_min))²
    sx = 2.0 / (x_max - x_min + 1e-8)
    sy = 2.0 / (y_max - y_min + 1e-8)
    sz = 2.0 / (z_max - z_min + 1e-8)
    st = 2.0 / (t_max - t_min + 1e-8)

    # T_physical = T_norm * T_range + T_min
    # So dT_phys/dt = T_range * dT_norm/dt_norm * st
    # d²T_phys/dx² = T_range * d²T_norm/dx_n² * sx²

    dT_dt_phys = T_range * dT_dtn * st
    laplacian_phys = T_range * (d2T_dxn2 * sx**2 + d2T_dyn2 * sy**2 + d2T_dzn2 * sz**2)

    residual = dT_dt_phys - alpha * laplacian_phys

    return T_pred, residual

# ============================================================
# 7. Mini-batch Training (needed for 48k+ points)
# ============================================================
batch_size = 4096
n_total = inputs.shape[0]

# Create output directory
os.makedirs("results", exist_ok=True)

# ============================================================
# 8. Training Schedule
# ============================================================
epochs = 5000
physics_weight_init = 0.001
physics_weight_final = 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# Loss history
history = {
    "total": [], "data": [], "physics": [], "lr": []
}

print("\n" + "="*70)
print("TRAINING STARTED")
print("="*70)

start_time = time.time()

for epoch in range(epochs):
    # Gradually increase physics weight (curriculum learning)
    progress = epoch / max(epochs - 1, 1)
    physics_weight = physics_weight_init + (physics_weight_final - physics_weight_init) * progress

    # Random mini-batch
    idx = torch.randperm(n_total, device=device)[:batch_size]
    inp_batch = inputs[idx]
    tgt_batch = targets[idx]

    optimizer.zero_grad()

    T_pred, residual = heat_equation_residual(model, inp_batch)

    data_loss = torch.mean((T_pred - tgt_batch) ** 2)
    physics_loss = torch.mean(residual ** 2)
    loss = data_loss + physics_weight * physics_loss

    loss.backward()

    # Gradient clipping (prevents exploding gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    # Record
    history["total"].append(loss.item())
    history["data"].append(data_loss.item())
    history["physics"].append(physics_loss.item())
    history["lr"].append(optimizer.param_groups[0]["lr"])

    if epoch % 500 == 0 or epoch == epochs - 1:
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:5d}/{epochs} | "
            f"Total: {loss.item():.3e} | "
            f"Data: {data_loss.item():.3e} | "
            f"Physics: {physics_loss.item():.3e} | "
            f"λ_phys: {physics_weight:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.1f} seconds")

# ============================================================
# 9. Save Model
# ============================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "normalization": {
        "x_min": x_min.item(), "x_max": x_max.item(),
        "y_min": y_min.item(), "y_max": y_max.item(),
        "z_min": z_min.item(), "z_max": z_max.item(),
        "t_min": t_min.item(), "t_max": t_max.item(),
        "T_min": T_min_val.item(), "T_max": T_max_val.item(),
    },
    "history": history,
}, "results/pinn_heat_treatment.pt")

print("Model saved to results/pinn_heat_treatment.pt")

# ============================================================
# 10. Plot Training History
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].semilogy(history["total"], label="Total", alpha=0.7)
axes[0].semilogy(history["data"], label="Data", alpha=0.7)
axes[0].semilogy(history["physics"], label="Physics", alpha=0.7)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Losses")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["lr"])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Learning Rate")
axes[1].set_title("Learning Rate Schedule")
axes[1].grid(True)

# Prediction vs Ground Truth at final time
with torch.no_grad():
    # Get data at last time step
    last_t_idx = Nt - 1
    idx_last = torch.arange(last_t_idx, Ns * Nt, Nt)
    inp_last = inputs[idx_last]
    T_pred_last = model(inp_last).cpu().squeeze()
    T_true_last = targets[idx_last].cpu().squeeze()

    # Denormalize
    T_pred_phys = denormalize_temperature(T_pred_last)
    T_true_phys = denormalize_temperature(T_true_last)

axes[2].scatter(T_true_phys, T_pred_phys, s=1, alpha=0.3)
axes[2].plot(
    [T_true_phys.min(), T_true_phys.max()],
    [T_true_phys.min(), T_true_phys.max()],
    "r--", label="Perfect"
)
axes[2].set_xlabel("OpenFOAM Temperature [K]")
axes[2].set_ylabel("PINN Temperature [K]")
axes[2].set_title(f"Prediction at t = {times[-1].item():.1f} s")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("results/training_results.png", dpi=150)
print("Plot saved to results/training_results.png")

# ============================================================
# 11. Error Analysis
# ============================================================
with torch.no_grad():
    T_pred_all = model(inputs).cpu().squeeze()
    T_true_all = targets.cpu().squeeze()

    T_pred_all_phys = denormalize_temperature(T_pred_all)
    T_true_all_phys = denormalize_temperature(T_true_all)

    mae = torch.mean(torch.abs(T_pred_all_phys - T_true_all_phys)).item()
    rmse = torch.sqrt(torch.mean((T_pred_all_phys - T_true_all_phys)**2)).item()
    max_err = torch.max(torch.abs(T_pred_all_phys - T_true_all_phys)).item()
    rel_err = (rmse / T_range.item()) * 100

print("\n" + "="*50)
print("ERROR ANALYSIS")
print("="*50)
print(f"MAE      : {mae:.2f} K")
print(f"RMSE     : {rmse:.2f} K")
print(f"Max Error: {max_err:.2f} K")
print(f"Relative : {rel_err:.2f} %")
print("="*50)