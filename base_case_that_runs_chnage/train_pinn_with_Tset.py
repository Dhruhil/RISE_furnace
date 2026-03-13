#!/usr/bin/env python3
"""
Train PINN with T_set as input: T = f(x, y, z, t, T_set, kappa, Cp, rho)
"""
import torch
import torch.nn as nn
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset_path = "/workspace/rise_furnace/parameter_study/combined_dataset.h5"
if not os.path.isfile(dataset_path):
    print(f"ERROR: {dataset_path} not found!")
    print(f"Run step3_create_datasets.py first!")
    exit()

with h5py.File(dataset_path, "r") as f:
    X_norm = torch.tensor(f["X_norm"][:], dtype=torch.float32)
    Y_norm = torch.tensor(f["Y_norm"][:], dtype=torch.float32)
    X_mean = torch.tensor(f["X_mean"][:], dtype=torch.float32)
    X_std  = torch.tensor(f["X_std"][:],  dtype=torch.float32)
    Y_mean = float(f["Y_mean"][()])
    Y_std  = float(f["Y_std"][()])
    cols   = json.loads(f.attrs["columns"])
    n_sims = int(f.attrs["n_simulations"])

n_inputs = X_norm.shape[1]
inputs  = X_norm.to(device)
targets = Y_norm.to(device)
print(f"Dataset: {inputs.shape[0]:,} points, {n_inputs} inputs")
print(f"Columns: {cols}")
print(f"Simulations: {n_sims}")

def denormalize_T(T_n):
    return T_n * Y_std + Y_mean

torch.manual_seed(42)
N = inputs.shape[0]
idx = torch.randperm(N)
n_train = int(0.80 * N)
n_val = int(0.10 * N)
n_test = N - n_train - n_val
train_inputs = inputs[idx[:n_train]]
train_targets = targets[idx[:n_train]]
val_inputs = inputs[idx[n_train:n_train+n_val]]
val_targets = targets[idx[n_train:n_train+n_val]]
test_inputs = inputs[idx[n_train+n_val:]]
test_targets = targets[idx[n_train+n_val:]]
print(f"Split: Train={n_train:,} Val={n_val:,} Test={n_test:,}")

class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.fc1 = nn.Linear(w, w)
        self.fc2 = nn.Linear(w, w)
        self.act = nn.Tanh()
    def forward(self, x):
        return x + self.act(self.fc2(self.act(self.fc1(x))))

class PINN_Tset(nn.Module):
    def __init__(self, n_in, width=256, n_blocks=5):
        super().__init__()
        self.input_layer = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(width, 1)
        self.act = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.act(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

model = PINN_Tset(n_in=n_inputs, width=256, n_blocks=5).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

def physics_residual(model, inp):
    inp = inp.requires_grad_(True)
    T_pred = model(inp)
    grad = torch.autograd.grad(T_pred, inp, torch.ones_like(T_pred), create_graph=True)[0]
    dT_dx, dT_dy, dT_dz, dT_dt = grad[:,0:1], grad[:,1:2], grad[:,2:3], grad[:,3:4]
    d2T_dx2 = torch.autograd.grad(dT_dx, inp, torch.ones_like(dT_dx), create_graph=True)[0][:,0:1]
    d2T_dy2 = torch.autograd.grad(dT_dy, inp, torch.ones_like(dT_dy), create_graph=True)[0][:,1:2]
    d2T_dz2 = torch.autograd.grad(dT_dz, inp, torch.ones_like(dT_dz), create_graph=True)[0][:,2:3]
    kappa_phys = inp[:,5:6] * X_std[5].to(device) + X_mean[5].to(device)
    Cp_phys = inp[:,6:7] * X_std[6].to(device) + X_mean[6].to(device)
    rho_phys = inp[:,7:8] * X_std[7].to(device) + X_mean[7].to(device)
    alpha = kappa_phys / (rho_phys * Cp_phys + 1e-8)
    sx, sy, sz, st = X_std[0].to(device), X_std[1].to(device), X_std[2].to(device), X_std[3].to(device)
    dT_dt_phys = (Y_std / st) * dT_dt
    laplacian = (Y_std/sx**2)*d2T_dx2 + (Y_std/sy**2)*d2T_dy2 + (Y_std/sz**2)*d2T_dz2
    residual = (dT_dt_phys - alpha * laplacian) / (Y_std / st + 1e-8)
    return T_pred, residual

def evaluate(model, inp, tgt):
    model.eval()
    with torch.no_grad():
        T_pred = denormalize_T(model(inp).cpu().squeeze())
        T_true = denormalize_T(tgt.cpu().squeeze())
        mae = torch.mean(torch.abs(T_pred - T_true)).item()
        rmse = torch.sqrt(torch.mean((T_pred - T_true)**2)).item()
        max_err = torch.max(torch.abs(T_pred - T_true)).item()
    model.train()
    return {"mae": mae, "rmse": rmse, "max_err": max_err}

batch_size = 4096
os.makedirs("results", exist_ok=True)
history = {"train_data": [], "train_phys": [], "val_mae": [], "val_rmse": []}
best_val_mae = float("inf")
best_state = None
start_time = time.time()

print(f"\n{'='*60}")
print(f"PHASE 1: DATA PRETRAINING (2000 epochs)")
print(f"{'='*60}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, 1e-4)
for epoch in range(2000):
    model.train()
    bi = torch.randperm(n_train, device=device)[:batch_size]
    optimizer.zero_grad()
    loss = torch.mean((model(train_inputs[bi]) - train_targets[bi])**2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    history["train_data"].append(loss.item())
    history["train_phys"].append(0.0)
    if epoch % 500 == 0 or epoch == 1999:
        val = evaluate(model, val_inputs, val_targets)
        history["val_mae"].append(val["mae"])
        history["val_rmse"].append(val["rmse"])
        if val["mae"] < best_val_mae:
            best_val_mae = val["mae"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch:5d} | Data: {loss.item():.6f} | Val MAE: {val['mae']:.2f} K | Time: {time.time()-start_time:.1f}s")

print(f"\n{'='*60}")
print(f"PHASE 2: PHYSICS FINE-TUNING (3000 epochs)")
print(f"{'='*60}")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3000, 1e-6)
for epoch in range(3000):
    model.train()
    lam = 0.001 + 0.099 * min(epoch/1000.0, 1.0)
    bi = torch.randperm(n_train, device=device)[:batch_size]
    optimizer.zero_grad()
    T_pred, residual = physics_residual(model, train_inputs[bi])
    data_loss = torch.mean((T_pred - train_targets[bi])**2)
    phys_loss = torch.mean(residual**2)
    loss = data_loss + lam * phys_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    history["train_data"].append(data_loss.item())
    history["train_phys"].append(phys_loss.item())
    if epoch % 500 == 0 or epoch == 2999:
        val = evaluate(model, val_inputs, val_targets)
        history["val_mae"].append(val["mae"])
        history["val_rmse"].append(val["rmse"])
        if val["mae"] < best_val_mae:
            best_val_mae = val["mae"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch:5d} | Data: {data_loss.item():.6f} | Phys: {phys_loss.item():.3e} | Val MAE: {val['mae']:.2f} K | Time: {time.time()-start_time:.1f}s")

total_time = time.time() - start_time
model.load_state_dict(best_state)
model.eval()
train_m = evaluate(model, train_inputs, train_targets)
val_m = evaluate(model, val_inputs, val_targets)
test_m = evaluate(model, test_inputs, test_targets)

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"{'Metric':<15} {'Train':>10} {'Val':>10} {'Test':>10}")
print("-"*50)
print(f"{'MAE (K)':<15} {train_m['mae']:>10.2f} {val_m['mae']:>10.2f} {test_m['mae']:>10.2f}")
print(f"{'RMSE (K)':<15} {train_m['rmse']:>10.2f} {val_m['rmse']:>10.2f} {test_m['rmse']:>10.2f}")
print(f"{'Max Err (K)':<15} {train_m['max_err']:>10.2f} {val_m['max_err']:>10.2f} {test_m['max_err']:>10.2f}")

with h5py.File(dataset_path, "r") as f:
    X_raw = torch.tensor(f["X_raw"][:], dtype=torch.float32)
unique_Tset = torch.unique(X_raw[:, 4])
print(f"\n{'T_set':<8} {'MAE':>8} {'RMSE':>8} {'MaxErr':>8} {'Points':>10}")
print("-"*45)
with torch.no_grad():
    T_pred_all = denormalize_T(model(inputs).cpu().squeeze())
    T_true_all = denormalize_T(targets.cpu().squeeze())
    for ts in unique_Tset:
        mask = X_raw[:,4] == ts
        mae = torch.mean(torch.abs(T_pred_all[mask]-T_true_all[mask])).item()
        rmse = torch.sqrt(torch.mean((T_pred_all[mask]-T_true_all[mask])**2)).item()
        maxe = torch.max(torch.abs(T_pred_all[mask]-T_true_all[mask])).item()
        print(f"{ts.item():<8.0f} {mae:>8.2f} {rmse:>8.2f} {maxe:>8.2f} {mask.sum().item():>10,}")

with torch.no_grad():
    _ = model(inputs[:100])
    if device.type == "cuda": torch.cuda.synchronize()
    start = time.time()
    for _ in range(100): _ = model(inputs)
    if device.type == "cuda": torch.cuda.synchronize()
    pinn_time = (time.time()-start)/100
openfoam_time = 1800
print(f"\nSpeed: OpenFOAM={openfoam_time}s, PINN={pinn_time:.4f}s, Speedup={openfoam_time/pinn_time:,.0f}x")

torch.save({"model_state_dict": best_state, "metrics": {"train": train_m, "val": val_m, "test": test_m}, "history": history, "n_inputs": n_inputs, "columns": cols, "n_sims": n_sims, "training_time": total_time}, "results/pinn_Tset_model.pt")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes[0,0].semilogy(history["train_data"], alpha=0.5)
axes[0,0].axvline(x=2000, color='red', linestyle='--')
axes[0,0].set_title("Data Loss")
axes[0,0].grid(True)
phys = [p for p in history["train_phys"] if p > 0]
if phys: axes[0,1].semilogy(phys, alpha=0.5, color="green")
axes[0,1].set_title("Physics Loss")
axes[0,1].grid(True)
axes[0,2].plot(history["val_mae"], "o-", color="orange")
axes[0,2].set_title("Val MAE (K)")
axes[0,2].grid(True)
axes[1,0].scatter(T_true_all.numpy(), T_pred_all.numpy(), s=1, alpha=0.05)
lims = [T_true_all.min().item(), T_true_all.max().item()]
axes[1,0].plot(lims, lims, "r--")
axes[1,0].set_title(f"All (MAE={evaluate(model,inputs,targets)['mae']:.2f}K)")
axes[1,0].grid(True)
with torch.no_grad():
    tp = denormalize_T(model(test_inputs).cpu().squeeze())
    tt = denormalize_T(test_targets.cpu().squeeze())
axes[1,1].scatter(tt.numpy(), tp.numpy(), s=1, alpha=0.1)
axes[1,1].plot(lims, lims, "r--")
axes[1,1].set_title(f"Test (MAE={test_m['mae']:.2f}K)")
axes[1,1].grid(True)
tset_maes = []
with torch.no_grad():
    for ts in unique_Tset:
        mask = X_raw[:,4]==ts
        tset_maes.append(torch.mean(torch.abs(T_pred_all[mask]-T_true_all[mask])).item())
axes[1,2].bar([str(int(t)) for t in unique_Tset.numpy()], tset_maes, color="steelblue")
axes[1,2].set_title("Error per T_set")
axes[1,2].grid(True)
plt.suptitle(f"PINN T_set - {n_sims} sims, {n_params:,} params", fontsize=16)
plt.tight_layout()
plt.savefig("results/pinn_Tset_results.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/pinn_Tset_results.png")
print("DONE!")
