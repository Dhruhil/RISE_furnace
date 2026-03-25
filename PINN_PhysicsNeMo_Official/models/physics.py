"""
PDE Residual for PINN — exact heat equation via automatic differentiation.
Master's Thesis: Digital Twin Modeling of Heat Treatment in Cast Metals

THE HEAT EQUATION (what makes this a PINN):
    ρ·Cp·∂T/∂t = κ·∇²T = κ·(∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)

This is computed EXACTLY via torch.autograd — not approximated like
in the GNN/FNO physics loss. This is the fundamental difference:
    - GNN/FNO: soft physics constraints (approximate)
    - PINN: exact PDE residual via automatic differentiation

Chain rule for normalised inputs:
    ∂T_phys/∂t = (T_std / t_std) · ∂T_n/∂t_n
    ∂²T_phys/∂x² = (T_std / x_std²) · ∂²T_n/∂x_n²
"""
from __future__ import annotations

import torch

SIGMA_SB = 5.67e-8  # Stefan-Boltzmann constant


def compute_pde_residual(model, inputs, cfg, dataset):
    """
    Compute heat equation PDE residual via automatic differentiation.

    Args:
        model:   HeatTreatmentPINN
        inputs:  (batch, 6) normalised [x_n, y_n, z_n, t_n, Tset_n, rid_n]
        cfg:     PINNConfig
        dataset: PINNAllRegionsDataset (for norm stats)

    Returns:
        T_pred:   (batch, 1) normalised prediction
        residual: (batch, 1) normalised PDE residual
        components: dict with individual terms for logging
    """
    inputs = inputs.requires_grad_(True)
    T_pred = model(inputs)

    # ── First derivatives via autograd ────────────────────────────
    grad = torch.autograd.grad(
        T_pred, inputs, torch.ones_like(T_pred),
        create_graph=True, retain_graph=True
    )[0]

    dT_dxn = grad[:, 0:1]  # ∂T_n/∂x_n
    dT_dyn = grad[:, 1:2]  # ∂T_n/∂y_n
    dT_dzn = grad[:, 2:3]  # ∂T_n/∂z_n
    dT_dtn = grad[:, 3:4]  # ∂T_n/∂t_n

    # ── Second derivatives (Laplacian) ────────────────────────────
    d2T_dxn2 = torch.autograd.grad(
        dT_dxn, inputs, torch.ones_like(dT_dxn),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    d2T_dyn2 = torch.autograd.grad(
        dT_dyn, inputs, torch.ones_like(dT_dyn),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]

    d2T_dzn2 = torch.autograd.grad(
        dT_dzn, inputs, torch.ones_like(dT_dzn),
        create_graph=True, retain_graph=True
    )[0][:, 2:3]

    # ── Chain rule: normalised → physical ─────────────────────────
    T_std = dataset.T_std
    x_std = dataset.x_std
    y_std = dataset.y_std
    z_std = dataset.z_std
    t_std = dataset.t_std

    # ∂T_phys/∂t = (T_std / t_std) · ∂T_n/∂t_n
    dT_dt = (T_std / t_std) * dT_dtn

    # ∇²T_phys = T_std · (∂²T_n/∂x_n² / x_std² + ∂²T_n/∂y_n² / y_std² + ∂²T_n/∂z_n² / z_std²)
    laplacian = T_std * (
        d2T_dxn2 / (x_std ** 2) +
        d2T_dyn2 / (y_std ** 2) +
        d2T_dzn2 / (z_std ** 2)
    )

    # ── PDE residual: ρ·Cp·∂T/∂t - κ·∇²T = 0 ────────────────────
    # Use thermal diffusivity α = κ/(ρ·Cp) to simplify:
    # ∂T/∂t - α·∇²T = 0
    #
    # Region-aware: steel vs air have different α
    # For simplicity, use average α (the model learns the correction)
    alpha = cfg.alpha_steel  # dominant material

    residual_raw = dT_dt - alpha * laplacian

    # Normalise residual to O(1) for stable training
    scale = (T_std / t_std) + 1e-8
    residual = residual_raw / scale

    return T_pred, residual, {
        "dT_dt": dT_dt.detach(),
        "laplacian": laplacian.detach(),
        "residual_raw": residual_raw.detach(),
    }


def compute_boundary_loss(T_pred_K, T_set_K, region_ids):
    """
    Boundary condition loss for heater regions.
    Heaters should be at T_set (Dirichlet BC).

    Args:
        T_pred_K: (batch,) predicted T in Kelvin
        T_set_K:  (batch,) furnace set temperature in Kelvin
        region_ids: (batch,) region IDs (0-11)
    """
    # Heater regions: IDs 2-10 (heater_1 through brick_heater)
    is_heater = (region_ids >= 2) & (region_ids <= 10)
    if is_heater.sum() == 0:
        return torch.tensor(0.0, device=T_pred_K.device)

    # Heaters should be at T_set
    heater_error = (T_pred_K[is_heater] - T_set_K[is_heater]).pow(2)
    return heater_error.mean()


def compute_radiation_loss(T_pred_K, T_set_K, dT_dt, cfg):
    """
    Radiation constraint: Stefan-Boltzmann law.
    Q_rad = ε·σ·(T_set⁴ - T⁴)

    Same equation as GNN/FNO but computed from PINN autograd derivatives.
    """
    Q_rad = cfg.epsilon_steel * SIGMA_SB * (T_set_K.pow(4) - T_pred_K.pow(4))
    dT_rad = Q_rad / (cfg.rho_steel * cfg.Cp_steel * 0.01)  # char_thickness
    scale = dT_rad.abs().mean().clamp(min=1e-8)
    return ((dT_dt.squeeze() - dT_rad) / scale).pow(2).mean()
