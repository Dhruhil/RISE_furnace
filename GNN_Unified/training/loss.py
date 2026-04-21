"""
Physics-Informed Loss for Heat Treatment GNN — fully compatible with
the new BaseConfig (char_thickness, epsilon_steel) and new dataset.py
(kappa_raw, Cp_raw, rho_raw, T_set_raw attached directly to Data objects).

THREE PHYSICS EQUATIONS:

  1. CONDUCTION — Fourier's Law
       rho * Cp * dT/dt = kappa * laplacian(T)
     Graph Laplacian: laplacian(T_i) = mean_j(T_j - T_i)
     This is exactly how OpenFOAM discretises the diffusion term.

  2. CONVECTION — Newton's Law of Cooling
       Q_conv = h * (T_set - T_surface)
     Constraints enforced:
       a) T_steel must not exceed T_set
       b) T_steel must rise toward T_set during heating phase

  3. RADIATION — Stefan-Boltzmann Law
       Q_rad = epsilon * sigma * (T_set^4 - T^4)
       dT/dt_rad = Q_rad / (rho * Cp * thickness)
     epsilon = 0.80, sigma = 5.67e-8 W/(m²·K⁴)

TOTAL LOSS:
    L = L_data + lambda(epoch) * (w_c*L_cond + w_v*L_conv + w_r*L_rad)

CURRICULUM (lambda grows over training):
    Epoch   1– 50 : 0.001  (learn data patterns first)
    Epoch  51–100 : 0.01   (light physics)
    Epoch 101–150 : 0.05   (balanced)
    Epoch 151–200 : 0.10   (full physics enforcement)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_SB: float = 5.67e-8   # Stefan-Boltzmann [W/(m²·K⁴)]


# ─────────────────────────────────────────────────────────────────────────────
# Data loss
# ─────────────────────────────────────────────────────────────────────────────

class MaskedMSELoss(nn.Module):
    """MSE loss that skips NaN targets."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], target[mask])


# ─────────────────────────────────────────────────────────────────────────────
# Physics loss 1 — Conduction (Fourier's Law)
# ─────────────────────────────────────────────────────────────────────────────

def conduction_loss(
    delta_T_pred_norm: torch.Tensor,
    T_current_K:       torch.Tensor,
    edge_index:        torch.Tensor,
    kappa:             torch.Tensor,
    rho:               torch.Tensor,
    Cp:                torch.Tensor,
    dt:                float,
    Y_std:             float,
    dT_std:            float = 1.0,
    dT_mean:           float = 0.0,
) -> torch.Tensor:
    N   = T_current_K.shape[0]
    src = edge_index[0]
    dst = edge_index[1]

    dT_dt = (delta_T_pred_norm.squeeze(-1) * dT_std + dT_mean) / dt

    T_diff = T_current_K[dst] - T_current_K[src]
    lap_T  = torch.zeros(N, device=T_current_K.device, dtype=T_current_K.dtype)
    degree = torch.zeros(N, device=T_current_K.device, dtype=T_current_K.dtype)
    lap_T.scatter_add_(0, src, T_diff)
    degree.scatter_add_(0, src, torch.ones_like(T_diff))
    lap_T = lap_T / degree.clamp(min=1.0)

    residual = rho * Cp * dT_dt - kappa * lap_T
    scale    = (rho * Cp * dT_dt.abs()).mean().clamp(min=1.0)
    return (residual / scale).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Physics loss 2 — Convection (Newton's Law)
# ─────────────────────────────────────────────────────────────────────────────

def convection_loss(
    T_pred_next_K: torch.Tensor,
    T_current_K:   torch.Tensor,
    T_set_K:       torch.Tensor,
) -> torch.Tensor:
    overshoot     = F.relu(T_pred_next_K - T_set_K)
    heating_phase = (T_current_K < T_set_K).float()
    wrong_dir     = heating_phase * F.relu(T_current_K - T_pred_next_K)
    scale         = T_set_K.mean().clamp(min=300.0)
    return ((overshoot + 0.5 * wrong_dir) / scale).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Physics loss 3 — Radiation (Stefan-Boltzmann Law)
# ─────────────────────────────────────────────────────────────────────────────

def radiation_loss(
    delta_T_pred_norm: torch.Tensor,
    T_current_K:       torch.Tensor,
    T_set_K:           torch.Tensor,
    rho:               torch.Tensor,
    Cp:                torch.Tensor,
    dt:                float,
    Y_std:             float,
    epsilon:           float = 0.80,
    thickness:         float = 0.01,
    dT_std:            float = 1.0,
    dT_mean:           float = 0.0,
) -> torch.Tensor:
    dT_dt_model = (delta_T_pred_norm.squeeze(-1) * dT_std + dT_mean) / dt
    Q_rad       = epsilon * SIGMA_SB * (T_set_K.pow(4) - T_current_K.pow(4))
    dT_dt_rad   = Q_rad / (rho * Cp * thickness)
    scale       = dT_dt_rad.abs().mean().clamp(min=1e-8)
    return ((dT_dt_model - dT_dt_rad) / scale).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combined Physics-Informed Loss
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    Combined data + physics loss.

    Args:
        lambda_physics : physics weight (updated each epoch by curriculum)
        w_cond         : weight for Fourier conduction term
        w_conv         : weight for Newton convection term
        w_rad          : weight for Stefan-Boltzmann radiation term
        epsilon_steel  : steel surface emissivity (0.7–0.9)
        char_thickness : characteristic surface thickness for radiation [m]
    """

    def __init__(
        self,
        lambda_physics: float = 0.001,
        w_cond:         float = 1.0,
        w_conv:         float = 0.5,
        w_rad:          float = 0.3,
        epsilon_steel:  float = 0.80,
        char_thickness: float = 0.01,    # ← was missing from old loss.py
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.w_cond         = w_cond
        self.w_conv         = w_conv
        self.w_rad          = w_rad
        self.epsilon        = epsilon_steel
        self.thickness      = char_thickness
        self._mse           = MaskedMSELoss()

    def forward(
        self,
        delta_T_pred: torch.Tensor,   # (N, 1) normalised model output
        target:       torch.Tensor,   # (N, 1) normalised ground truth delta_T
        batch,                        # PyG Batch
        Y_std:        float,
        dt:           float = 10.0,
        dT_std:       float = 1.0,
        dT_mean:      float = 0.0,   
 ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, breakdown_dict).
        breakdown_dict keys: data, cond, conv, rad, physics, total
        """

        L_data = self._mse(delta_T_pred, target)

        if self.lambda_physics < 1e-10:
            return L_data, {
                "data": L_data.item(), "cond": 0.0,
                "conv": 0.0, "rad": 0.0, "physics": 0.0,
                "total": L_data.item(),
            }

        dev = delta_T_pred.device

        # ── Get raw physical quantities ──────────────────────────────
        # The fixed dataset.py attaches kappa_raw, Cp_raw, rho_raw, T_set_raw
        # directly to every Data object. Use them if available.
        if hasattr(batch, "kappa_raw") and batch.kappa_raw is not None:
            kappa = batch.kappa_raw.to(dev)
            Cp    = batch.Cp_raw.to(dev)
            rho   = batch.rho_raw.to(dev)
            T_set = batch.T_set_raw.to(dev)
        else:
            # Fallback: de-normalise from node features using stored stats
            x = batch.x
            if hasattr(batch, "node_mean") and batch.node_mean.numel() >= 10:
                nm = batch.node_mean.view(-1)
                ns = batch.node_std.view(-1)
                kappa = x[:, 7] * ns[7] + nm[7]
                Cp    = x[:, 8] * ns[8] + nm[8]
                rho   = x[:, 9] * ns[9] + nm[9]
                T_set = x[:, 4] * ns[4] + nm[4]
            else:
                # Last resort: typical steel values
                N     = delta_T_pred.shape[0]
                kappa = torch.full((N,), 60.0,   device=dev)
                Cp    = torch.full((N,), 450.0,  device=dev)
                rho   = torch.full((N,), 7800.0, device=dev)
                T_set = torch.full((N,), 1000.0, device=dev)

        T_now  = batch.T_current.to(dev)
        T_next = T_now + delta_T_pred.squeeze(-1) * Y_std

        # ── Three physics losses ─────────────────────────────────────
        L_cond = conduction_loss(
            delta_T_pred, T_now, batch.edge_index,
            kappa, rho, Cp, dt, Y_std,
            dT_std=dT_std, dT_mean=dT_mean, 
        )
        L_conv = convection_loss(T_next, T_now, T_set)
        L_rad  = radiation_loss(
            delta_T_pred, T_now, T_set, rho, Cp, dt, Y_std,
            epsilon=self.epsilon, thickness=self.thickness,
            dT_std=dT_std, dT_mean=dT_mean,
        )

        L_physics = self.w_cond * L_cond + self.w_conv * L_conv + self.w_rad * L_rad
        L_total   = L_data + self.lambda_physics * L_physics

        # Equilibrium constraint: near T_set, dT should approach zero
        delta_T_K = delta_T_pred.squeeze(-1) * Y_std
        gap = (T_set - T_now).abs()
        near_eq_weight = torch.exp(-gap / 20.0)
        L_eq = (delta_T_K * near_eq_weight).pow(2).mean()
        L_total = L_total + 0.5 * L_eq

        return L_total, {
            "data":    L_data.item(),
            "cond":    L_cond.item(),
            "conv":    L_conv.item(),
            "rad":     L_rad.item(),
            "physics": L_physics.item(),
            "total":   L_total.item(),
        }
