"""
training/loss.py
----------------
Loss functions for DeepONet_Unified.
Mirrors GNN_Unified/training/loss.py.
"""

import torch
import torch.nn as nn
from configs.base_config import TARGET_REGIONS, SIGMA, EPSILON_STEEL, CHAR_THICKNESS


class DeepONetLoss(nn.Module):
    """
    Weighted MSE loss with optional physics regularization.

    L = (1 - lambda) * L_data + lambda * L_physics

    L_data    : region-weighted MSE between predicted and true T
    L_physics : Stefan-Boltzmann radiation constraint on steel cylinder
    """

    def __init__(self, lambda_physics: float = 0.003):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss(reduction="none")

        # Region weights — steel cylinder gets highest weight
        self.region_weights = {
            "steel_cylinder": 10.0,
            "inner_box":       3.0,
            "outer_box":       1.0,
            "brick_heater":    1.0,
        }

    def forward(
        self,
        pred: torch.Tensor,      # (batch, 1) normalized prediction
        true: torch.Tensor,      # (batch, 1) normalized target
        x:    torch.Tensor,      # (batch, 8) trunk input
        a:    torch.Tensor,      # (batch, 18) branch input
        normalizer: dict,
        step: int = 0,
    ) -> torch.Tensor:

        # ── Data loss ─────────────────────────────────────────────────────
        # Per-region weighting via one-hot columns in x (columns 4:8)
        region_idx = x[:, 4:8].argmax(dim=1)  # (batch,)
        weights    = torch.ones(len(pred), device=pred.device)
        for r_idx, region in enumerate(TARGET_REGIONS):
            mask = (region_idx == r_idx)
            weights[mask] = self.region_weights.get(region, 1.0)

        mse_vals = self.mse(pred, true).squeeze(-1)  # (batch,)
        l_data   = (weights * mse_vals).mean()

        if self.lambda_physics <= 0:
            return l_data

        # ── Physics loss — Stefan-Boltzmann radiation constraint ──────────
        # Only applied on steel_cylinder nodes
        steel_mask = (region_idx == TARGET_REGIONS.index("steel_cylinder"))
        if steel_mask.sum() == 0:
            return l_data

        # Denormalize to Kelvin
        u_mean = normalizer["u_mean"]
        u_std  = normalizer["u_std"]
        pred_K = pred[steel_mask] * u_std + u_mean   # (N_steel, 1)
        true_K = true[steel_mask] * u_std + u_mean

        # T_set from branch input (column 0, denormalized)
        a_mean = torch.tensor(normalizer["a_mean"], device=a.device)
        a_std  = torch.tensor(normalizer["a_std"],  device=a.device)
        a_denorm = a[steel_mask] * a_std + a_mean    # (N_steel, 18)
        t_set_K  = a_denorm[:, 0:1]                  # (N_steel, 1)

        # Stefan-Boltzmann: q_rad = eps * sigma * (T_set^4 - T^4)
        q_rad  = EPSILON_STEEL * SIGMA * (t_set_K**4 - true_K**4)
        # Temperature rate from radiation
        mat    = {"rho": 7800.0, "Cp": 450.0}
        t_dot  = q_rad / (mat["rho"] * mat["Cp"] * CHAR_THICKNESS)

        # Predicted T_dot ≈ (pred - true) / delta_t  — use MSE as proxy
        l_rad  = ((pred_K - true_K - t_dot * 10.0) ** 2).mean()
        scale  = max(float(t_dot.abs().mean()), 1e-8)
        l_rad  = l_rad / (scale ** 2)

        return (1.0 - self.lambda_physics) * l_data + self.lambda_physics * l_rad
