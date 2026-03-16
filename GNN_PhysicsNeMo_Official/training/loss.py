"""
Physics-Informed Loss for Heat Treatment GNN.

Your thesis states the model must reflect three heat transfer mechanisms:

  1. CONDUCTION  — heat moves through the solid steel cylinder
                   Fourier's Law:
                       rho * Cp * dT/dt  =  kappa * laplacian(T)
                   rho   = density        [kg/m³]
                   Cp    = specific heat  [J/kg·K]
                   kappa = conductivity   [W/m·K]
                   This is the main governing equation for the steel.

  2. CONVECTION  — heat transfers between steel surface and furnace gas
                   Newton's Law of Cooling:
                       Q_conv = h * (T_fluid - T_surface)   [W/m²]
                   T_fluid = T_set (heater set-point in your simulation)
                   Enforced as: steel temperature must not exceed T_set.

  3. RADIATION   — heaters radiate energy to the cylinder surface
                   Stefan-Boltzmann Law:
                       Q_rad = epsilon * sigma * (T_heater^4 - T_surface^4)
                   sigma   = 5.67e-8  W/(m²·K⁴)
                   epsilon = 0.8      (typical emissivity of steel)
                   This drives heating when T_heater >> T_surface.

Total loss used during training:
    L_total = L_data  +  lambda * (L_conduction + L_convection + L_radiation)

Lambda schedule (curriculum — starts small so model learns data first):
    Epoch   1– 50 :  lambda = 0.001   (mostly data loss)
    Epoch  51–100 :  lambda = 0.01    (add light physics)
    Epoch 101–150 :  lambda = 0.05    (balanced)
    Epoch 151–200 :  lambda = 0.10    (full physics enforcement)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Stefan-Boltzmann constant [W / (m² · K⁴)]
SIGMA = 5.67e-8


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOSS  (already in your code — kept here for completeness)
# ─────────────────────────────────────────────────────────────────────────────
class MaskedMSELoss(nn.Module):
    """Standard MSE loss that skips any NaN entries."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], target[mask])


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS LOSS 1 — CONDUCTION  (Fourier's Law)
#
#   rho * Cp * dT/dt  =  kappa * laplacian(T)
#
# The graph Laplacian approximates the spatial second derivative:
#   laplacian(T_i) ≈ mean over neighbours j of [ T_j - T_i ]
# This is the standard finite-difference-on-graph approximation,
# exactly how OpenFOAM discretises the diffusion term.
#
# Residual that should be zero:
#   R_i = rho_i * Cp_i * (delta_T_i / dt)  -  kappa_i * laplacian(T_i)
# ─────────────────────────────────────────────────────────────────────────────
def conduction_loss(
    delta_T_pred_norm: torch.Tensor,  # (N,1) model output, normalised
    T_current_K:       torch.Tensor,  # (N,)  current temperature [K]
    edge_index:        torch.Tensor,  # (2,E) k-NN graph
    kappa:             torch.Tensor,  # (N,)  thermal conductivity [W/m·K]
    rho:               torch.Tensor,  # (N,)  density [kg/m³]
    Cp:                torch.Tensor,  # (N,)  specific heat [J/kg·K]
    dt:                float,         # time step [s]
    Y_std:             float,         # std used to normalise ΔT
) -> torch.Tensor:
    N   = T_current_K.shape[0]
    src = edge_index[0]               # source nodes
    dst = edge_index[1]               # destination / neighbour nodes

    # Convert normalised model output → physical dT/dt  [K/s]
    dT_dt = (delta_T_pred_norm.squeeze(-1) * Y_std) / dt

    # Graph Laplacian:  laplacian(T_i) = mean_j [ T_j - T_i ]
    T_diff = T_current_K[dst] - T_current_K[src]         # (E,) [K]
    lap_T  = torch.zeros(N, device=T_current_K.device)
    degree = torch.zeros(N, device=T_current_K.device)
    lap_T.scatter_add_(0, src, T_diff)
    degree.scatter_add_(0, src, torch.ones_like(T_diff))
    lap_T = lap_T / degree.clamp(min=1.0)                # (N,) [K / node-unit²]

    # Fourier residual:  rho * Cp * dT/dt  -  kappa * laplacian(T)  → 0
    residual = rho * Cp * dT_dt - kappa * lap_T          # (N,) [W/m³ equivalent]

    # Normalise so loss is dimensionless
    scale = (kappa * lap_T.abs()).mean().clamp(min=1.0)
    return (residual / scale).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS LOSS 2 — CONVECTION  (Newton's Law of Cooling)
#
#   Q_conv = h * (T_fluid - T_surface)
#
# In your simulation:
#   T_fluid   ≈ T_set  (heater set-point temperature)
#   T_surface = temperature of the steel cells
#
# Physical constraints enforced:
#   a) Steel temperature cannot exceed T_set (energy conservation)
#   b) Steel temperature must increase toward T_set during heating phase
#      (heat always flows from hot heater → cold steel, not the reverse)
# ─────────────────────────────────────────────────────────────────────────────
def convection_loss(
    T_pred_next_K: torch.Tensor,  # (N,)  predicted T at t+dt  [K]
    T_current_K:   torch.Tensor,  # (N,)  current T at t       [K]
    T_set_K:       torch.Tensor,  # (N,)  heater set-point      [K]
) -> torch.Tensor:

    # Constraint a: predicted T must not exceed heater temperature
    # relu(x) = max(0, x) — only penalises when T_pred > T_set
    over_heater = F.relu(T_pred_next_K - T_set_K)         # (N,) [K]

    # Constraint b: temperature must not drop during the heating phase
    # (steel is colder than T_set, so it should always be warming up)
    heating_phase = T_current_K < T_set_K                 # (N,) bool mask
    cooling_down  = F.relu(T_current_K - T_pred_next_K)   # (N,) [K]
    wrong_dir     = heating_phase.float() * cooling_down   # only penalise when heating

    scale = T_set_K.mean().clamp(min=300.0)
    return ((over_heater + 0.5 * wrong_dir) / scale).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS LOSS 3 — RADIATION  (Stefan-Boltzmann Law)
#
#   Q_rad = epsilon * sigma * (T_heater^4 - T_surface^4)   [W/m²]
#
# The temperature rise rate driven by radiation:
#   dT/dt_radiation = Q_rad / (rho * Cp * thickness)
#                   = epsilon * sigma * (T_set^4 - T^4) / (rho * Cp * thickness)
#
# Residual:  predicted dT/dt  should be consistent with this radiation rate.
# If the model predicts very fast heating but radiation flux is small, or
# predicts slow heating when the flux is large, this loss penalises it.
# ─────────────────────────────────────────────────────────────────────────────
def radiation_loss(
    delta_T_pred_norm: torch.Tensor,  # (N,1) model output, normalised
    T_current_K:       torch.Tensor,  # (N,)  current T   [K]
    T_set_K:           torch.Tensor,  # (N,)  heater T    [K]
    rho:               torch.Tensor,  # (N,)  density     [kg/m³]
    Cp:                torch.Tensor,  # (N,)  specific heat [J/kg·K]
    dt:                float,
    Y_std:             float,
    epsilon:           float = 0.8,   # steel emissivity (0.7–0.9 typical)
    thickness:         float = 0.01,  # characteristic surface thickness [m]
) -> torch.Tensor:

    # Physical dT/dt from model  [K/s]
    dT_dt_pred = (delta_T_pred_norm.squeeze(-1) * Y_std) / dt

    # Radiative heat flux → temperature rate  [K/s]
    Q_rad       = epsilon * SIGMA * (T_set_K.pow(4) - T_current_K.pow(4))  # [W/m²]
    dT_dt_rad   = Q_rad / (rho * Cp * thickness)                            # [K/s]

    # Residual: model rate vs radiation-driven rate
    scale = dT_dt_rad.abs().mean().clamp(min=1e-8)
    return ((dT_dt_pred - dT_dt_rad) / scale).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED PHYSICS-INFORMED LOSS
#
#   L_total = L_data + lambda * (w_cond*L_cond + w_conv*L_conv + w_rad*L_rad)
#
# lambda is updated each epoch by the training loop (curriculum schedule).
# ─────────────────────────────────────────────────────────────────────────────
class PhysicsInformedLoss(nn.Module):
    """
    Combined data + physics loss.

    Physics terms enforce:
      Conduction : Fourier's Law         rho*Cp*dT/dt = kappa*laplacian(T)
      Convection : Newton's Law          T_surface <= T_set
      Radiation  : Stefan-Boltzmann      dT/dt proportional to T_set^4 - T^4

    Args:
        lambda_physics : weight multiplying the total physics loss
        w_cond         : relative weight of conduction term
        w_conv         : relative weight of convection term
        w_rad          : relative weight of radiation term
        epsilon_steel  : emissivity of steel surface (default 0.8)
    """

    def __init__(
        self,
        lambda_physics: float = 0.001,
        w_cond:         float = 1.0,
        w_conv:         float = 0.5,
        w_rad:          float = 0.3,
        epsilon_steel:  float = 0.8,
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.w_cond         = w_cond
        self.w_conv         = w_conv
        self.w_rad          = w_rad
        self.epsilon        = epsilon_steel
        self._mse           = MaskedMSELoss()

    def forward(
        self,
        delta_T_pred:  torch.Tensor,  # (N,1) normalised model output
        target:        torch.Tensor,  # (N,1) normalised ground truth ΔT
        batch,                        # PyG Batch object
        Y_std:         float,
        dt:            float = 10.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, breakdown_dict).

        breakdown_dict keys: data, cond, conv, rad, physics
        All values are plain floats for printing/logging.
        """

        # ── Data loss (MSE against OpenFOAM ground truth) ─────────────
        L_data = self._mse(delta_T_pred, target)

        # If lambda is essentially zero, skip the expensive physics terms
        if self.lambda_physics < 1e-9:
            return L_data, {"data": L_data.item(),
                            "cond": 0.0, "conv": 0.0,
                            "rad":  0.0, "physics": 0.0}

        # ── Extract physical quantities from batch ─────────────────────
        # Node feature columns in GNN input (10 features):
        #  [0]x [1]y [2]z [3]T_now  [4]T_set
        #  [5]cy [6]cz [7]kappa [8]Cp [9]rho
        x = batch.x                                          # (N,10) normalised

        # De-normalise to physical units using stored node_mean / node_std
        if hasattr(batch, "node_mean") and batch.node_mean.numel() >= 10:
            nm = batch.node_mean.view(-1)
            ns = batch.node_std.view(-1)
            kappa_K  = x[:, 7] * ns[7] + nm[7]             # [W/m·K]
            Cp_K     = x[:, 8] * ns[8] + nm[8]             # [J/kg·K]
            rho_K    = x[:, 9] * ns[9] + nm[9]             # [kg/m³]
            T_set_K  = x[:, 4] * ns[4] + nm[4]             # [K]
        else:
            # Fallback to typical steel values from your parameter ranges
            kappa_K = torch.full((x.size(0),), 60.0,   device=x.device)
            Cp_K    = torch.full((x.size(0),), 450.0,  device=x.device)
            rho_K   = torch.full((x.size(0),), 7800.0, device=x.device)
            T_set_K = torch.full((x.size(0),), 1000.0, device=x.device)

        # Raw temperatures [K] stored in batch by dataset.py
        T_now  = batch.T_current.to(x.device)               # (N,) [K]
        T_next = T_now + delta_T_pred.squeeze(-1) * Y_std   # (N,) [K]

        # ── Physics loss 1: Conduction ─────────────────────────────────
        L_cond = conduction_loss(
            delta_T_pred_norm = delta_T_pred,
            T_current_K       = T_now,
            edge_index        = batch.edge_index,
            kappa             = kappa_K,
            rho               = rho_K,
            Cp                = Cp_K,
            dt                = dt,
            Y_std             = Y_std,
        )

        # ── Physics loss 2: Convection ─────────────────────────────────
        L_conv = convection_loss(
            T_pred_next_K = T_next,
            T_current_K   = T_now,
            T_set_K       = T_set_K,
        )

        # ── Physics loss 3: Radiation ──────────────────────────────────
        L_rad = radiation_loss(
            delta_T_pred_norm = delta_T_pred,
            T_current_K       = T_now,
            T_set_K           = T_set_K,
            rho               = rho_K,
            Cp                = Cp_K,
            dt                = dt,
            Y_std             = Y_std,
            epsilon           = self.epsilon,
        )

        # ── Combine ────────────────────────────────────────────────────
        L_physics = (self.w_cond * L_cond
                   + self.w_conv * L_conv
                   + self.w_rad  * L_rad)

        L_total = L_data + self.lambda_physics * L_physics

        breakdown = {
            "data":    L_data.item(),
            "cond":    L_cond.item(),
            "conv":    L_conv.item(),
            "rad":     L_rad.item(),
            "physics": L_physics.item(),
        }
        return L_total, breakdown