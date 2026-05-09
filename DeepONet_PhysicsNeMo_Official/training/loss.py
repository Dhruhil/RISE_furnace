"""
PI-DeepONet physics loss — full PDE residual computed at the
query points.

Energy balance:
    rho*Cp*dT/dt = kappa*∇²T + h*(T_set - T)/δ + ε*σ*(T_set⁴ - T⁴)/δ

Two design choices worth flagging up front:

  ∇²T via torch.autograd.grad
      The DeepONet predicts T at arbitrary continuous coordinates,
      so the spatial Laplacian falls out of automatic differentiation
      against the trunk's xyz input. This is the cleanest possible
      Laplacian — exact at every query point, no finite-difference
      stencil error, no grid spacing to choose. It's also the main
      reason this physics loss looks different from the FNO version,
      which has to use a 7-point stencil on the voxel grid.

  ∂T/∂t via finite difference
      Same convention as the FNO and GNN losses: (T_next - T_now)/dt.
      Keeping it identical means the time-residual term enters at the
      same scale across all three architectures, so the physics
      contribution is comparable in the thesis tables.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# Physical constants used in the residual terms below.
# CHAR_THICKNESS here is 0.01 m (10 mm), tuned for the cylinder/cavity
# scale — slightly different from the FNO config's 0.0167 m because
# the DeepONet residual operates at the cylinder surface scale rather
# than the voxel scale.
SIGMA_SB         = 5.67e-8
EMISSIVITY_STEEL = 0.80
H_CONV           = 25.0
CHAR_THICKNESS   = 0.01


def weighted_mse(pred, target, weight):
    """Per-query-point weighted MSE — same shape conventions as the FNO loss."""
    return ((pred - target).pow(2) * weight).sum() / (weight.sum() + 1e-8)


def spatial_laplacian(T_pred, xyz):
    """
    Compute ∇²T at every query point via two passes of autograd.

    First pass gives ∂T/∂x_i for i ∈ {x, y, z}; second pass gives
    ∂²T/∂x_i² by differentiating each component again. Summing the
    diagonals of the Hessian gives the Laplacian.

    create_graph=True is required because the Laplacian itself is
    going to be used inside a loss that gets backpropagated — so
    the gradient computation needs to stay in the graph.
    """
    # First derivative: gradient of the scalar T sum w.r.t. xyz.
    # Summing T_pred is the standard trick for getting a per-point
    # gradient out of autograd in one call.
    grad_T = torch.autograd.grad(
        outputs=T_pred.sum(), inputs=xyz,
        create_graph=True, retain_graph=True,
    )[0]

    # Second derivative: differentiate each component of grad_T
    # with respect to xyz again, then pluck out the matching
    # diagonal entry.
    lap = torch.zeros_like(T_pred)
    for i in range(3):
        grad2 = torch.autograd.grad(
            outputs=grad_T[..., i].sum(), inputs=xyz,
            create_graph=True, retain_graph=True,
        )[0]
        lap = lap + grad2[..., i]
    return lap


def pi_deeponet_physics_loss(
    T_pred_norm, T_pred_next_norm, xyz, T_cur_K, T_set,
    region_id, is_heater, kappa, Cp, rho,
    T_mean, T_std, dt=10.0,
):
    """
    Four-term physics-informed residual at the query points.

    Returns (L_phys, breakdown_dict) so the training loop can log
    each component separately — handy when tuning the physics weight
    or chasing down which term is dominating.

    Each residual is divided by its own characteristic scale so all
    four land in roughly the same numerical range (otherwise the
    radiation term would swamp everything else at 1100 °C).
    """
    # Bring predictions back into Kelvin so the residuals can be
    # written in plain SI units.
    T_pred_K = T_pred_norm      * T_std + T_mean
    T_next_K = T_pred_next_norm * T_std + T_mean
    non_heater = 1.0 - is_heater

    # Time derivative via finite difference — same scheme as the FNO
    # and GNN losses, so the time-residual scale is comparable.
    dT_dt = (T_next_K - T_pred_K) / dt

    # ---- 1. Conduction — Fourier ∇²T via autograd ------------------
    lap_T      = spatial_laplacian(T_pred_K, xyz)
    alpha      = kappa / (rho * Cp + 1e-8)
    dT_dt_cond = alpha * lap_T
    # /10 brings the residual into roughly [0, 1] range
    cond_res   = (dT_dt - dT_dt_cond) / 10.0
    L_cond     = ((cond_res * non_heater).pow(2)).mean()

    # ---- 2. Convection — Newton's cooling + overshoot guard --------
    T_set_b    = T_set.unsqueeze(-1)
    rho_cp     = rho * Cp + 1e-8
    dT_dt_conv = H_CONV * (T_set_b - T_pred_K) / (rho_cp * CHAR_THICKNESS)
    # /100 because dT_dt_conv tops out around 300 K/s
    conv_res   = (dT_dt - dT_dt_conv) / 100.0
    L_conv_match = ((conv_res * non_heater).pow(2)).mean()
    # Soft penalty against predicting T_next > T_set during heating —
    # physically impossible, but easy to slip into without the guard.
    overshoot    = F.relu(T_next_K - T_set_b) * non_heater
    L_overshoot  = (overshoot / T_set_b.clamp(min=300.0)).pow(2).mean()
    L_conv       = 0.5 * L_conv_match + 0.5 * L_overshoot

    # ---- 3. Radiation — Stefan-Boltzmann ---------------------------
    # T⁴ dependence dominates at 900-1100 °C, so this is the term
    # that needs the largest scale factor (/1000) to stay tame.
    dT_dt_rad = (EMISSIVITY_STEEL * SIGMA_SB *
                 (T_set_b.pow(4) - T_pred_K.pow(4)) / (rho_cp * CHAR_THICKNESS))
    rad_res = (dT_dt - dT_dt_rad) / 1000.0
    L_rad   = ((rad_res * non_heater).pow(2)).mean()

    # ---- 4. Energy balance — combined residual ---------------------
    # Predicted rate should match the sum of the three mechanism
    # rates. Normalising by T_std² keeps it scale-consistent with
    # the data MSE.
    dT_dt_total = dT_dt_cond + dT_dt_conv + dT_dt_rad
    eng_res     = (dT_dt - dT_dt_total)
    L_eng       = ((eng_res * non_heater).pow(2)).mean() / ((T_std ** 2) + 1e-8)

    # Weighted sum — same 0.4/0.3/0.2/0.1 split used by the GNN
    # final physics loss, so the three architectures can be compared
    # on equal terms when the physics weight is held constant.
    L_phys = 0.4 * L_cond + 0.3 * L_conv + 0.2 * L_rad + 0.1 * L_eng

    # Detached scalars for the training-loop logger
    breakdown = {
        "cond": float(L_cond.detach()), "conv": float(L_conv.detach()),
        "rad":  float(L_rad.detach()),  "overshoot": float(L_overshoot.detach()),
        "eng":  float(L_eng.detach()),
        "physics": float(L_phys.detach()),
    }
    return L_phys, breakdown


class DeepONetLoss(nn.Module):
    """
    Combined data + physics loss for the DeepONet training loop.

    The forward signature is wide because the physics terms need
    raw SI inputs (xyz, kappa, Cp, rho, ...) on top of the
    normalised tensors used by the data MSE. Calling code can pass
    None for the physics-only inputs to skip the physics residual
    entirely — useful for early-epoch warmup or for ablation runs.
    """

    def __init__(self, lambda_physics=0.003):
        super().__init__()
        # Gentle default weight, same idea as the FNO — the OpenFOAM
        # data already encodes the physics, so this term just nudges
        # the predictions toward energy-balance consistency.
        self.lambda_physics = lambda_physics

    def forward(self, pred_norm, target_norm, weight,
                T_set, T_mean, T_std,
                pred_next_norm=None, xyz=None, T_cur_K=None,
                region_id=None, is_heater=None,
                kappa=None, Cp=None, rho=None, dt=10.0):
        # Data MSE first — works whether or not the physics inputs
        # are present, so the early-warmup branch below can early-out.
        data = weighted_mse(pred_norm, target_norm, weight)

        # Physics residual is only computable when every required
        # input is present. The lambda check also lets ablation runs
        # set lambda_physics=0 to disable physics without changing
        # any other code path.
        has_phys = all(v is not None for v in [pred_next_norm, xyz, T_cur_K,
                        region_id, is_heater, kappa, Cp, rho])
        if not has_phys or self.lambda_physics < 1e-10:
            return data, {"data": float(data.detach()), "physics": 0.0,
                          "cond": 0.0, "conv": 0.0, "rad": 0.0, "overshoot": 0.0}

        L_phys, bd = pi_deeponet_physics_loss(
            pred_norm, pred_next_norm, xyz, T_cur_K, T_set,
            region_id, is_heater, kappa, Cp, rho,
            T_mean, T_std, dt)
        total = data + self.lambda_physics * L_phys
        return total, {"data": float(data.detach()), **bd}