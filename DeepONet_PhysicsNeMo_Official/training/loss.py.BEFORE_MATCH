"""
PI-DeepONet physics loss — full PDE residual with autograd.
rho*Cp*dT/dt = kappa*∇²T + h*(T_set-T)/δ + ε*σ*(T_set⁴-T⁴)/δ

∇²T via torch.autograd.grad (exact, spatial)
∂T/∂t via finite difference (matches FNO convention)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_SB         = 5.67e-8
EMISSIVITY_STEEL = 0.80
H_CONV           = 25.0
CHAR_THICKNESS   = 0.01


def weighted_mse(pred, target, weight):
    return ((pred - target).pow(2) * weight).sum() / (weight.sum() + 1e-8)


def spatial_laplacian(T_pred, xyz):
    grad_T = torch.autograd.grad(
        outputs=T_pred.sum(), inputs=xyz,
        create_graph=True, retain_graph=True,
    )[0]
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
    T_pred_K = T_pred_norm      * T_std + T_mean
    T_next_K = T_pred_next_norm * T_std + T_mean
    non_heater = 1.0 - is_heater

    dT_dt = (T_next_K - T_pred_K) / dt

    lap_T      = spatial_laplacian(T_pred_K, xyz)
    alpha      = kappa / (rho * Cp + 1e-8)
    dT_dt_cond = alpha * lap_T
    cond_res   = (dT_dt - dT_dt_cond) / 10.0
    L_cond     = ((cond_res * non_heater).pow(2)).mean()

    T_set_b    = T_set.unsqueeze(-1)
    rho_cp     = rho * Cp + 1e-8
    dT_dt_conv = H_CONV * (T_set_b - T_pred_K) / (rho_cp * CHAR_THICKNESS)
    conv_res   = (dT_dt - dT_dt_conv) / 100.0
    L_conv_match = ((conv_res * non_heater).pow(2)).mean()
    overshoot    = F.relu(T_next_K - T_set_b) * non_heater
    L_overshoot  = (overshoot / T_set_b.clamp(min=300.0)).pow(2).mean()
    L_conv       = 0.5 * L_conv_match + 0.5 * L_overshoot

    dT_dt_rad = (EMISSIVITY_STEEL * SIGMA_SB *
                 (T_set_b.pow(4) - T_pred_K.pow(4)) / (rho_cp * CHAR_THICKNESS))
    rad_res = (dT_dt - dT_dt_rad) / 1000.0
    L_rad   = ((rad_res * non_heater).pow(2)).mean()

    L_phys = 0.5 * L_conv + 0.3 * L_cond + 0.2 * L_rad
    breakdown = {
        "cond": float(L_cond.detach()), "conv": float(L_conv.detach()),
        "rad":  float(L_rad.detach()),  "overshoot": float(L_overshoot.detach()),
        "physics": float(L_phys.detach()),
    }
    return L_phys, breakdown


class DeepONetLoss(nn.Module):
    def __init__(self, lambda_physics=0.003):
        super().__init__()
        self.lambda_physics = lambda_physics

    def forward(self, pred_norm, target_norm, weight,
                T_set, T_mean, T_std,
                pred_next_norm=None, xyz=None, T_cur_K=None,
                region_id=None, is_heater=None,
                kappa=None, Cp=None, rho=None, dt=10.0):
        data = weighted_mse(pred_norm, target_norm, weight)
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
