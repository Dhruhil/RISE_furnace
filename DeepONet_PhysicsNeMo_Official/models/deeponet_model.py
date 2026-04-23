"""
Heat Treatment DeepONet — OFFICIAL NVIDIA PhysicsNeMo Sym DeepONetArch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key


class HeatTreatmentDeepONet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.branch_flat_dim = (cfg.branch_in_channels * cfg.n_sensors
                                + cfg.branch_scalar_inputs)
        self.branch = FullyConnectedArch(
            input_keys=[Key("a", size=self.branch_flat_dim)],
            output_keys=[Key("branch", size=cfg.latent_dim)],
            nr_layers=len(cfg.branch_hidden),
            layer_size=cfg.branch_hidden[0],
        )
        self.trunk = FullyConnectedArch(
            input_keys=[Key("x", size=cfg.trunk_in_features)],
            output_keys=[Key("trunk", size=cfg.latent_dim)],
            nr_layers=len(cfg.trunk_hidden),
            layer_size=cfg.trunk_hidden[0],
        )
        self.deeponet = DeepONetArch(
            output_keys=[Key("u", size=1)],
            branch_net=self.branch,
            trunk_net=self.trunk,
            branch_dim=cfg.latent_dim,
            trunk_dim=cfg.latent_dim,
        )
        print(f"[INFO] Using NVIDIA PhysicsNeMo Sym DeepONetArch "
              f"(params: {sum(p.numel() for p in self.parameters()):,})")

    def forward(self, branch_u, branch_scalars, trunk_y):
        B, C, N = branch_u.shape
        n_q = trunk_y.shape[1]
        branch_flat = branch_u.reshape(B, C * N)
        a = torch.cat([branch_flat, branch_scalars], dim=-1)
        a_rep = a.repeat_interleave(n_q, dim=0)
        x_flat = trunk_y.reshape(B * n_q, -1)
        out_dict = self.deeponet({"a": a_rep, "x": x_flat})
        u = out_dict["u"]
        return u.reshape(B, n_q)
