"""
DeepONet model for the heat-treatment surrogate.

Wraps NVIDIA PhysicsNeMo Sym's DeepONetArch with the project-
specific input shapes and feature counts. Two sub-networks make
the operator-learning structure work:

  branch — encodes the current temperature field into a latent
           vector. Sees a flat concatenation of the sensor-lattice
           field plus the per-sim scalar inputs (T_set, time, and
           cylinder geometry).
  trunk  — encodes each query coordinate plus its per-cell static
           channels into the same latent space.

The DeepONet's predictor then takes the inner product of the two
latents to produce a scalar prediction (normalised T_next) at
every query point.

Training and rollout scripts call forward(branch_u, branch_scalars,
trunk_y) directly — the unwrapping into PhysicsNeMo's dict-based
interface happens inside this class so the rest of the pipeline
doesn't have to know about it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key


class HeatTreatmentDeepONet(nn.Module):
    """
    Project-specific DeepONet wrapper.

    Branch and trunk are both FullyConnectedArch MLPs that project
    into the same cfg.latent_dim, which is what lets the final
    DeepONet inner product produce a scalar at each query point.

    Layer counts come from cfg.branch_hidden / cfg.trunk_hidden.
    layer_size uses the first entry of each list; PhysicsNeMo's
    FullyConnectedArch only takes a single layer_size, so the
    config keeps a list mostly for documentation / readability.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Total branch input dimension after flattening:
        #   per-sensor channels * number of sensors + per-sim scalars
        # E.g. 6 channels * 2160 sensors + 7 scalars = 12967 features.
        self.branch_flat_dim = (cfg.branch_in_channels * cfg.n_sensors
                                + cfg.branch_scalar_inputs)

        # ---- branch net: function-encoding MLP -------------------------
        # PhysicsNeMo's FullyConnectedArch is keyed: "a" is the
        # branch input symbol and "branch" is the output. The keys
        # have to match the dict passed to self.deeponet(...) below.
        self.branch = FullyConnectedArch(
            input_keys=[Key("a", size=self.branch_flat_dim)],
            output_keys=[Key("branch", size=cfg.latent_dim)],
            nr_layers=len(cfg.branch_hidden),
            layer_size=cfg.branch_hidden[0],
        )

        # ---- trunk net: per-query-point coordinate MLP -----------------
        # "x" is the input key (per-query-point feature vector),
        # "trunk" is the output. Slightly deeper than the branch by
        # default since the trunk has to learn a richer per-point
        # basis across the full furnace volume.
        self.trunk = FullyConnectedArch(
            input_keys=[Key("x", size=cfg.trunk_in_features)],
            output_keys=[Key("trunk", size=cfg.latent_dim)],
            nr_layers=len(cfg.trunk_hidden),
            layer_size=cfg.trunk_hidden[0],
        )

        # ---- final DeepONet predictor ---------------------------------
        # Combines branch and trunk via inner product into a scalar
        # output keyed "u". branch_dim and trunk_dim must agree
        # with cfg.latent_dim — they're what makes the dot product
        # well-defined.
        self.deeponet = DeepONetArch(
            output_keys=[Key("u", size=1)],
            branch_net=self.branch,
            trunk_net=self.trunk,
            branch_dim=cfg.latent_dim,
            trunk_dim=cfg.latent_dim,
        )

        # Quick parameter count printout — useful sanity check after
        # changing latent_dim, branch_hidden, or trunk_hidden.
        print(f"[INFO] Using NVIDIA PhysicsNeMo Sym DeepONetArch "
              f"(params: {sum(p.numel() for p in self.parameters()):,})")

    def forward(self, branch_u, branch_scalars, trunk_y):
        """
        Run a forward pass on a batch of (branch input, query points).

        Parameters
        ----------
        branch_u : (B, C, N) — per-batch sensor-lattice field
            B = batch size, C = branch_in_channels, N = n_sensors
        branch_scalars : (B, branch_scalar_inputs)
            Per-sim scalar inputs (T_set, time, cylinder geometry).
        trunk_y : (B, n_q, trunk_in_features)
            Query coordinates + per-cell static channels.

        Returns
        -------
        torch.Tensor of shape (B, n_q) — normalised T_next at every
        query point of every sample.
        """
        B, C, N = branch_u.shape
        n_q = trunk_y.shape[1]

        # Flatten sensor field and concatenate the per-sim scalars
        # into one fat branch input vector per sample.
        branch_flat = branch_u.reshape(B, C * N)
        a = torch.cat([branch_flat, branch_scalars], dim=-1)

        # PhysicsNeMo's DeepONetArch wants the branch input replicated
        # once per query point (every query point sees the same branch
        # encoding for its sample). repeat_interleave is the cheapest
        # way to broadcast that without an explicit loop.
        a_rep = a.repeat_interleave(n_q, dim=0)

        # Trunk gets every (sample, query) pair flattened into a
        # single batch dimension. Reshape afterwards to put the
        # query axis back.
        x_flat = trunk_y.reshape(B * n_q, -1)

        out_dict = self.deeponet({"a": a_rep, "x": x_flat})
        u = out_dict["u"]
        return u.reshape(B, n_q)