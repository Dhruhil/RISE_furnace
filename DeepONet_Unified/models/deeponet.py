"""
models/deeponet.py
------------------
Official NVIDIA PhysicsNeMo DeepONetArch wrapper.

Mirrors GNN_Unified/models/meshgraphnet.py — same interface,
different backbone (DeepONet instead of MeshGraphNet).

Architecture:
    Branch Net:  FullyConnectedArch  (BRANCH_INPUT_DIM → LATENT_DIM)
    Trunk Net:   FullyConnectedArch  (TRUNK_INPUT_DIM  → LATENT_DIM)
    DeepONetArch: dot-product of branch + trunk → T
"""

import torch
import torch.nn as nn

from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key

from configs.base_config import (
    BRANCH_INPUT_DIM, TRUNK_INPUT_DIM, OUTPUT_DIM,
    LATENT_DIM, LAYER_SIZE, NR_LAYERS,
)


class PhysicsNeMoDeepONet(nn.Module):
    """
    Official NVIDIA PhysicsNeMo DeepONetArch.

    Parameters
    ----------
    branch_input_dim : int   default BRANCH_INPUT_DIM (18)
    trunk_input_dim  : int   default TRUNK_INPUT_DIM  (8)
    latent_dim       : int   default LATENT_DIM       (256)
    layer_size       : int   default LAYER_SIZE       (512)
    nr_layers        : int   default NR_LAYERS        (6)
    """

    def __init__(
        self,
        branch_input_dim: int = BRANCH_INPUT_DIM,
        trunk_input_dim:  int = TRUNK_INPUT_DIM,
        latent_dim:       int = LATENT_DIM,
        layer_size:       int = LAYER_SIZE,
        nr_layers:        int = NR_LAYERS,
    ):
        super().__init__()

        # ── Branch Net ────────────────────────────────────────────────────
        # Encodes case-level parameters a → latent vector b
        self.branch_net = FullyConnectedArch(
            input_keys  = [Key("a",      size=branch_input_dim)],
            output_keys = [Key("branch", size=latent_dim)],
            layer_size  = layer_size,
            nr_layers   = nr_layers,
        )

        # ── Trunk Net ─────────────────────────────────────────────────────
        # Encodes query point (x,y,z,t,region) → latent vector t
        self.trunk_net = FullyConnectedArch(
            input_keys  = [Key("x",     size=trunk_input_dim)],
            output_keys = [Key("trunk", size=latent_dim)],
            layer_size  = layer_size,
            nr_layers   = nr_layers,
        )

        # ── Official PhysicsNeMo DeepONetArch ─────────────────────────────
        # Combines branch + trunk via inner product → output T
        self.deeponet = DeepONetArch(
            branch_net  = self.branch_net,
            trunk_net   = self.trunk_net,
            output_keys = [Key("T", size=OUTPUT_DIM)],
            branch_dim  = latent_dim,
            trunk_dim   = latent_dim,
        )

    def forward(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        a : Tensor (batch, BRANCH_INPUT_DIM)  case-level parameters
        x : Tensor (batch, TRUNK_INPUT_DIM)   query point + one-hot region

        Returns
        -------
        T : Tensor (batch, 1)  predicted normalized temperature
        """
        outvar = self.deeponet({"a": a, "x": x})
        return outvar["T"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
