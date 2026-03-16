"""
MeshGraphNet wrapper using NVIDIA PhysicsNeMo official implementation.

PhysicsNeMo 25.06 provides:
    physicsnemo.models.meshgraphnet.MeshGraphNet

This wrapper adds:
  - convenience forward pass returning delta_T
  - autoregressive rollout helper
  - checkpoint save/load
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch

try:
    from physicsnemo.models.meshgraphnet import MeshGraphNet as _MGN
    PHYSICSNEMO_AVAILABLE = True
except ImportError:
    PHYSICSNEMO_AVAILABLE = False
    print("[WARNING] physicsnemo not found — using fallback MeshGraphNet implementation.")

from configs.base_config import BaseConfig


# ---------------------------------------------------------------------------
# Fallback: lightweight MeshGraphNet when PhysicsNeMo is not installed
# ---------------------------------------------------------------------------
class _MLPBlock(nn.Module):
    def __init__(self, in_f: int, out_f: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_f),
        )
        self.norm = nn.LayerNorm(out_f)

    def forward(self, x):
        return self.norm(self.net(x))


class _FallbackMGN(nn.Module):
    """Minimal MeshGraphNet: encode → N message-passing steps → decode."""

    def __init__(self, node_in: int, edge_in: int,
                 hidden: int, n_layers: int, out: int):
        super().__init__()
        self.node_encoder = _MLPBlock(node_in, hidden)
        self.edge_encoder = _MLPBlock(edge_in, hidden)

        self.edge_mlps = nn.ModuleList([_MLPBlock(3*hidden, hidden) for _ in range(n_layers)])
        self.node_mlps = nn.ModuleList([_MLPBlock(2*hidden, hidden) for _ in range(n_layers)])

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x, edge_index, edge_attr):
        h_n = self.node_encoder(x)
        h_e = self.edge_encoder(edge_attr)
        row, col = edge_index

        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            # Edge update
            msg = torch.cat([h_n[row], h_n[col], h_e], dim=-1)
            h_e = h_e + edge_mlp(msg)
            # Node update (sum aggregation)
            agg = torch.zeros_like(h_n).scatter_add(
                0, row.unsqueeze(-1).expand_as(h_e), h_e)
            h_n = h_n + node_mlp(torch.cat([h_n, agg], dim=-1))

        return self.decoder(h_n)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
class HeatTreatmentGNN(nn.Module):
    """
    Graph Neural Network surrogate for heat treatment temperature prediction.

    Predicts delta_T (normalised) given the current state graph.
    Autoregressive rollout is handled externally in rollout.py.
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg

        if PHYSICSNEMO_AVAILABLE:
            # Use official PhysicsNeMo MeshGraphNet
            self.gnn = _MGN(
                input_node_dim    = cfg.node_in_features,
                input_edge_dim    = cfg.edge_in_features,
                output_node_dim   = cfg.output_features,
                hidden_dim        = cfg.hidden_features,
                num_message_passing_layers = cfg.n_message_passing_layers,
            )
            self._backend = "physicsnemo"
        else:
            # Fallback lightweight GNN
            self.gnn = _FallbackMGN(
                node_in  = cfg.node_in_features,
                edge_in  = cfg.edge_in_features,
                hidden   = cfg.hidden_features,
                n_layers = cfg.n_message_passing_layers,
                out      = cfg.output_features,
            )
            self._backend = "fallback"

        n_params = sum(p.numel() for p in self.parameters())
        print(f"HeatTreatmentGNN [{self._backend}] | Parameters: {n_params:,}")

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch with x, edge_index, edge_attr

        Returns:
            delta_T_norm: (n_nodes_total, 1)
        """
        if PHYSICSNEMO_AVAILABLE:
            return self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        else:
            return self.gnn(batch.x, batch.edge_index, batch.edge_attr)

    def predict_delta_T(self, batch: Batch) -> torch.Tensor:
        """Same as forward — explicit name for clarity."""
        return self.forward(batch)

    # ------------------------------------------------------------------
    def save(self, path: str, epoch: int, optimizer_state: dict | None = None,
             metrics: dict | None = None) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch":           epoch,
            "model_state":     self.state_dict(),
            "optimizer_state": optimizer_state,
            "metrics":         metrics or {},
            "backend":         self._backend,
            "cfg": {
                "node_in_features":        self.cfg.node_in_features,
                "edge_in_features":        self.cfg.edge_in_features,
                "hidden_features":         self.cfg.hidden_features,
                "n_message_passing_layers": self.cfg.n_message_passing_layers,
                "output_features":         self.cfg.output_features,
            },
        }
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved → {path}  (epoch {epoch})")

    @classmethod
    def load(cls, path: str, cfg: BaseConfig,
             device: str = "cpu") -> "HeatTreatmentGNN":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        print(f"  Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
        return model