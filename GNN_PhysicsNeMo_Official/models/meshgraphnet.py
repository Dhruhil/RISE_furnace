"""
MeshGraphNet for the unified multi-region heat-treatment GNN.

Note: aggregation happens on dst (the receiver), not src (the sender).
The earlier version had this flipped, which made messages flow the
wrong way and basically killed training.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Batch

# Try to use the official PhysicsNeMo MeshGraphNet first, since it's
# the implementation referenced in the thesis. Fall back to a hand-
# rolled version if PhysicsNeMo isn't available — handy for local
# debugging on machines without the framework installed.
try:
    from physicsnemo.models.meshgraphnet import MeshGraphNet as _MGN
    PHYSICSNEMO_AVAILABLE = True
except ImportError:
    PHYSICSNEMO_AVAILABLE = False
    print("[INFO] physicsnemo not found — using fallback.")

from configs.base_config import BaseConfig


class _MLP(nn.Module):
    """
    Small MLP with LayerNorm at the end. ReLU between hidden layers,
    no activation after the final norm — this matches the standard
    MeshGraphNet block design from Pfaff et al. 2021.
    """
    def __init__(self, in_dim, out_dim, hidden, n_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.net(x))


class _MPBlock(nn.Module):
    """
    One round of message passing.

    Edge update: combines source node, destination node, and the
    current edge embedding into a new edge feature.
    Node update: each receiver aggregates the messages on its
    incoming edges (sum) and updates its own state with a residual.
    """
    def __init__(self, hidden):
        super().__init__()
        # Edge MLP takes [h_src, h_dst, h_edge] -> new edge embedding
        self.edge_mlp = _MLP(3 * hidden, hidden, hidden)
        # Node MLP takes [h_node, aggregated_messages] -> new node embedding
        self.node_mlp = _MLP(2 * hidden, hidden, hidden)

    def forward(self, h_n, h_e, edge_index):
        src, dst = edge_index[0], edge_index[1]
        N = h_n.shape[0]

        # Update edges first — residual update like in the original MGN paper
        h_e = h_e + self.edge_mlp(torch.cat([h_n[src], h_n[dst], h_e], dim=-1))

        # Aggregate at dst (the receiver), not src. Aggregating at the
        # sender would push messages outward instead of pooling them
        # inward, and the model just won't learn anything useful.
        agg = torch.zeros(N, h_e.shape[-1], device=h_n.device, dtype=h_n.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(h_e), h_e)

        # Residual node update
        h_n = h_n + self.node_mlp(torch.cat([h_n, agg], dim=-1))
        return h_n, h_e


class _FallbackMGN(nn.Module):
    """
    Stand-in MeshGraphNet implementation for environments where
    PhysicsNeMo isn't installed. Same encode-process-decode pattern
    and the same number of message-passing layers, so results should
    match the official version up to numerical noise.
    """
    def __init__(self, node_in, edge_in, hidden, n_layers, out):
        super().__init__()
        # Encode raw node and edge features into the hidden dimension
        self.node_encoder = _MLP(node_in, hidden, hidden)
        self.edge_encoder = _MLP(edge_in, hidden, hidden)
        # Stack of message-passing blocks (the "process" part)
        self.mp_blocks    = nn.ModuleList([_MPBlock(hidden) for _ in range(n_layers)])
        # Per-node decoder back to the output dimension (1 = temperature delta)
        self.decoder      = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out)
        )

    def forward(self, x, edge_index, edge_attr):
        h_n = self.node_encoder(x)
        h_e = self.edge_encoder(edge_attr)
        for block in self.mp_blocks:
            h_n, h_e = block(h_n, h_e, edge_index)
        return self.decoder(h_n)


class HeatTreatmentGNN(nn.Module):
    """
    Wrapper that picks PhysicsNeMo's MeshGraphNet when available, or
    falls back to the hand-rolled version otherwise. Either way the
    public interface (forward / predict_delta_T / save / load) stays
    identical, so the training and rollout scripts don't care which
    backend is running underneath.
    """
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_features

        if PHYSICSNEMO_AVAILABLE:
            # Official PhysicsNeMo implementation — this is what's
            # used for the numbers reported in the thesis.
            self.gnn = _MGN(
                input_dim_nodes=cfg.node_in_features,
                input_dim_edges=cfg.edge_in_features,
                output_dim=cfg.output_features,
                processor_size=cfg.n_message_passing_layers,
                mlp_activation_fn="relu",
                num_layers_node_processor=2, num_layers_edge_processor=2,
                hidden_dim_processor=h,
                hidden_dim_node_encoder=h, num_layers_node_encoder=2,
                hidden_dim_edge_encoder=h, num_layers_edge_encoder=2,
                hidden_dim_node_decoder=h, num_layers_node_decoder=2,
                aggregation="sum",
            )
            self._backend = "physicsnemo"
        else:
            # Fallback path — keeps things working on machines
            # without PhysicsNeMo (e.g., a quick laptop debug run).
            self.gnn = _FallbackMGN(
                cfg.node_in_features, cfg.edge_in_features,
                h, cfg.n_message_passing_layers, cfg.output_features,
            )
            self._backend = "fallback"

        # Quick parameter count printout — useful sanity check after
        # changing hidden_features or n_message_passing_layers
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  HeatTreatmentGNN [{self._backend}]")
        print(f"    hidden={h}  layers={cfg.n_message_passing_layers}  "
              f"node_in={cfg.node_in_features}  edge_in={cfg.edge_in_features}")
        print(f"    Trainable parameters: {n_params:,}")

    def forward(self, batch: Batch) -> torch.Tensor:
        if self._backend == "physicsnemo":
            # PhysicsNeMo expects a DGL graph, while the dataset
            # produces PyG-style edge_index tensors — convert here.
            # Going through CPU first avoids a couple of edge cases
            # in dgl.graph when src/dst sit on GPU memory.
            import dgl
            n_nodes = batch.x.shape[0]
            src = batch.edge_index[0].cpu()
            dst = batch.edge_index[1].cpu()
            g = dgl.graph((src, dst), num_nodes=n_nodes).to(batch.x.device)
            out = self.gnn(batch.x, batch.edge_attr, g)
        else:
            out = self.gnn(batch.x, batch.edge_index, batch.edge_attr)

        # Make sure the output is always [N, 1] even if the backend
        # squeezed it down to [N] for a single-channel output.
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out

    def predict_delta_T(self, batch):
        # Thin alias kept around for readability in the training
        # loop — semantically the model predicts a normalised dT.
        return self.forward(batch)

    def save(self, path, epoch, optimizer_state=None, scheduler_state=None, metrics=None):
        """
        Save a full checkpoint: model weights, optimiser & scheduler
        state, the metrics for this epoch, and the model config.

        The model_cfg dict gets bundled too so a checkpoint can be
        reloaded later without needing the exact original cfg file.
        """
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch, "model_state": self.state_dict(),
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "metrics": metrics or {}, "backend": self._backend,
            "model_cfg": {
                "node_in_features": self.cfg.node_in_features,
                "edge_in_features": self.cfg.edge_in_features,
                "hidden_features": self.cfg.hidden_features,
                "n_message_passing_layers": self.cfg.n_message_passing_layers,
                "output_features": self.cfg.output_features,
            },
        }, path)
        mae_str = f"  val_MAE={metrics['mae']:.3f} K" if metrics and "mae" in metrics else ""
        print(f"  Checkpoint saved -> {path}  (epoch {epoch}){mae_str}")

    @classmethod
    def load(cls, path, cfg, device="cpu"):
        """
        Load a checkpoint produced by .save(). The cfg passed in here
        should match the one used at training time — the saved
        model_cfg block is mostly there for cross-checks and audits.
        """
        ckpt = torch.load(path, map_location=device)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        metrics = ckpt.get("metrics", {})
        mae_str = f"  val_MAE={metrics['mae']:.3f} K" if "mae" in metrics else ""
        print(f"  Checkpoint loaded <- {path}  (epoch {ckpt.get('epoch','?')}){mae_str}")
        return model