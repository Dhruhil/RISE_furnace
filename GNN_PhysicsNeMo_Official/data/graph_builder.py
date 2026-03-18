"""
Graph builder for mesh-based GNN — no torch-cluster dependency.

Builds a k-NN graph from 3-D cell-centre coordinates using pure PyTorch
(cdist for pairwise distances). This works in the NVIDIA PhysicsNeMo
container without needing torch-cluster installed separately.

Edge features (4-D):
    [0] dx       = x_j - x_i
    [1] dy       = y_j - y_i
    [2] dz       = z_j - z_i
    [3] dist_norm = Euclidean distance / mean neighbour distance

This matches the MeshGraphNet paper (Pfaff et al., 2021) and NVIDIA
PhysicsNeMo's own graph construction approach.
"""

from __future__ import annotations

import torch
import numpy as np


def build_knn_graph(
    coords: torch.Tensor,   # (n_cells, 3)  float32
    k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a directed k-NN graph using pure PyTorch (no torch-cluster needed).

    For each node i, connect to its k nearest neighbours j.
    Self-loops are excluded.

    Args:
        coords : (n_cells, 3) cell-centre coordinates [m]
        k      : number of nearest neighbours per node

    Returns:
        edge_index : (2, n_edges)  LongTensor   [src, dst]
        edge_attr  : (n_edges, 4)  FloatTensor  [dx, dy, dz, dist_norm]
    """
    coords  = coords.float()
    n_cells = coords.shape[0]

    # Clamp k to avoid requesting more neighbours than nodes exist
    k_actual = min(k, n_cells - 1)

    # ── Pairwise squared distances via cdist ─────────────────────────
    # For large meshes (>5000 cells) this may be slow — see chunked
    # version below. For typical cylinder mesh (~450 cells) it is instant.
    with torch.no_grad():
        dist_matrix = torch.cdist(coords, coords, p=2)   # (n_cells, n_cells)

        # Set self-distance to infinity so self-loops are never selected
        dist_matrix.fill_diagonal_(float("inf"))

        # k nearest neighbours for each node
        _, knn_idx = dist_matrix.topk(k_actual, dim=1, largest=False)
        # knn_idx: (n_cells, k_actual) — indices of k nearest neighbours

    # ── Build edge_index [src, dst] ──────────────────────────────────
    # src[i*k + j] = i,  dst[i*k + j] = knn_idx[i, j]
    src = torch.arange(n_cells, device=coords.device) \
               .unsqueeze(1) \
               .expand(n_cells, k_actual) \
               .reshape(-1)              # (n_cells * k_actual,)
    dst = knn_idx.reshape(-1)           # (n_cells * k_actual,)
    edge_index = torch.stack([src, dst], dim=0).long()   # (2, n_edges)

    # ── Edge features ────────────────────────────────────────────────
    rel_pos   = coords[dst] - coords[src]              # (n_edges, 3)  [dx, dy, dz]
    dist      = rel_pos.norm(dim=-1, keepdim=True)     # (n_edges, 1)
    mean_dist = dist.mean().clamp(min=1e-8)
    dist_norm = dist / mean_dist                       # (n_edges, 1)  dimensionless

    edge_attr = torch.cat([rel_pos, dist_norm], dim=-1)  # (n_edges, 4)

    return edge_index, edge_attr


def build_knn_graph_chunked(
    coords: torch.Tensor,
    k: int = 16,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient k-NN graph builder for large meshes (>5000 cells).

    Computes pairwise distances in chunks to avoid OOM on large meshes.
    For typical cylinder mesh (~450 cells) use build_knn_graph() instead.
    """
    coords  = coords.float()
    n_cells = coords.shape[0]
    k_actual = min(k, n_cells - 1)

    all_src = []
    all_dst = []

    with torch.no_grad():
        for start in range(0, n_cells, chunk_size):
            end      = min(start + chunk_size, n_cells)
            chunk    = coords[start:end]                    # (chunk, 3)
            dists    = torch.cdist(chunk, coords, p=2)      # (chunk, n_cells)

            # Mask self-distances within this chunk
            for local_i in range(end - start):
                global_i = start + local_i
                dists[local_i, global_i] = float("inf")

            _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

            src_chunk = torch.arange(start, end, device=coords.device) \
                             .unsqueeze(1) \
                             .expand(end - start, k_actual) \
                             .reshape(-1)
            dst_chunk = knn_idx.reshape(-1)

            all_src.append(src_chunk)
            all_dst.append(dst_chunk)

    src = torch.cat(all_src)
    dst = torch.cat(all_dst)
    edge_index = torch.stack([src, dst], dim=0).long()

    rel_pos   = coords[dst] - coords[src]
    dist      = rel_pos.norm(dim=-1, keepdim=True)
    mean_dist = dist.mean().clamp(min=1e-8)
    dist_norm = dist / mean_dist
    edge_attr = torch.cat([rel_pos, dist_norm], dim=-1)

    return edge_index, edge_attr


def build_radius_graph(
    coords: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Radius-based graph — connects all cells within `radius` metres.
    Uses pure PyTorch, no torch-cluster needed.
    """
    coords  = coords.float()
    n_cells = coords.shape[0]

    with torch.no_grad():
        dist_matrix = torch.cdist(coords, coords, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        mask = dist_matrix <= radius

    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).long()

    rel_pos   = coords[dst] - coords[src]
    dist      = rel_pos.norm(dim=-1, keepdim=True)
    mean_dist = dist.mean().clamp(min=1e-8)
    dist_norm = dist / mean_dist
    edge_attr = torch.cat([rel_pos, dist_norm], dim=-1)

    return edge_index, edge_attr