"""
Build k-NN or radius graphs from 3D cell-center coordinates.
"""
from __future__ import annotations
import torch
from torch_geometric.nn import knn_graph, radius_graph


def build_knn_graph(
    coords: torch.Tensor,
    k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a k-NN graph from 3D coordinates.

    Args:
        coords: (n_nodes, 3) cell-center positions
        k:      number of nearest neighbours

    Returns:
        edge_index: (2, n_edges)
        edge_attr:  (n_edges, 4)  [dx, dy, dz, dist]
    """
    coords = coords.float()
    edge_index = knn_graph(coords, k=k, loop=False)
    row, col = edge_index
    diff = coords[col] - coords[row]
    dist = diff.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([diff, dist], dim=-1)
    return edge_index, edge_attr


def build_radius_graph(
    coords: torch.Tensor,
    r: float = 0.03,
    max_num_neighbors: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a radius graph from 3D coordinates."""
    coords = coords.float()
    edge_index = radius_graph(coords, r=r, loop=False, max_num_neighbors=max_num_neighbors)
    row, col = edge_index
    diff = coords[col] - coords[row]
    dist = diff.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([diff, dist], dim=-1)
    return edge_index, edge_attr