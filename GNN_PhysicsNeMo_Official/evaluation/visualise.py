"""
Visualisation of predicted vs ground-truth temperature fields.

Produces:
  - Parity plots (predicted vs true T at each time step)
  - Spatial temperature field maps (projected to y-z plane)
  - Time series at selected cell locations
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path


def plot_parity(T_pred: np.ndarray, T_true: np.ndarray,
                title: str = "", save_path: str | None = None):
    """Scatter plot of predicted vs true temperature."""
    T_pred_flat = T_pred.ravel()
    T_true_flat = T_true.ravel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(T_true_flat, T_pred_flat, s=1, alpha=0.2, color="steelblue")
    lim = [min(T_true_flat.min(), T_pred_flat.min()),
           max(T_true_flat.max(), T_pred_flat.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    mae = np.mean(np.abs(T_pred_flat - T_true_flat))
    ax.set_xlabel("OpenFOAM T [K]")
    ax.set_ylabel("GNN T [K]")
    ax.set_title(f"{title} | MAE = {mae:.2f} K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_temperature_field(
    coords: np.ndarray,      # (n_cells, 3)
    T: np.ndarray,            # (n_cells,)
    T_ref: np.ndarray | None = None,
    title: str = "",
    save_path: str | None = None,
    projection: str = "yz",   # "yz", "xz", "xy"
):
    """
    2D scatter plot of the temperature field projected onto a plane.

    Args:
        coords:     Cell center coordinates (n_cells, 3)
        T:          Temperature array (n_cells,)
        T_ref:      Reference (ground truth) for residual plot
        title:      Plot title
        save_path:  Where to save the figure
        projection: Which plane to project onto
    """
    axis_map = {"yz": (1, 2), "xz": (0, 2), "xy": (0, 1)}
    i1, i2 = axis_map[projection]
    axis_labels = {0: "x [m]", 1: "y [m]", 2: "z [m]"}

    n_cols = 3 if T_ref is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    def scatter_panel(ax, values, label, cmap="hot"):
        sc = ax.scatter(coords[:, i1], coords[:, i2],
                        c=values, cmap=cmap, s=10, vmin=values.min(), vmax=values.max())
        plt.colorbar(sc, ax=ax, label="T [K]")
        ax.set_xlabel(axis_labels[i1])
        ax.set_ylabel(axis_labels[i2])
        ax.set_title(label)
        ax.set_aspect("equal")

    scatter_panel(axes[0], T, f"GNN Prediction\n{title}")
    if T_ref is not None:
        scatter_panel(axes[1], T_ref, "OpenFOAM Ground Truth")
        scatter_panel(axes[2], np.abs(T - T_ref), "|Error| [K]", cmap="Reds")
    else:
        scatter_panel(axes[1], np.abs(T - T), "|Error|", cmap="Reds")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_time_series_at_cells(
    T_pred: np.ndarray,    # (n_steps, n_cells)
    T_true: np.ndarray,    # (n_steps, n_cells)
    times: np.ndarray,     # (n_steps,)
    cell_indices: list[int] = None,
    n_cells_show: int = 5,
    title: str = "",
    save_path: str | None = None,
):
    """Plot predicted vs true T time series at selected cells."""
    n_cells = T_pred.shape[1]
    if cell_indices is None:
        rng = np.random.default_rng(42)
        cell_indices = rng.choice(n_cells, size=min(n_cells_show, n_cells), replace=False)

    fig, axes = plt.subplots(len(cell_indices), 1,
                              figsize=(10, 2.5 * len(cell_indices)), sharex=True)
    if len(cell_indices) == 1:
        axes = [axes]

    for ax, ci in zip(axes, cell_indices):
        ax.plot(times, T_true[:, ci], "k-", lw=1.5, label="OpenFOAM")
        ax.plot(times, T_pred[:, ci], "r--", lw=1.5, label="GNN")
        mae_i = np.mean(np.abs(T_pred[:, ci] - T_true[:, ci]))
        ax.set_ylabel("T [K]")
        ax.set_title(f"Cell {ci} | MAE = {mae_i:.2f} K")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)