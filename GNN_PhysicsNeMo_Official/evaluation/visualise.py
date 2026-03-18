"""
Visualisation utilities for heat treatment GNN evaluation.

All functions save figures to disk (no display — designed for headless servers).

Functions:
    plot_parity(...)              → predicted vs true scatter plot
    plot_temperature_field(...)   → 2-D scatter of T values on mesh
    plot_time_series_at_cells(...)→ T(t) curves at selected cells
    plot_error_map(...)           → spatial distribution of error at one time
    plot_rollout_summary(...)     → MAE/RMSE over time + mean T evolution
    plot_future_prediction(...)   → full timeline incl. extrapolation zone
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Parity plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity(
    T_pred:    np.ndarray,   # (n_steps, n_nodes) or (N,)
    T_true:    np.ndarray,
    title:     str  = "",
    save_path: str | None = None,
) -> None:
    """
    Parity (predicted vs true) scatter plot.
    Accepts either flat arrays or (n_steps, n_nodes) rollout arrays.
    """
    p = np.asarray(T_pred).ravel()
    t = np.asarray(T_true).ravel()

    # Sub-sample if very large
    if len(p) > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(p), size=200_000, replace=False)
        p, t = p[idx], t[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, s=0.5, alpha=0.2, color="steelblue", rasterized=True)

    lim_min = min(t.min(), p.min())
    lim_max = max(t.max(), p.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="Perfect")

    # Error bands
    for band, color in [(5, "orange"), (10, "gold")]:
        ax.fill_between(
            [lim_min, lim_max],
            [lim_min - band, lim_max - band],
            [lim_min + band, lim_max + band],
            alpha=0.10, color=color, label=f"±{band} K band",
        )

    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r2   = float(1 - np.sum((p - t) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-8))

    ax.set_xlabel("OpenFOAM T [K]", fontsize=11)
    ax.set_ylabel("GNN T [K]",      fontsize=11)
    ax.set_title(f"{title}\nMAE={mae:.2f} K  RMSE={rmse:.2f} K  R²={r2:.4f}",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature field (spatial scatter)
# ─────────────────────────────────────────────────────────────────────────────

def plot_temperature_field(
    coords:     np.ndarray,         # (n_cells, 3)
    T:          np.ndarray,         # (n_cells,) predicted
    T_ref:      np.ndarray | None,  # (n_cells,) ground truth, or None
    title:      str  = "",
    save_path:  str | None = None,
    projection: str  = "yz",        # "xy", "xz", or "yz"
) -> None:
    """
    Scatter plot of temperature field on the mesh cross-section.
    Shows prediction, ground truth, and absolute error side by side.
    """
    axis_map   = {"yz": (1, 2), "xz": (0, 2), "xy": (0, 1)}
    axis_label = {0: "x [m]", 1: "y [m]", 2: "z [m]"}
    i1, i2    = axis_map.get(projection, (1, 2))

    n_cols = 3 if T_ref is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    vmin = T.min() if T_ref is None else min(T.min(), T_ref.min())
    vmax = T.max() if T_ref is None else max(T.max(), T_ref.max())

    def _panel(ax, vals, label, cmap="hot", vmin_=None, vmax_=None):
        sc = ax.scatter(
            coords[:, i1], coords[:, i2],
            c=vals, cmap=cmap, s=8,
            vmin=vmin_ if vmin_ is not None else vals.min(),
            vmax=vmax_ if vmax_ is not None else vals.max(),
        )
        plt.colorbar(sc, ax=ax, label="T [K]" if "error" not in label.lower() else "|ΔT| [K]")
        ax.set_xlabel(axis_label[i1])
        ax.set_ylabel(axis_label[i2])
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")

    _panel(axes[0], T, f"GNN Prediction\n{title}", vmin_=vmin, vmax_=vmax)
    if T_ref is not None:
        _panel(axes[1], T_ref,         "OpenFOAM Ground Truth", vmin_=vmin, vmax_=vmax)
        _panel(axes[2], np.abs(T - T_ref), "|Error| [K]", cmap="Reds")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Time series at individual cells
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series_at_cells(
    T_pred:       np.ndarray,             # (n_steps, n_cells)
    T_true:       np.ndarray,             # (n_steps, n_cells)
    times:        np.ndarray,             # (n_steps,) in seconds
    cell_indices: list[int] | None = None,
    n_cells_show: int               = 5,
    title:        str               = "",
    save_path:    str | None        = None,
    t_end_train:  float | None      = None,  # mark end of training window
) -> None:
    """
    Plot predicted vs true temperature time series at selected cells.
    Optionally marks the end of the training window (for future prediction).
    """
    n_cells = T_pred.shape[1]
    if cell_indices is None:
        rng          = np.random.default_rng(42)
        cell_indices = list(
            rng.choice(n_cells, size=min(n_cells_show, n_cells), replace=False)
        )

    n_show = len(cell_indices)
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for ax, ci in zip(axes, cell_indices):
        ax.plot(times, T_true[:, ci], "k-",  lw=1.5, label="OpenFOAM", zorder=3)
        ax.plot(times, T_pred[:, ci], "r--", lw=1.5, label="GNN",      zorder=2)

        mae_i = float(np.mean(np.abs(T_pred[:, ci] - T_true[:, ci])))
        ax.set_ylabel("T [K]", fontsize=9)
        ax.set_title(f"Cell {ci}  |  MAE = {mae_i:.2f} K", fontsize=9)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3)

        if t_end_train is not None:
            ax.axvline(t_end_train, color="orange", ls="--", lw=1.2, alpha=0.7,
                       label=f"Train end {t_end_train:.0f}s")

    axes[-1].set_xlabel("Time [s]", fontsize=10)
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout summary (MAE/RMSE + mean T evolution)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rollout_summary(
    T_pred:      np.ndarray,     # (n_steps+1, n_cells) — may be longer than T_true
    T_true:      np.ndarray,     # (n_gt+1,   n_cells)
    cfg,
    sim_idx:     int  = 0,
    save_path:   str | None = None,
) -> None:
    """
    Two-panel figure:
      Left:  MAE and RMSE over time (within ground-truth window)
      Right: Mean temperature evolution (GNN vs OpenFOAM + future zone)
    """
    n_gt = T_true.shape[0]
    times_gt   = np.arange(n_gt) * cfg.dt
    times_pred = np.arange(T_pred.shape[0]) * cfg.dt

    from utils.metrics import metrics_per_timestep
    step_m    = metrics_per_timestep(T_pred[:n_gt], T_true)
    step_mae  = np.array([m["mae"]  for m in step_m])
    step_rmse = np.array([m["rmse"] for m in step_m])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: error over time
    ax1.plot(times_gt, step_mae,  color="steelblue", lw=1.5, label="MAE [K]")
    ax1.plot(times_gt, step_rmse, color="tomato",    lw=1.5, ls="--", label="RMSE [K]")
    ax1.axvline(cfg.t_end, color="orange", ls="--", lw=1.2,
                label=f"Training end ({cfg.t_end:.0f}s)")
    ax1.set_xlabel("Time [s]");  ax1.set_ylabel("Error [K]")
    ax1.set_title(f"Rollout error — Sim {sim_idx}")
    ax1.legend(fontsize=9);      ax1.grid(True, alpha=0.3)

    # Right: mean T evolution
    ax2.plot(times_gt,          T_true.mean(axis=1),
             "k-",  lw=2, label="OpenFOAM mean T")
    ax2.plot(times_pred[:n_gt], T_pred[:n_gt].mean(axis=1),
             "r--", lw=2, label="GNN mean T (GT window)")

    if T_pred.shape[0] > n_gt:
        ax2.plot(times_pred[n_gt:], T_pred[n_gt:].mean(axis=1),
                 "m--", lw=2, label="GNN mean T (FUTURE)")
        ax2.axvspan(cfg.t_end, times_pred[-1], alpha=0.07, color="purple",
                    label="Future extrapolation zone")
        ax2.axvline(cfg.t_end, color="orange", ls="--", lw=1.2)

    ax2.set_xlabel("Time [s]");  ax2.set_ylabel("Mean T [K]")
    ax2.set_title(f"Temperature evolution — Sim {sim_idx}")
    ax2.legend(fontsize=9);      ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Future prediction summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_future_prediction(
    T_pred:      np.ndarray,   # (n_steps_total+1, n_cells)
    T_true:      np.ndarray,   # (n_gt+1, n_cells)
    target_times: list[float], # times to mark vertically [s]
    cfg,
    sim_idx:     int  = 0,
    save_path:   str | None = None,
) -> None:
    """
    Single wide figure showing full timeline:
      - Ground-truth window (0 to t_end)
      - Future extrapolation zone (t_end to predict_future_time)
      - Vertical markers at key target times
      - Min / mean / max temperature bands
    """
    n_gt          = T_true.shape[0]
    n_total       = T_pred.shape[0]
    times_gt      = np.arange(n_gt)    * cfg.dt
    times_pred    = np.arange(n_total) * cfg.dt

    T_mean_gt     = T_true.mean(axis=1)
    T_min_gt      = T_true.min(axis=1)
    T_max_gt      = T_true.max(axis=1)

    T_mean_pred   = T_pred.mean(axis=1)
    T_min_pred    = T_pred.min(axis=1)
    T_max_pred    = T_pred.max(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))

    # OpenFOAM ground truth band
    ax.fill_between(times_gt, T_min_gt, T_max_gt, alpha=0.15, color="black",
                    label="OpenFOAM [min, max]")
    ax.plot(times_gt, T_mean_gt, "k-", lw=2, label="OpenFOAM mean T")

    # GNN within training window
    ax.fill_between(times_pred[:n_gt], T_min_pred[:n_gt], T_max_pred[:n_gt],
                    alpha=0.15, color="red")
    ax.plot(times_pred[:n_gt], T_mean_pred[:n_gt], "r--", lw=2,
            label="GNN mean T (0–4000 s)")

    # GNN future extrapolation
    if n_total > n_gt:
        ax.fill_between(times_pred[n_gt:], T_min_pred[n_gt:], T_max_pred[n_gt:],
                        alpha=0.20, color="purple", label="GNN future [min, max]")
        ax.plot(times_pred[n_gt:], T_mean_pred[n_gt:], "m-", lw=2.5,
                label=f"GNN mean T (future to {times_pred[-1]:.0f}s)")
        ax.axvspan(cfg.t_end, times_pred[-1], alpha=0.06, color="purple")

    # Training window boundary
    ax.axvline(cfg.t_end, color="darkorange", ls="--", lw=2,
               label=f"Training end ({cfg.t_end:.0f}s)")

    # Target time markers
    colors = plt.colormaps["tab10"](np.linspace(0, 0.7, len(target_times)))
    for t_target, col in zip(target_times, colors):
        style = "-" if t_target <= cfg.t_end else ":"
        ax.axvline(t_target, color=col, ls=style, lw=1.5, alpha=0.8,
                   label=f"t={t_target:.0f}s")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Temperature [K]", fontsize=12)
    ax.set_title(
        f"Future temperature prediction — Sim {sim_idx}\n"
        f"Training: 0–{cfg.t_end:.0f}s  |  "
        f"Future prediction to: {cfg.predict_future_time:.0f}s",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
