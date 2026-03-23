"""
Metrics for FNO evaluation.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Same interface as GNN project for consistent comparison.
"""
from __future__ import annotations

import numpy as np


def compute_metrics(y_pred, y_true) -> dict:
    """Compute MAE, RMSE, and R² between predicted and true arrays."""
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def within_tolerance(y_pred, y_true, tol_K) -> float:
    """Percentage of predictions within tol_K of ground truth."""
    return float(np.mean(np.abs(y_pred - y_true) < tol_K) * 100)


def metrics_per_timestep(T_pred, T_true) -> list[dict]:
    """Compute metrics at each timestep."""
    n = min(T_pred.shape[0], T_true.shape[0])
    return [compute_metrics(T_pred[i], T_true[i]) for i in range(n)]
