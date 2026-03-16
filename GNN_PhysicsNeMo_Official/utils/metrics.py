"""Error metrics for temperature prediction."""
from __future__ import annotations
import numpy as np


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute MAE, RMSE, MaxError, R2 between two 1-D arrays."""
    err = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    maxe = float(np.max(np.abs(err)))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean())**2) + 1e-8
    r2   = float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "max_err": maxe, "r2": r2}


def metrics_per_timestep(T_pred: np.ndarray, T_true: np.ndarray) -> list[dict]:
    """
    Per-time-step metrics.

    Args:
        T_pred: (n_steps, n_nodes)
        T_true: (n_steps, n_nodes)

    Returns:
        list of metric dicts, one per time step
    """
    return [compute_metrics(T_pred[i], T_true[i])
            for i in range(T_pred.shape[0])]