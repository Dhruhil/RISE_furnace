"""
Error metrics for temperature prediction.

BUGS FIXED vs old version:
  1. Missing within_tolerance() function
     → training/train.py calls within_tolerance(y_pred, y_true, 5.0)
     → ImportError crash on startup
  2. Missing rollout_summary() and metrics_per_timestep() in some versions
"""

from __future__ import annotations
import numpy as np


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute MAE, RMSE, MaxError, R² between two arrays."""
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    err    = y_pred - y_true
    mae    = float(np.mean(np.abs(err)))
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    maxe   = float(np.max(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-8
    r2     = float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "max_err": maxe, "r2": r2}


def within_tolerance(
    y_pred:      np.ndarray,
    y_true:      np.ndarray,
    tolerance_K: float,
) -> float:
    """
    Percentage of predictions within ±tolerance_K of ground truth.

    FIX: This function was missing from old metrics.py.
    training/train.py calls:
        within_tolerance(y_pred, y_true, 5.0)
        within_tolerance(y_pred, y_true, 10.0)
        within_tolerance(y_pred, y_true, 20.0)
    Without this function → ImportError crash on startup.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    return 100.0 * float(np.mean(np.abs(y_pred - y_true) <= tolerance_K))


def metrics_per_timestep(
    T_pred: np.ndarray,   # (n_steps, n_nodes)
    T_true: np.ndarray,   # (n_steps, n_nodes)
) -> list[dict]:
    """Per-time-step metrics for a rollout."""
    return [
        compute_metrics(T_pred[i], T_true[i])
        for i in range(T_pred.shape[0])
    ]


def rollout_summary(
    T_pred: np.ndarray,
    T_true: np.ndarray,
    dt:     float = 10.0,
) -> dict:
    """Full rollout summary: overall + per-step + within-tolerance."""
    overall   = compute_metrics(T_pred.ravel(), T_true.ravel())
    per_step  = metrics_per_timestep(T_pred, T_true)
    times     = np.arange(T_pred.shape[0]) * dt
    step_mae  = np.array([m["mae"]  for m in per_step])
    step_rmse = np.array([m["rmse"] for m in per_step])
    step_r2   = np.array([m["r2"]   for m in per_step])
    return {
        "overall":    overall,
        "times":      times.tolist(),
        "step_mae":   step_mae.tolist(),
        "step_rmse":  step_rmse.tolist(),
        "step_r2":    step_r2.tolist(),
        "within_5K":  within_tolerance(T_pred, T_true,  5.0),
        "within_10K": within_tolerance(T_pred, T_true, 10.0),
        "within_20K": within_tolerance(T_pred, T_true, 20.0),
    }
