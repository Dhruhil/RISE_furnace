"""
Metrics for FNO evaluation.

Same three helpers as the GNN's utils/metrics.py — kept here as a
parallel module so the FNO scripts can stay self-contained and so
the FNO training loop doesn't depend on imports from the GNN
package. Function signatures and return-dict keys match exactly,
which keeps the comparison numbers in the thesis tables on equal
footing across architectures.
"""
from __future__ import annotations

import numpy as np


def compute_metrics(y_pred, y_true) -> dict:
    """
    Compute MAE, RMSE, and R^2 between predicted and true arrays.

    Both inputs get flattened first, so the same function works for
    per-step (1D), per-region (2D), or full-rollout (3D) tensors.
    The +1e-12 in the R^2 denominator keeps things finite when
    y_true is constant — happens in the late-rollout plateau where
    every cell has reached T_set.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def within_tolerance(y_pred, y_true, tol_K) -> float:
    """
    Percentage of predictions within tol_K of the ground truth.

    The 5K / 10K / 20K bands are the practically meaningful summary
    numbers for industrial heat treatment, where the typical control
    resolution is around +/-10 K.
    """
    return float(np.mean(np.abs(y_pred - y_true) < tol_K) * 100)


def metrics_per_timestep(T_pred, T_true) -> list[dict]:
    """
    Compute metrics independently at each rollout step.

    Both arrays are expected to be shaped [n_steps, n_cells].
    Returns one metrics dict per timestep, in chronological order —
    used by the plotting scripts to produce the error-vs-time
    curves in Section 5.3 of the thesis.
    """
    # Clip to the shorter array so a missing tail step on either
    # side doesn't trip an index error.
    n = min(T_pred.shape[0], T_true.shape[0])
    return [compute_metrics(T_pred[i], T_true[i]) for i in range(n)]