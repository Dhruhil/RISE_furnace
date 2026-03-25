"""Metrics for PINN evaluation — same interface as GNN/FNO."""
from __future__ import annotations
import numpy as np

def compute_metrics(y_pred, y_true) -> dict:
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {"mae": mae, "rmse": rmse, "r2": r2}

def within_tolerance(y_pred, y_true, tol_K) -> float:
    return float(np.mean(np.abs(y_pred - y_true) < tol_K) * 100)
