import numpy as np


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if pred.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    err = pred - true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + 1e-12
    r2   = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def within_tolerance(pred, true, tol=5.0):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true)) <= tol))
