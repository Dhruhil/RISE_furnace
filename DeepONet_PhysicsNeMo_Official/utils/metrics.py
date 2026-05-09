"""
Metrics for the DeepONet evaluation pipeline.

Same two helpers as the GNN / FNO metrics modules — function
signatures and return-dict keys match exactly, which keeps the
comparison numbers in the thesis tables on equal footing across
architectures.

Slight difference vs the FNO version: this one also filters out
non-finite values up front. The DeepONet's autograd path can
occasionally produce a NaN at a query point during long rollouts
(usually right where the trunk extrapolates outside the training
geometry), and dropping those silently here keeps a single bad
sample from flipping the whole MAE / R² to NaN.
"""
import numpy as np


def compute_metrics(pred, true):
    """
    Compute MAE, RMSE, and R^2 between predicted and true arrays.

    Both inputs get flattened first, so the same function works for
    per-step (1D), per-region (2D), or full-rollout (3D) tensors.
    The +1e-12 in the R^2 denominator keeps things finite when
    `true` is constant — happens in the late-rollout plateau
    where every cell has reached T_set.
    """
    # Cast to float64 — R^2 in particular is sensitive to
    # catastrophic cancellation when residuals are small.
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)

    # Drop any NaN / inf entries. Better to score on the finite
    # subset than have one stray non-finite value poison the
    # whole batch's metrics.
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]

    # Empty after filtering — return NaNs rather than dividing by
    # zero. Caller can decide whether to skip or warn.
    if pred.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    err = pred - true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # R^2 the textbook way: 1 - SS_res / SS_tot
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + 1e-12
    r2   = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def within_tolerance(pred, true, tol=5.0):
    """
    Fraction of predictions within `tol` Kelvin of the ground truth.

    Returns a value in [0, 1] — the eval scripts multiply by 100
    when they want a percentage. The 5K / 10K / 20K bands are the
    practically meaningful summary numbers for industrial heat
    treatment, where the typical control resolution is around
    +/-10 K.
    """
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true)) <= tol))