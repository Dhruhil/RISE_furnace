"""
Error metrics for temperature prediction.

Small grab-bag of helpers used in two places:
  - validation step inside the training loop, where compute_metrics()
    boils a batch of predictions down to one row in the log
  - rollout evaluation, where rollout_summary() pulls out the
    per-step error trajectory used for the thesis figures

Everything operates on plain numpy arrays. PyTorch tensors should
be moved to CPU and converted with .numpy() before calling.
"""
from __future__ import annotations
import numpy as np


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Standard regression metrics for a flattened temperature field.

    Returns MAE, RMSE, max absolute error, and R^2. Everything gets
    cast to float64 first because R^2 in particular is sensitive to
    catastrophic cancellation when the residuals are small.
    """
    # Float64 + ravel so this works for any input shape — per-cell
    # arrays, per-timestep stacks, whatever the caller hands in.
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    err    = y_pred - y_true
    mae    = float(np.mean(np.abs(err)))
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    # Guard against an empty prediction array (can happen if a
    # region has zero cells — shouldn't, but cheap to handle)
    maxe   = float(np.max(np.abs(err))) if len(err) > 0 else 0.0

    # R^2 the textbook way: 1 - SS_res / SS_tot.
    # The +1e-8 keeps the denominator non-zero when y_true is
    # constant (e.g. early heater rollout where T_set hasn't moved).
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-8
    r2     = float(1.0 - ss_res / ss_tot)

    return {"mae": mae, "rmse": rmse, "max_err": maxe, "r2": r2}


def within_tolerance(y_pred: np.ndarray, y_true: np.ndarray, tolerance_K: float) -> float:
    """
    Percentage of cells where |y_pred - y_true| <= tolerance (in K).

    The 5K / 10K / 20K tolerance bands are the more practically
    meaningful summary numbers for industrial heat treatment, where
    the typical control resolution is around +/-10K.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    return 100.0 * float(np.mean(np.abs(y_pred - y_true) <= tolerance_K))


def metrics_per_timestep(T_pred: np.ndarray, T_true: np.ndarray) -> list[dict]:
    """
    Compute metrics independently at each rollout step.

    Both arrays are expected to be shaped [n_steps, n_cells].
    Returns one metrics dict per timestep, in chronological order.
    Used by rollout_summary() to build the error-vs-time curves.
    """
    return [compute_metrics(T_pred[i], T_true[i]) for i in range(T_pred.shape[0])]


def rollout_summary(T_pred: np.ndarray, T_true: np.ndarray, dt: float = 10.0) -> dict:
    """
    One-call summary of an entire autoregressive rollout.

    Bundles the overall (flattened-across-time) metrics with the
    per-step trajectories and the tolerance-band coverage figures.
    Output is a plain dict with all-list values, so it serialises
    straight to JSON without any post-processing.

    Parameters
    ----------
    T_pred : np.ndarray, shape [n_steps, n_cells]
        Surrogate predictions across the rollout horizon.
    T_true : np.ndarray, shape [n_steps, n_cells]
        OpenFOAM ground truth on the matching timesteps.
    dt     : float, default 10.0
        Solver dump interval in seconds — only used to build the
        time axis returned alongside the per-step metrics.
    """
    # Overall metrics — everything flattened across time and space
    overall  = compute_metrics(T_pred.ravel(), T_true.ravel())

    # Per-step error curves — what gets plotted as "MAE vs time"
    per_step = metrics_per_timestep(T_pred, T_true)
    times    = np.arange(T_pred.shape[0]) * dt

    return {
        "overall":    overall,
        "times":      times.tolist(),
        "step_mae":   [m["mae"]  for m in per_step],
        "step_rmse":  [m["rmse"] for m in per_step],
        "step_r2":    [m["r2"]   for m in per_step],
        # Coverage at three tolerance bands. 10K is the headline
        # number reported in the thesis since it matches the
        # industrial control resolution; the other two are for
        # context.
        "within_5K":  within_tolerance(T_pred, T_true,  5.0),
        "within_10K": within_tolerance(T_pred, T_true, 10.0),
        "within_20K": within_tolerance(T_pred, T_true, 20.0),
    }