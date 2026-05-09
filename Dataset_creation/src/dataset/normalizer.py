"""Normalisation statistics for the combined dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalisationStats:
    """Z-score normalisation parameters."""

    X_mean: np.ndarray  # (n_features,)
    X_std: np.ndarray   # (n_features,)
    Y_mean: float
    Y_std: float


def compute_normalisation_stats(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float = 1e-8,
) -> NormalisationStats:
    """Compute per-feature mean/std for Z-score normalisation."""
    return NormalisationStats(
        X_mean=X.mean(axis=0).astype(np.float32),
        X_std=(X.std(axis=0) + eps).astype(np.float32),
        Y_mean=float(Y.mean()),
        Y_std=float(Y.std()) + eps,
    )