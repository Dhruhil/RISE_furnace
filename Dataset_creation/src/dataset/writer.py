"""Save the combined dataset to HDF5."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from configs.parameters import FEATURE_COLUMNS, TARGET_COLUMN
from src.dataset.normalizer import NormalisationStats
from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_combined_dataset(
    path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    stats: NormalisationStats,
    all_X: list[np.ndarray],
    case_summary: list[dict[str, Any]],
    n_simulations: int,
) -> Path:
    """Write the combined, normalised dataset with full metadata.

    HDF5 layout:
      /X_raw, /Y_raw           — raw feature/target matrices
      /X_norm, /Y_norm         — Z-score normalised
      /X_mean, /X_std          — normalisation parameters (REQUIRED for inference)
      /Y_mean, /Y_std
      /sim_start_indices       — per-simulation row boundaries
      /sim_n_rows
      attrs: feature_cols, target_col, n_simulations, ...
    """
    X_norm = ((X - stats.X_mean) / stats.X_std).astype(np.float32)
    Y_norm = ((Y - stats.Y_mean) / stats.Y_std).astype(np.float32)

    with h5py.File(path, "w") as f:
        # Raw data
        f.create_dataset("X_raw", data=X, compression="gzip", chunks=True)
        f.create_dataset("Y_raw", data=Y, compression="gzip", chunks=True)

        # Normalised data
        f.create_dataset("X_norm", data=X_norm, compression="gzip", chunks=True)
        f.create_dataset("Y_norm", data=Y_norm, compression="gzip", chunks=True)

        # Normalisation stats
        f.create_dataset("X_mean", data=stats.X_mean)
        f.create_dataset("X_std", data=stats.X_std)
        f.create_dataset("Y_mean", data=np.float32(stats.Y_mean))
        f.create_dataset("Y_std", data=np.float32(stats.Y_std))

        # Metadata
        f.attrs["feature_cols"] = json.dumps(FEATURE_COLUMNS)
        f.attrs["target_col"] = TARGET_COLUMN
        f.attrs["n_simulations"] = n_simulations
        f.attrs["total_points"] = int(X.shape[0])
        f.attrs["n_features"] = len(FEATURE_COLUMNS)
        f.attrs["case_summary"] = json.dumps(case_summary)

        # Per-simulation indices (for data-leakage-free train/val splits)
        sim_starts = np.array(
            [0] + list(np.cumsum([b.shape[0] for b in all_X[:-1]])),
            dtype=np.int64,
        )
        f.create_dataset("sim_start_indices", data=sim_starts)
        f.create_dataset(
            "sim_n_rows",
            data=np.array([b.shape[0] for b in all_X], dtype=np.int64),
        )

    size_mb = os.path.getsize(path) / 1e6
    logger.info("Saved: %s — X=%s, Y=%s (%.1f MB)", path, X.shape, Y.shape, size_mb)
    return path