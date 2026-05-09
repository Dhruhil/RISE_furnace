"""Write the combined dataset (raw + normalised + metadata) to one HDF5 file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

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
    """Write the full dataset with everything needed for inference and
    leak-free train/val splits.

    HDF5 layout:
        /X_raw, /Y_raw           raw feature/target matrices
        /X_norm, /Y_norm         z-score normalised
        /X_mean, /X_std          per-feature normalisation stats
        /Y_mean, /Y_std          scalar normalisation stats
        /sim_start_indices       row index where each simulation begins
        /sim_n_rows              row count per simulation
        attrs:
            feature_cols, target_col, n_simulations, total_points,
            n_features, case_summary
    """
    X_norm = ((X - stats.X_mean) / stats.X_std).astype(np.float32)
    Y_norm = ((Y - stats.Y_mean) / stats.Y_std).astype(np.float32)

    sim_n_rows = np.array([b.shape[0] for b in all_X], dtype=np.int64)
    # cumulative starts: sim 0 begins at row 0, sim 1 at len(sim 0), etc.
    sim_starts = np.concatenate(([0], np.cumsum(sim_n_rows[:-1]))).astype(np.int64)

    with h5py.File(path, "w") as f:
        f.create_dataset("X_raw",  data=X,      compression="gzip", chunks=True)
        f.create_dataset("Y_raw",  data=Y,      compression="gzip", chunks=True)
        f.create_dataset("X_norm", data=X_norm, compression="gzip", chunks=True)
        f.create_dataset("Y_norm", data=Y_norm, compression="gzip", chunks=True)

        # normalisation stats - REQUIRED to denormalise predictions at inference
        f.create_dataset("X_mean", data=stats.X_mean)
        f.create_dataset("X_std",  data=stats.X_std)
        f.create_dataset("Y_mean", data=np.float32(stats.Y_mean))
        f.create_dataset("Y_std",  data=np.float32(stats.Y_std))

        # per-simulation indices - lets train/val split by case so no
        # rows from the same simulation appear in both splits
        f.create_dataset("sim_start_indices", data=sim_starts)
        f.create_dataset("sim_n_rows",        data=sim_n_rows)

        f.attrs["feature_cols"]  = json.dumps(FEATURE_COLUMNS)
        f.attrs["target_col"]    = TARGET_COLUMN
        f.attrs["n_simulations"] = n_simulations
        f.attrs["total_points"]  = int(X.shape[0])
        f.attrs["n_features"]    = len(FEATURE_COLUMNS)
        f.attrs["case_summary"]  = json.dumps(case_summary)

    size_mb = os.path.getsize(path) / 1e6
    logger.info("Saved: %s - X=%s, Y=%s (%.1f MB)", path, X.shape, Y.shape, size_mb)
    return path