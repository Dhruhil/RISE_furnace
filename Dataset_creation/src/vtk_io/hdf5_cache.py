"""Per-case HDF5 caching for simulation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

_CACHE_FILENAME = "steel_cylinder_T_timeseries.h5"


def save_case_h5(
    case_dir: Path,
    coords: np.ndarray,
    times: np.ndarray,
    T_array: np.ndarray,
    cyl_params: dict[str, Any],
) -> Path:
    """Cache simulation results as HDF5."""
    h5_path = case_dir / _CACHE_FILENAME
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("times", data=times)
        f.create_dataset("T", data=T_array)
        for k, v in cyl_params.items():
            f.attrs[k] = float(v)

    logger.info("Cached: %s", h5_path)
    return h5_path


def load_case_h5(
    case_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None:
    """Load cached simulation results.

    Returns:
        ``(coords, times, T_array, params)`` or None if no cache exists.
    """
    h5_path = case_dir / _CACHE_FILENAME
    if not h5_path.is_file():
        return None

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:].astype(np.float64)
        times = f["times"][:].astype(np.float64)
        T_array = f["T"][:].astype(np.float64)
        params = {k: float(v) for k, v in f.attrs.items()}

    return coords, times, T_array, params