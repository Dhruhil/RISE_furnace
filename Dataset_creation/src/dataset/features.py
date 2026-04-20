"""
Build the feature matrix X and target vector Y from simulation data.

Feature layout (14 columns — immutable after first training):
  [0]  x        [1]  y       [2]  z        [3]  t
  [4]  T_set    [5]  cx      [6]  cy       [7]  cz
  [8]  radius   [9]  height  [10] kappa    [11] Cp
  [12] rho      [13] brick_heater_kappa

CHANGES from v1 (15 cols):
  - Removed: volume (col 10), mass (col 11)
  - Added:   brick_heater_kappa (col 13)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from configs.parameters import BASE_PARAMS, FEATURE_COLUMNS


def build_feature_matrix(
    coords: np.ndarray,
    times: np.ndarray,
    T_array: np.ndarray,
    cyl: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct X (features) and Y (target) matrices.

    Args:
        coords:  ``(n_cells, 3)`` cell-center coordinates.
        times:   ``(n_times,)`` simulation times.
        T_array: ``(n_times, n_cells)`` temperatures.
        cyl:     Cylinder/material parameters.

    Returns:
        ``(X, Y)`` with shapes ``(N, 14)`` and ``(N, 1)`` as float32,
        where ``N = n_valid_timesteps × n_cells``.
    """
    n_cells = coords.shape[0]
    blocks_X: list[np.ndarray] = []
    blocks_Y: list[np.ndarray] = []

    for ti, t_val in enumerate(times):
        if np.any(np.isnan(T_array[ti])):
            continue

        X_block = np.column_stack([
            coords[:, 0],                                                    # x
            coords[:, 1],                                                    # y
            coords[:, 2],                                                    # z
            np.full(n_cells, t_val, dtype=np.float64),                       # t
            np.full(n_cells, cyl["T_set"], dtype=np.float32),                # T_set
            np.full(n_cells, cyl.get("cx", 0.0), dtype=np.float64),         # cx
            np.full(n_cells, cyl["cy"], dtype=np.float64),                   # cy
            np.full(n_cells, cyl["cz"], dtype=np.float64),                   # cz
            np.full(n_cells, cyl["radius"], dtype=np.float64),               # radius
            np.full(n_cells, cyl["height"], dtype=np.float64),               # height
            np.full(n_cells, cyl["kappa"], dtype=np.float32),                # kappa
            np.full(n_cells, cyl["Cp"], dtype=np.float32),                   # Cp
            np.full(n_cells, cyl["rho"], dtype=np.float32),                  # rho
            np.full(n_cells, cyl.get("brick_heater_kappa", 8.0),            # brick_heater_kappa
                    dtype=np.float32),
        ]).astype(np.float32)

        Y_block = T_array[ti, :].reshape(-1, 1).astype(np.float32)

        blocks_X.append(X_block)
        blocks_Y.append(Y_block)

    X = np.concatenate(blocks_X, axis=0)
    Y = np.concatenate(blocks_Y, axis=0)
    return X, Y


def load_cylinder_params(
    case_dir: Path,
    manifest_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load cylinder parameters with fallback chain.

    Priority:
      1. ``cylinder_params.json`` (written by case builder)
      2. Manifest entry
      3. BASE_PARAMS defaults
    """
    json_path = case_dir / "cylinder_params.json"

    if json_path.is_file():
        with open(json_path) as f:
            p = json.load(f)
        _ensure_derived_fields(p)
        return p

    if manifest_entry is not None:
        p = {
            k: float(manifest_entry[k])
            for k in FEATURE_COLUMNS
            if k not in ("x", "y", "z", "t") and k in manifest_entry
        }
        _ensure_derived_fields(p)
        return p

    return BASE_PARAMS.to_dict()


def _ensure_derived_fields(p: dict[str, Any]) -> None:
    """Add brick_heater_kappa default if missing (backward compat)."""
    if "brick_heater_kappa" not in p:
        p["brick_heater_kappa"] = 8.0