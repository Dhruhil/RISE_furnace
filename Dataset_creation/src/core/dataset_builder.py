"""Build the combined ML dataset from completed OpenFOAM simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from configs.defaults import PipelineConfig
from configs.parameters import BASE_PARAMS
from src.core.manifest import Manifest
from src.dataset.features import build_feature_matrix, load_cylinder_params
from src.dataset.normalizer import compute_normalisation_stats
from src.dataset.writer import save_combined_dataset
from src.utils.logging import get_logger
from src.vtk_io.hdf5_cache import load_case_h5, save_case_h5
from src.vtk_io.reader import read_steel_timeseries

logger = get_logger(__name__)

# Drop cells whose temperature exceeds this threshold. Above 1773K
# (1500C) is well outside the physical range and almost always
# indicates a numerical artefact at a boundary cell.
_T_OUTLIER_THRESHOLD: float = 1773.0


def build_dataset(cfg: PipelineConfig) -> Path | None:
    """Walk the manifest, read every completed case, write one HDF5 file.

    Returns the path to the saved dataset, or None if nothing could
    be loaded (no base case, or every parameter case was unreadable).
    """
    manifest = Manifest(cfg.manifest_path)
    manifest.load()

    all_X: list[np.ndarray] = []
    all_Y: list[np.ndarray] = []
    case_summary: list[dict[str, Any]] = []

    # the base case has to load - it's both training data and the
    # canonical reference for parameter defaults
    base_result = _process_case(cfg.base_case, BASE_PARAMS.to_dict(), label="base")
    if base_result is None:
        logger.error("Cannot load base case - aborting!")
        return None

    X_b, Y_b, cyl = base_result
    all_X.append(X_b)
    all_Y.append(Y_b)
    case_summary.append({"case": "base", **cyl, "n_rows": X_b.shape[0]})

    # parameter-study cases - failures are logged but don't abort
    for entry in manifest.entries:
        if entry["case"] == "base_case_that_runs_chnage":
            continue

        case_dir = cfg.output_dir / entry["case"]
        result = _process_case(case_dir, entry, label=entry["case"])
        if result is None:
            continue

        X_i, Y_i, cyl = result
        all_X.append(X_i)
        all_Y.append(Y_i)
        manifest.update_status(entry["case"], "completed")
        case_summary.append({"case": entry["case"], **cyl, "n_rows": X_i.shape[0]})

    if not all_X:
        logger.error("No simulation data loaded!")
        return None

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    Y = np.concatenate(all_Y, axis=0).astype(np.float32)
    stats = compute_normalisation_stats(X, Y)

    out_path = save_combined_dataset(
        path=cfg.dataset_path,
        X=X,
        Y=Y,
        stats=stats,
        all_X=all_X,
        case_summary=case_summary,
        n_simulations=len(all_X),
    )
    manifest.save()

    logger.info(
        "DONE: %d simulations -> %s training points",
        len(all_X), f"{X.shape[0]:,}",
    )
    return out_path


def _process_case(
    case_dir: Path,
    params: dict[str, Any],
    label: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Load one simulation and return its feature matrix + targets.

    Tries the per-case HDF5 cache first; falls back to reading the raw
    VTK output (which is slow) and writing a cache for next time.
    Returns None if neither cache nor VTK is readable.
    """
    logger.info("Processing: %s", label)

    cached = load_case_h5(case_dir)
    if cached is not None:
        coords, times, T_array, cyl_cached = cached
        cyl = load_cylinder_params(case_dir, params)
        cyl.update(cyl_cached)
        logger.info("  Loaded from HDF5 cache")
    else:
        result = read_steel_timeseries(case_dir)
        if result is None:
            logger.warning("  VTK read failed for %s", label)
            return None
        coords, times, T_array = result
        cyl = load_cylinder_params(case_dir, params)
        save_case_h5(case_dir, coords, times, T_array, cyl)

    X, Y = build_feature_matrix(coords, times, T_array, cyl)

    # filter unphysical boundary cells before they pollute the normalisation
    mask = Y.ravel() < _T_OUTLIER_THRESHOLD
    n_dropped = len(mask) - int(mask.sum())
    if n_dropped > 0:
        logger.info("  Filtered %d outlier cells (T > %.0f K)",
                    n_dropped, _T_OUTLIER_THRESHOLD)
        X, Y = X[mask], Y[mask]

    logger.info("  Rows: %s", f"{X.shape[0]:,}")
    return X, Y, cyl