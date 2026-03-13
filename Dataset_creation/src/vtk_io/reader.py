"""
Read steel_cylinder temperature time series from OpenFOAM VTK output.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path

import numpy as np
import pyvista as pv

from src.utils.logging import get_logger

logger = get_logger(__name__)


def read_steel_timeseries(
    case_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read steel_cylinder temperature over time from VTK.

    Returns:
        ``(coords, times, T_array)`` or None on failure.

        - coords:  ``(n_cells, 3)`` cell-center coordinates
        - times:   ``(n_times,)`` simulation times [s]
        - T_array: ``(n_times, n_cells)`` temperature [K]
    """
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.exists():
        logger.warning("No VTK/ in %s", case_dir)
        return None

    # Find .series file
    series_files = list(vtk_dir.glob("*.series"))
    if not series_files:
        logger.warning("No .series file in %s", vtk_dir)
        series_files = _create_fallback_series(vtk_dir)
        if not series_files:
            return None

    with open(series_files[0]) as f:
        series = json.load(f)

    entries = series["files"]
    file_key = "file" if "file" in entries[0] else "name"
    vtm_paths = [str(vtk_dir / e[file_key]) for e in entries]
    times = np.array(
        [float(e.get("time", i)) for i, e in enumerate(entries)],
        dtype=np.float64,
    )

    logger.info("Time steps: %d, t=[%.1f, %.1f] s", len(times), times[0], times[-1])

    # Coordinates from t=0
    mb0 = pv.read(vtm_paths[0])
    _, ug = _find_steel_internal(mb0)
    if ug is None:
        logger.error("Steel internal block not found. Available: %s", list(mb0.keys()))
        return None

    coords = ug.cell_centers().points.astype(np.float64)
    logger.info("Steel cells: %d", coords.shape[0])

    # Temperature over all time steps
    T_frames: list[np.ndarray] = []
    for vtm_path in vtm_paths:
        mb = pv.read(vtm_path)
        _, ug_t = _find_steel_internal(mb)
        if ug_t is None or "T" not in ug_t.cell_data:
            logger.warning("Missing T in %s", Path(vtm_path).name)
            T_frames.append(np.full(coords.shape[0], np.nan, dtype=np.float64))
        else:
            T_frames.append(ug_t.cell_data["T"].astype(np.float64))

    T_array = np.stack(T_frames, axis=0)
    logger.info("T range: [%.1f, %.1f] K", np.nanmin(T_array), np.nanmax(T_array))

    return coords, times, T_array


def _find_steel_internal(
    multiblock: pv.MultiBlock,
) -> tuple[str | None, pv.UnstructuredGrid | None]:
    """Navigate MultiBlock to find steel region's internal mesh."""
    steel_key = None
    for k in multiblock.keys():
        if "steel" in k.lower():
            steel_key = k
            break

    if steel_key is None:
        return None, None

    steel_mb = multiblock[steel_key]

    # Look for "internal" block
    for k in steel_mb.keys():
        if "internal" in k.lower():
            return steel_key, steel_mb[k]

    # Fallback: first UnstructuredGrid
    for k in steel_mb.keys():
        blk = steel_mb[k]
        if isinstance(blk, pv.UnstructuredGrid):
            return steel_key, blk

    return steel_key, None


def _create_fallback_series(vtk_dir: Path) -> list[Path]:
    """Create a .series file from loose .vtm files."""
    vtm_files = sorted(vtk_dir.glob("*.vtm"))
    if not vtm_files:
        return []

    logger.info("Found %d .vtm files, creating fallback .series", len(vtm_files))
    entries = [
        {"name": f.name, "time": float(i)}
        for i, f in enumerate(vtm_files)
    ]
    series_path = vtk_dir / "case.series"
    with open(series_path, "w") as f:
        json.dump({"files": entries}, f)

    return [series_path]