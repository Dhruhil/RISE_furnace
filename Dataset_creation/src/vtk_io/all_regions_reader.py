"""
Read ALL furnace regions from OpenFOAM VTK output.

Extends the existing reader.py (which only reads steel_cylinder)
to extract temperature fields from all 12 regions:
  steel_cylinder, inner_box, heater_1-8, brick_heater, outer_box

Used by scripts/create_all_regions_dataset.py (Step 5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from src.utils.logging import get_logger

logger = get_logger(__name__)


# All regions in the furnace model
REGIONS = [
    "steel_cylinder",
    "inner_box",
    "heater_1",
    "heater_2",
    "heater_3",
    "heater_4",
    "heater_5",
    "heater_6",
    "heater_7",
    "heater_8",
    "brick_heater",
    "outer_box",
]


def find_region_block(
    multiblock: pv.MultiBlock, region_name: str
) -> pv.UnstructuredGrid | None:
    """
    Navigate the OpenFOAM MultiBlock VTK structure to find a region.

    OpenFOAM VTK structure (from foamToVTK -allRegions):
        multiblock
        └── region_name (e.g. "steel_cylinder")
            ├── "internal" → UnstructuredGrid (cell data with T)
            └── boundary patches...

    Or flat naming:
        multiblock
        └── "region_name_internal" → UnstructuredGrid
    """
    for i in range(multiblock.n_blocks):
        name = multiblock.get_block_name(i)
        block = multiblock[i]

        if block is None:
            continue

        # Direct match: block name is the region
        if name == region_name:
            if isinstance(block, pv.MultiBlock):
                # Look for "internal" sub-block
                for j in range(block.n_blocks):
                    sub_name = block.get_block_name(j)
                    sub_block = block[j]
                    if sub_name and "internal" in sub_name.lower():
                        if isinstance(sub_block, pv.UnstructuredGrid):
                            return sub_block
                # If no "internal" found, try first UnstructuredGrid
                for j in range(block.n_blocks):
                    sub_block = block[j]
                    if isinstance(sub_block, pv.UnstructuredGrid):
                        return sub_block
            elif isinstance(block, pv.UnstructuredGrid):
                return block

        # Flat naming: "steel_cylinder_internal"
        if name and region_name in name and "internal" in name.lower():
            if isinstance(block, pv.UnstructuredGrid):
                return block

    return None


def read_case_all_regions(
    case_dir: Path,
) -> dict[str, Any] | None:
    """
    Read all regions from one case's VTK output.

    Returns:
        {
            "times": np.ndarray (n_times,) float32,
            "regions": {
                "steel_cylinder": {"coords": (n_cells, 3), "T": (n_times, n_cells)},
                "inner_box": {...},
                ...
            }
        }
        or None on failure.
    """
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.exists():
        logger.warning("No VTK/ in %s", case_dir)
        return None

    # ── Find .series file or .vtm files ────────────────────────────
    series_files = list(vtk_dir.glob("*.series"))
    if not series_files:
        vtm_files = sorted(vtk_dir.glob("*.vtm"))
        if not vtm_files:
            logger.warning("No .series or .vtm files in %s", vtk_dir)
            return None
        vtm_paths = [str(f) for f in vtm_files]
        times = np.arange(len(vtm_files), dtype=np.float64)
    else:
        with open(series_files[0]) as f:
            series = json.load(f)

        entries = series["files"]
        file_key = "file" if "file" in entries[0] else "name"
        vtm_paths = [str(vtk_dir / e[file_key]) for e in entries]
        times = np.array(
            [float(e.get("time", i)) for i, e in enumerate(entries)],
            dtype=np.float64,
        )

    logger.info("  Time steps: %d, t=[%.1f, %.1f] s", len(times), times[0], times[-1])

    # ── Read first timestep to discover regions and get coordinates ─
    mb0 = pv.read(vtm_paths[0])

    region_data: dict[str, dict] = {}
    for region in REGIONS:
        ug = find_region_block(mb0, region)
        if ug is None:
            continue
        coords = ug.cell_centers().points.astype(np.float32)
        n_cells = coords.shape[0]
        region_data[region] = {
            "coords": coords,
            "n_cells": n_cells,
            "T_frames": [],
        }

    if not region_data:
        logger.warning("  No matching regions found. Available blocks:")
        _print_multiblock_tree(mb0)
        return None

    found_regions = list(region_data.keys())
    logger.info("  Regions found: %s", found_regions)
    for r in found_regions:
        logger.info("    %s: %d cells", r, region_data[r]["n_cells"])

    # ── Read temperature at every timestep ──────────────────────────
    for ti, vtm_path in enumerate(vtm_paths):
        mb = pv.read(vtm_path)
        for region in found_regions:
            ug = find_region_block(mb, region)
            if ug is not None and "T" in ug.cell_data:
                T = ug.cell_data["T"].astype(np.float32)
                region_data[region]["T_frames"].append(T)
            else:
                n_cells = region_data[region]["n_cells"]
                region_data[region]["T_frames"].append(
                    np.full(n_cells, np.nan, dtype=np.float32)
                )

    # ── Stack into arrays and build result ─────────────────────────
    result_regions = {}
    for region in found_regions:
        T_array = np.stack(region_data[region]["T_frames"], axis=0)
        result_regions[region] = {
            "coords": region_data[region]["coords"],
            "T": T_array,
        }
        logger.info(
            "    %s: T shape=%s, range=[%.1f, %.1f] K",
            region, T_array.shape,
            np.nanmin(T_array), np.nanmax(T_array),
        )

    return {"times": times.astype(np.float32), "regions": result_regions}


def _print_multiblock_tree(mb: pv.MultiBlock, indent: int = 4) -> None:
    """Debug helper: print the MultiBlock tree structure."""
    for i in range(mb.n_blocks):
        name = mb.get_block_name(i)
        block = mb[i]
        prefix = " " * indent
        if block is None:
            logger.info("%s- %s: None", prefix, name)
        elif isinstance(block, pv.MultiBlock):
            logger.info("%s- %s: MultiBlock (%d blocks)", prefix, name, block.n_blocks)
            _print_multiblock_tree(block, indent + 2)
        elif isinstance(block, pv.UnstructuredGrid):
            arrays = list(block.cell_data.keys()) if block.cell_data else []
            logger.info(
                "%s- %s: UnstructuredGrid (%d cells, arrays=%s)",
                prefix, name, block.n_cells, arrays,
            )
        else:
            logger.info("%s- %s: %s", prefix, name, type(block).__name__)
