"""
Orchestrates creating a single OpenFOAM case directory.

Responsibilities:
  1. Copy base case
  2. Clean old time folders / artefacts
  3. Patch heater temperatures
  4. Write thermophysical properties
  5. Patch .geo file for cylinder geometry
  6. Write Allmesh / fix Allrun
  7. Save cylinder_params.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from configs.furnace import HEATER_REGIONS
from configs.parameters import GEO_FILENAME
from src.openfoam.case_cleaner import clean_case
from src.openfoam.heater_patcher import patch_heater_temperatures
from src.openfoam.thermo_writer import write_thermophysical_properties
from src.openfoam.allmesh_writer import write_allmesh
from src.openfoam.allrun_fixer import fix_allrun
from src.geometry.geo_patcher import patch_geo_file
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_single_case(
    base_case: Path,
    case_dir: Path,
    params: dict[str, Any],
) -> bool:
    """Create one OpenFOAM case directory with patched parameters.

    Args:
        base_case: Path to the template OpenFOAM case.
        case_dir:  Destination path for the new case.
        params:    Dictionary of cylinder/material parameters.

    Returns:
        True if all patches succeeded, False otherwise.
    """
    success = True

    # 1. Copy base case
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(base_case, case_dir)

    # 2. Clean old artefacts
    n_cleaned = clean_case(case_dir)
    logger.info("Cleaned %d old items", n_cleaned)

    # 3. Patch heater temperatures
    patch_heater_temperatures(case_dir, params["T_set"], HEATER_REGIONS)

    # 4. Write thermophysical properties
    write_thermophysical_properties(case_dir, params)

    # 5. Patch .geo file
    geo_ok = patch_geo_file(case_dir, params, GEO_FILENAME, base_case)
    if not geo_ok:
        logger.warning("Geometry patch incomplete for %s", case_dir.name)
        success = False

    # 6. Write Allmesh
    write_allmesh(case_dir, GEO_FILENAME, base_case)

    # 7. Fix Allrun
    fix_allrun(case_dir)

    # 8. Save cylinder_params.json
    _write_cylinder_params(case_dir, params)

    return success


def _write_cylinder_params(case_dir: Path, params: dict[str, Any]) -> None:
    """Persist cylinder parameters for use by the dataset builder."""
    keys = [
        "T_set", "cx", "cy", "cz", "radius", "height",
        "volume", "mass", "kappa", "Cp", "rho", "mol_weight",
    ]
    out = {k: float(params[k]) for k in keys if k in params}
    path = case_dir / "cylinder_params.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)