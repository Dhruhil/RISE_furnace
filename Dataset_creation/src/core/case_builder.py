"""Build a single OpenFOAM case directory from the base case + parameters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from configs.furnace import HEATER_REGIONS
from configs.parameters import GEO_FILENAME
from src.geometry.geo_patcher import patch_geo_file
from src.openfoam.allmesh_writer import write_allmesh
from src.openfoam.allrun_fixer import fix_allrun
from src.openfoam.case_cleaner import clean_case
from src.openfoam.heater_patcher import patch_heater_temperatures
from src.openfoam.thermo_writer import write_thermophysical_properties
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Keys persisted to cylinder_params.json. Match FEATURE_COLUMNS minus
# (x, y, z, t) which come from the VTK output, not from the params.
_PERSIST_KEYS: tuple[str, ...] = (
    "T_set", "cx", "cy", "cz", "radius", "height",
    "kappa", "Cp", "rho", "mol_weight", "brick_heater_kappa",
)


def build_single_case(
    base_case: Path,
    case_dir: Path,
    params: dict[str, Any],
) -> bool:
    """Create one OpenFOAM case directory with patched parameters.

    Steps performed in order:
        1. Copy base case (overwriting any existing case_dir)
        2. Clean stale time folders, VTK output, and old artefacts
        3. Patch heater temperatures in 0/heater_*/T
        4. Write steel + brick thermophysicalProperties
        5. Patch the .geo file with new cylinder geometry
        6. Regenerate Allmesh (with the viewFactorWall fix)
        7. Strip the broken `cd "${0%/*}"` line from Allrun
        8. Persist cylinder_params.json for the dataset builder

    Returns True if all patches applied cleanly. Currently only the
    .geo patch can report partial failure - everything else either
    succeeds or raises.
    """
    # 1. fresh copy of the base case
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(base_case, case_dir)

    # 2. clean stale artefacts that travelled with the copy
    n_cleaned = clean_case(case_dir)
    logger.info("Cleaned %d old items", n_cleaned)

    # 3-4. material properties
    patch_heater_temperatures(case_dir, params["T_set"], HEATER_REGIONS)
    write_thermophysical_properties(case_dir, params)

    # 5. geometry - the only step that can partially fail
    geo_ok = patch_geo_file(case_dir, params, GEO_FILENAME, base_case)
    if not geo_ok:
        logger.warning("Geometry patch incomplete for %s", case_dir.name)

    # 6-7. mesh + run scripts
    write_allmesh(case_dir, GEO_FILENAME, base_case)
    fix_allrun(case_dir)

    # 8. record what was used so the dataset builder doesn't need the manifest
    _write_cylinder_params(case_dir, params)

    return geo_ok


def _write_cylinder_params(case_dir: Path, params: dict[str, Any]) -> None:
    """Persist case parameters next to the OpenFOAM files."""
    out = {k: float(params[k]) for k in _PERSIST_KEYS if k in params}
    path = case_dir / "cylinder_params.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)