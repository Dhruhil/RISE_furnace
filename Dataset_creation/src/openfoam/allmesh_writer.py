"""
Write a clean Allmesh script that invokes gmsh before blockMesh.

The Allmesh must:
  1. Source RunFunctions (NO broken ``cd`` line)
  2. Run gmsh to regenerate mesh from the patched .geo
  3. Continue with blockMesh, topoSet, splitMeshRegions, etc.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Lines to strip from any existing Allmesh
_BROKEN_CD_PREFIXES = (
    'cd "${0%/*}"',
    "cd '${0%/*}'",
    "cd ${0%/*}",
)

# Fallback template when the base Allmesh is missing or unreadable
_FALLBACK_TEMPLATE = """\
#!/bin/sh
. ${{WM_PROJECT_DIR:?}}/bin/tools/RunFunctions
#------------------------------------------

# Regenerate mesh from patched .geo
gmsh -3 {geo} -o {msh} -format msh2

runApplication blockMesh
runApplication topoSet
rm log.topoSet
runApplication topoSet -dict system/topoSetDict.f1
restore0Dir

runApplication splitMeshRegions -cellZones -overwrite

for region in $(foamListRegions solid)
do
    rm -f 0/$region/{{nut,alphat,epsilon,k,U,p_rgh}}
    rm -f processor*/0/$region/{{nut,alphat,epsilon,k,U,p_rgh}}
done

for region in $(foamListRegions)
do
    runApplication -s $region changeDictionary -region $region
done

runApplication createBaffles -region rightFluid -overwrite

echo
echo "End"

#------------------------------------------
"""


def write_allmesh(
    case_dir: Path,
    geo_filename: str,
    base_case: Path,
) -> None:
    """Write a corrected Allmesh script with gmsh invocation.

    If the base case Allmesh exists, it is patched (broken ``cd`` removed,
    gmsh inserted). Otherwise, a complete fallback is written.
    """
    allmesh_path = case_dir / "Allmesh"
    msh_filename = geo_filename.replace(".geo", ".msh")
    base_allmesh = base_case / "Allmesh"

    if base_allmesh.is_file():
        lines = base_allmesh.read_text().splitlines(keepends=True)
        new_lines = _patch_existing_allmesh(lines, geo_filename, msh_filename)
    else:
        logger.warning("Base Allmesh not found — writing from template")
        new_lines = [_FALLBACK_TEMPLATE.format(geo=geo_filename, msh=msh_filename)]

    allmesh_path.write_text("".join(new_lines))
    os.chmod(allmesh_path, 0o755)
    logger.info("Allmesh written and made executable")


def _patch_existing_allmesh(
    lines: list[str],
    geo_filename: str,
    msh_filename: str,
) -> list[str]:
    """Remove broken cd, insert gmsh after RunFunctions source line."""
    new_lines: list[str] = []
    gmsh_inserted = False

    for line in lines:
        stripped = line.strip()

        # Skip broken cd lines
        if any(stripped.startswith(prefix) for prefix in _BROKEN_CD_PREFIXES):
            continue

        new_lines.append(line)

        # Insert gmsh right after the RunFunctions source line
        if not gmsh_inserted and "RunFunctions" in stripped:
            new_lines.append("\n")
            new_lines.append("# Regenerate mesh from patched .geo\n")
            new_lines.append(
                f"gmsh -3 {geo_filename} -o {msh_filename} -format msh2\n"
            )
            new_lines.append("\n")
            gmsh_inserted = True

    return new_lines