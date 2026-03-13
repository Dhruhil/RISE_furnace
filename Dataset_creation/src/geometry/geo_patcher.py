"""
Patch the GMSH .geo file for modified cylinder parameters.

Modifies:
  1. Disk(45) center and radius
  2. Extrude height for Surface{45}
  3. Strips structured mesh constraints (Layers, Transfinite, Recombine)
  4. Appends unstructured mesh settings
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Mesh settings appended to every patched .geo file
_UNSTRUCTURED_MESH_FOOTER = """
// ---- Mesh settings for unstructured tet meshing ----
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.OptimizeNetgen = 0;
Mesh.Optimize = 1;
Mesh.CharacteristicLengthMin = 0.003;
Mesh.CharacteristicLengthMax = 0.015;
"""


def patch_geo_file(
    case_dir: Path,
    params: dict[str, Any],
    geo_filename: str,
    base_case: Path,
) -> bool:
    """Patch .geo file with new cylinder geometry and clean mesh constraints.

    Returns:
        True if both Disk and Extrude were patched exactly once.
    """
    geo_src = base_case / geo_filename
    geo_dst = case_dir / geo_filename

    if not geo_src.is_file():
        logger.warning(".geo not found: %s", geo_src)
        return False

    content = geo_src.read_text()

    # 1. Patch Disk(45)
    content, n_disk = _patch_disk(content, params)

    # 2. Patch Extrude height
    content, n_ext = _patch_extrude(content, params)

    # 3. Strip structured mesh constraints
    content = _strip_structured_constraints(content)

    # 4. Append unstructured settings
    content += _UNSTRUCTURED_MESH_FOOTER

    geo_dst.write_text(content)

    ok = (n_disk == 1) and (n_ext == 1)
    logger.info(
        "Geo patched: Disk=%d, Extrude=%d → %s",
        n_disk, n_ext, "OK" if ok else "PARTIAL",
    )
    return ok


def _patch_disk(content: str, params: dict[str, Any]) -> tuple[str, int]:
    """Replace Disk(45) = {cx, cy, cz, r, r}."""
    pattern = re.compile(r'(Disk\s*\(\s*45\s*\)\s*=\s*\{)[^}]+(\};)')
    replacement = (
        rf"\g<1>{params['cx']}, {params['cy']}, {params['cz']}, "
        rf"{params['radius']}, {params['radius']}\g<2>"
    )
    return pattern.subn(replacement, content)


def _patch_extrude(content: str, params: dict[str, Any]) -> tuple[str, int]:
    """Replace extrude height in Extrude{h,0,0}{Surface{45}...}."""
    lines = content.split("\n")
    result_lines: list[str] = []
    n_patched = 0

    extrude_pat = re.compile(
        r'^(\s*Extrude\s*\{)\s*([\d.\-e+]+)(\s*,\s*0\s*,\s*0\s*\}\s*\{.*)'
    )
    surface45_pat = re.compile(r'Surface\s*\{\s*45\s*\}')

    i = 0
    while i < len(lines):
        line = lines[i]
        m = extrude_pat.match(line)

        if m:
            # Collect entire Extrude block to check for Surface{45}
            block = [line]
            depth = line.count("{") - line.count("}")
            j = i + 1
            while j < len(lines) and depth > 0:
                block.append(lines[j])
                depth += lines[j].count("{") - lines[j].count("}")
                j += 1

            if surface45_pat.search("\n".join(block)):
                patched = f"{m.group(1)}{params['height']}{m.group(3)}"
                result_lines.append(patched)
                result_lines.extend(block[1:])
                n_patched += 1
                i = j
                continue

        result_lines.append(line)
        i += 1

    return "\n".join(result_lines), n_patched


def _strip_structured_constraints(content: str) -> str:
    """Remove Layers, Transfinite, and Recombine directives."""
    content = re.sub(
        r'Layers\s*\{[^}]*\}\s*;?',
        '/* Layers removed */',
        content,
    )
    content = re.sub(
        r'Transfinite\s+(Curve|Surface|Volume)\s*[^;]*;',
        r'/* Transfinite \1 removed */',
        content,
    )
    content = re.sub(
        r'Recombine\s+Surface\s*[^;]*;',
        '/* Recombine removed */',
        content,
    )
    return content