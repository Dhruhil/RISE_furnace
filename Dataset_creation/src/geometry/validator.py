"""
Validate that a cylinder geometry fits inside the furnace.
"""
from __future__ import annotations
from typing import Any
from configs.furnace import FurnaceBounds, FURNACE_BOUNDS

def validate_cylinder_geometry(
    params: dict[str, Any],
    bounds: FurnaceBounds = FURNACE_BOUNDS,
) -> bool:
    """Check that the cylinder fits inside the furnace.
    
    The cylinder can extend beyond the inner box into the outer box
    region (confirmed valid in Gmsh via BooleanFragments).
    
    Clearances:
      - x: cx >= -0.14 (outer box lower bound)
      - x: cx + height <= 0.346 (outer box upper bound)
      - y: cy +/- radius within [y_min + 0.07, y_max - 0.07]
      - z: cz +/- radius within [z_min + 0.01, z_max - 0.01]
    """
    r = params["radius"]
    h = params["height"]
    cx = params.get("cx", 0.0)
    cy = params["cy"]
    cz = params["cz"]

    # x: must stay within outer box
    if cx < -0.14:
        return False
    if cx + h > 0.346:
        return False

    # y: clear of brick_heater boundaries
    if cy - r < bounds.y_min + 0.07:
        return False
    if cy + r > bounds.y_max - 0.07:
        return False

    # z: clear of furnace walls
    if cz - r < bounds.z_min + 0.01:
        return False
    if cz + r > bounds.z_max - 0.01:
        return False

    return True
