"""
Validate that a cylinder geometry fits inside the furnace inner box.
"""

from __future__ import annotations

from typing import Any

from configs.furnace import FurnaceBounds, FURNACE_BOUNDS


def validate_cylinder_geometry(
    params: dict[str, Any],
    bounds: FurnaceBounds = FURNACE_BOUNDS,
) -> bool:
    """Check that the cylinder fits inside the furnace inner box.

    Clearances:
      - x: cx + height <= x_max - 0.01
      - y: cy ± radius within [y_min + 0.07, y_max - 0.07]  (clear of brick_heater)
      - z: cz ± radius within [z_min + 0.01, z_max - 0.01]
    """
    r = params["radius"]
    h = params["height"]
    cx = params.get("cx", 0.0)
    cy = params["cy"]
    cz = params["cz"]

    # x: extrude from cx to cx + h
    if cx + h > bounds.x_max - 0.01:
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