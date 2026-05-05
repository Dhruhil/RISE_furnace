"""
Furnace geometry constants.

From the .geo file analysis:
    inner_box ~ x:[0, 0.206], y:[0, 0.36], z:[0, 0.39]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FurnaceBounds:
    """Furnace inner-box dimensions in metres."""

    x_min: float = 0.0
    x_max: float = 0.206
    y_min: float = 0.0
    y_max: float = 0.36
    z_min: float = 0.0
    z_max: float = 0.39


# frozen singleton - safe to import from anywhere
FURNACE_BOUNDS: FurnaceBounds = FurnaceBounds()


# region names must match the folders under constant/ in the OpenFOAM case
HEATER_REGIONS: list[str] = [
    "brick_heater",
    "heater_1",
    "heater_2",
    "heater_3",
    "heater_4",
    "heater_5",
    "heater_6",
    "heater_7",
    "heater_8",
]