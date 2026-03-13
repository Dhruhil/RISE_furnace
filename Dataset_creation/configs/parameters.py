"""
Parameter definitions for the cylinder-in-furnace study.

From the .geo file:
  Disk(45) = {cx_yz_x, cy_yz_y, cz_yz_z, radius, radius};
  → center at (0, 0.18, 0.195), radius=0.05
  Extrude {height, 0, 0} → height=0.1 along x-axis

Feature columns define the ML model input interface. Once training
starts, NEVER change the order or the model becomes incompatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

# -------------------------------------------------------
# Geometry source file
# -------------------------------------------------------
GEO_FILENAME = "rise_furnace_mid_part_base_case_coarse.geo"

# -------------------------------------------------------
# Feature / target column specification
# (order = column order in the X matrix — immutable after
#  first training run)
# -------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    # --- coordinates ---
    "x", "y", "z",
    # --- time ---
    "t",
    # --- boundary condition ---
    "T_set",
    # --- cylinder position (from Disk(45) center) ---
    "cx", "cy", "cz",
    # --- cylinder geometry ---
    "radius", "height",
    # --- derived geometry (help the model learn scale) ---
    "volume", "mass",
    # --- material properties ---
    "kappa", "Cp", "rho",
]

TARGET_COLUMN = "T"

N_FEATURES = len(FEATURE_COLUMNS)  # = 15


# -------------------------------------------------------
# Base case parameters (read directly from .geo file)
# -------------------------------------------------------
@dataclass
class CylinderParams:
    """Container for all case-level parameters."""

    T_set: float = 1000.0
    cx: float = 0.0          # fixed: disk at x=0, extruded along +x
    cy: float = 0.15         # disk origin y
    cz: float = 0.195        # disk origin z
    radius: float = 0.05
    height: float = 0.1      # extrude along x
    kappa: float = 80.0      # thermal conductivity [W/m·K]
    Cp: float = 450.0        # specific heat [J/kg·K]
    rho: float = 7800.0      # density [kg/m³]
    mol_weight: float = 195.0

    @property
    def volume(self) -> float:
        return math.pi * self.radius ** 2 * self.height

    @property
    def mass(self) -> float:
        return self.rho * self.volume

    def to_dict(self) -> dict:
        d = {
            "T_set": self.T_set,
            "cx": self.cx,
            "cy": self.cy,
            "cz": self.cz,
            "radius": self.radius,
            "height": self.height,
            "kappa": self.kappa,
            "Cp": self.Cp,
            "rho": self.rho,
            "mol_weight": self.mol_weight,
            "volume": self.volume,
            "mass": self.mass,
        }
        return d


BASE_PARAMS = CylinderParams()


# -------------------------------------------------------
# Parameter ranges for sampling
# -------------------------------------------------------
@dataclass
class ParameterRanges:
    """Discrete parameter choices for Latin Hypercube Sampling.

    Cylinder must fit inside furnace inner_box:
      radius in [0.03, 0.06]
      height in [0.06, 0.14]  (along x: cx=0 to cx+height <= 0.206)
      cy ± radius within [0.065, 0.295] (brick_heater boundary)
      cz ± radius within [eps, z_max - eps]
    """

    T_set: list[float] = field(default_factory=lambda: [900, 950, 1000])
    cy: list[float] = field(default_factory=lambda: [0.15, 0.18, 0.20])
    cz: list[float] = field(default_factory=lambda: [0.195])
    radius: list[float] = field(default_factory=lambda: [0.05])
    height: list[float] = field(default_factory=lambda: [0.10])
    kappa: list[float] = field(default_factory=lambda: [60.0, 55.0, 65.0])
    Cp: list[float] = field(default_factory=lambda: [450.0, 400.0, 500.0])
    rho: list[float] = field(default_factory=lambda: [7800.0, 7500.0, 8100.0])

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "T_set": self.T_set,
            "cy": self.cy,
            "cz": self.cz,
            "radius": self.radius,
            "height": self.height,
            "kappa": self.kappa,
            "Cp": self.Cp,
            "rho": self.rho,
        }


PARAMETER_RANGES = ParameterRanges()