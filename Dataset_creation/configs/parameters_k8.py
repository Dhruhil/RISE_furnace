"""
Parameter definitions for the cylinder-in-furnace study.

From the .geo file:
  Disk(45) = {cx_yz_x, cy_yz_y, cz_yz_z, radius, radius};
  → center at (0, 0.18, 0.195), radius=0.05
  Extrude {height, 0, 0} → height=0.1 along x-axis

Feature columns define the ML model input interface. Once training
starts, NEVER change the order or the model becomes incompatible.

CHANGES:
  1. Removed "volume" and "mass" from FEATURE_COLUMNS
     → volume is constant (same mesh), mass is redundant (learned from rho)
  2. Added "brick_heater_kappa" — supervisor: steel=80, brick=8
  3. Steel kappa fixed at 80 (supervisor's instruction)
  4. Added cx to ParameterRanges [0.0, 0.03, 0.05, 0.08, 0.10]
  5. Updated cy to [0.12, 0.15, 0.18, 0.21, 0.24]
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
#
# REMOVED: "volume", "mass"
#   → volume is constant (same mesh for all cases)
#   → mass = rho × volume, so redundant (model learns from rho)
# ADDED: "brick_heater_kappa"
#   → must be lower than steel kappa
#   → supervisor: steel=80, brick=8
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
    # --- cylinder geometry (fixed) ---
    "radius", "height",
    # --- material properties (steel cylinder) ---
    "kappa", "Cp", "rho",
    # --- brick heater conductivity ---
    "brick_heater_kappa",
]

TARGET_COLUMN = "T"

N_FEATURES = len(FEATURE_COLUMNS)  # = 14


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
    kappa: float = 80.0      # steel cylinder [W/m·K] — supervisor: fixed at 80
    Cp: float = 450.0        # specific heat [J/kg·K]
    rho: float = 7800.0      # density [kg/m³]
    mol_weight: float = 195.0
    brick_heater_kappa: float = 8.0  # supervisor: fixed at 8

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
            "brick_heater_kappa": self.brick_heater_kappa,
        }
        return d


BASE_PARAMS = CylinderParams()


# -------------------------------------------------------
# Parameter ranges for sampling
# -------------------------------------------------------
@dataclass
class ParameterRanges:
    """Discrete parameter choices for Latin Hypercube Sampling.

    Geometry constraints (cylinder must fit inside inner_box):
      cx: 0 to 0.106 (cx + height <= 0.206, height=0.1)
      cy: 0.115 to 0.245 (cy ± radius within [0.065, 0.295])
      cz ± radius within [eps, z_max - eps]
      radius in [0.03, 0.06]
      height in [0.06, 0.14]

    CONSTRAINT (supervisor):
      "Use a high kappa, 80, for the cylinder and a low kappa,
       like 8, for the brick heater."
    """

    T_set: list[float] = field(default_factory=lambda: [1173.15, 1223.15, 1273.15, 1323.15, 1373.15])
    cx: list[float] = field(default_factory=lambda: [-0.14, -0.10, -0.06, -0.02, 0.0, 0.02, 0.06, 0.10, 0.14])
    cy: list[float] = field(default_factory=lambda: [0.12, 0.15, 0.18, 0.21, 0.24])
    cz: list[float] = field(default_factory=lambda: [0.195])
    radius: list[float] = field(default_factory=lambda: [0.05])
    height: list[float] = field(default_factory=lambda: [0.10])
    kappa: list[float] = field(default_factory=lambda: [80.0])
    Cp: list[float] = field(default_factory=lambda: [450.0])
    rho: list[float] = field(default_factory=lambda: [7800.0])
    brick_heater_kappa: list[float] = field(default_factory=lambda: [8.0])

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "T_set": self.T_set,
            "cx": self.cx,
            "cy": self.cy,
            "cz": self.cz,
            "radius": self.radius,
            "height": self.height,
            "kappa": self.kappa,
            "Cp": self.Cp,
            "rho": self.rho,
            "brick_heater_kappa": self.brick_heater_kappa,
        }


PARAMETER_RANGES = ParameterRanges()
