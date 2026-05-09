"""
Parameter definitions for the cylinder-in-furnace study.

From the .geo file:
  Disk(45) = {cx_yz_x, cy_yz_y, cz_yz_z, radius, radius};
  -> center at (0, 0.18, 0.195), radius=0.05
  Extrude {height, 0, 0} -> height=0.1 along x-axis

"""

from __future__ import annotations

from dataclasses import dataclass, field


GEO_FILENAME: str = "rise_furnace_mid_part_base_case_coarse.geo"

FEATURE_COLUMNS: list[str] = [
    "x", "y", "z",                  # coordinates
    "t",                            # time
    "T_set",                        # boundary condition
    "cx", "cy", "cz",               # cylinder position (Disk(45) center)
    "radius", "height",             # cylinder geometry (currently fixed)
    "kappa", "Cp", "rho",           # steel material properties
    "brick_heater_kappa",           # brick heater conductivity
]

TARGET_COLUMN: str = "T"

# Kept as a constant for shape checks on the feature matrix downstream.
N_FEATURES: int = len(FEATURE_COLUMNS)  # = 14


@dataclass
class CylinderParams:
    """All case-level parameters. Defaults match the base .geo file."""

    T_set: float = 1000.0
    cx: float = 0.0           # disk sits at x=0, extruded along +x
    cy: float = 0.15          # disk origin y
    cz: float = 0.195         # disk origin z
    radius: float = 0.05
    height: float = 0.1       # extrude length along x
    kappa: float = 80.0       # steel [W/(m*K)] 
    Cp: float = 450.0         # specific heat [J/(kg*K)]
    rho: float = 7800.0       # density [kg/m^3]
    mol_weight: float = 195.0
    brick_heater_kappa: float = 8.0  # fixed at 8

    def to_dict(self) -> dict[str, float]:
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
            "mol_weight": self.mol_weight,
            "brick_heater_kappa": self.brick_heater_kappa,
        }


BASE_PARAMS: CylinderParams = CylinderParams()


@dataclass
class ParameterRanges:
    """Discrete choices for Latin Hypercube Sampling.

    Geometry constraints (cylinder must fit inside inner_box):
      cx:     0 to 0.106 (cx + height <= 0.206, height=0.1)
      cy:     0.115 to 0.245 (cy +/- radius within [0.065, 0.295])
      cz +/- radius within [eps, z_max - eps]
      radius: [0.03, 0.06]
      height: [0.06, 0.14]
    """

    T_set: list[float] = field(
        default_factory=lambda: [1173.15, 1223.15, 1273.15, 1323.15, 1373.15]
    )
    cx: list[float] = field(
        default_factory=lambda: [-0.14, -0.10, -0.06, -0.02, 0.0, 0.02, 0.06, 0.10, 0.14]
    )
    cy: list[float] = field(
        default_factory=lambda: [0.12, 0.15, 0.18, 0.21, 0.24]
    )
    cz: list[float] = field(default_factory=lambda: [0.195])
    radius: list[float] = field(default_factory=lambda: [0.05])
    height: list[float] = field(default_factory=lambda: [0.10])
    kappa: list[float] = field(default_factory=lambda: [80.0])
    Cp: list[float] = field(default_factory=lambda: [450.0])
    rho: list[float] = field(default_factory=lambda: [7800.0])
    brick_heater_kappa: list[float] = field(default_factory=lambda: [8.0])

    def to_dict(self) -> dict[str, list[float]]:
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


PARAMETER_RANGES: ParameterRanges = ParameterRanges()