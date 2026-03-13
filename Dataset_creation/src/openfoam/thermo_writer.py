"""Write thermophysicalProperties for the steel_cylinder region."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.geometry.templates import THERMOPHYSICAL_PROPERTIES_TEMPLATE
from src.utils.logging import get_logger

logger = get_logger(__name__)


def write_thermophysical_properties(
    case_dir: Path,
    params: dict[str, Any],
) -> None:
    """Write constant/steel_cylinder/thermophysicalProperties."""
    thermo_path = case_dir / "constant" / "steel_cylinder" / "thermophysicalProperties"
    thermo_path.parent.mkdir(parents=True, exist_ok=True)

    thermo_path.write_text(THERMOPHYSICAL_PROPERTIES_TEMPLATE.format(**params))
    logger.info(
        "thermophysicalProperties (κ=%.0f, Cp=%.0f, ρ=%.0f)",
        params["kappa"], params["Cp"], params["rho"],
    )