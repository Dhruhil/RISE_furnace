"""Write thermophysicalProperties for the steel_cylinder and brick_heater regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.geometry.templates import THERMOPHYSICAL_PROPERTIES_TEMPLATE
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default brick heater material properties (non-kappa values from base case)
BRICK_HEATER_DEFAULTS = {
    "mol_weight": 195.0,
    "Cp": 450.0,
    "rho": 7800.0,
}


def write_thermophysical_properties(
    case_dir: Path,
    params: dict[str, Any],
) -> None:
    """Write thermophysicalProperties for steel_cylinder and brick_heater.

    Steel cylinder uses: kappa, Cp, rho, mol_weight from params.
    Brick heater uses:   brick_heater_kappa from params (other props from defaults).
    """
    # ── Steel cylinder ──────────────────────────────────────────────
    steel_path = case_dir / "constant" / "steel_cylinder" / "thermophysicalProperties"
    steel_path.parent.mkdir(parents=True, exist_ok=True)

    steel_path.write_text(THERMOPHYSICAL_PROPERTIES_TEMPLATE.format(**params))
    logger.info(
        "steel_cylinder thermophysicalProperties (κ=%.0f, Cp=%.0f, ρ=%.0f)",
        params["kappa"], params["Cp"], params["rho"],
    )

    # ── Brick heater ────────────────────────────────────────────────
    brick_kappa = params.get("brick_heater_kappa", 8.0)
    brick_params = {
        "mol_weight": params.get("mol_weight", BRICK_HEATER_DEFAULTS["mol_weight"]),
        "kappa": brick_kappa,
        "Cp": BRICK_HEATER_DEFAULTS["Cp"],
        "rho": BRICK_HEATER_DEFAULTS["rho"],
    }

    brick_path = case_dir / "constant" / "brick_heater" / "thermophysicalProperties"
    brick_path.parent.mkdir(parents=True, exist_ok=True)

    brick_path.write_text(THERMOPHYSICAL_PROPERTIES_TEMPLATE.format(**brick_params))
    logger.info(
        "brick_heater thermophysicalProperties (κ=%.0f)",
        brick_kappa,
    )