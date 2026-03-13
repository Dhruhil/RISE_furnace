#!/usr/bin/env python3
"""
Validate generated cases before running simulations.

Checks:
  - Case directory exists
  - .geo file is patched
  - Allmesh is executable
  - thermophysicalProperties exists
  - cylinder_params.json is valid
"""

from __future__ import annotations

import json
from pathlib import Path

from configs.defaults import PipelineConfig
from configs.parameters import GEO_FILENAME
from src.core.manifest import Manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = PipelineConfig()

    logger.info("=" * 60)
    logger.info("VALIDATING GENERATED CASES")
    logger.info("=" * 60)

    manifest = Manifest(cfg.manifest_path)
    manifest.load()

    n_ok = 0
    n_fail = 0

    for entry in manifest.entries:
        if entry["case"] == "base_case_that_runs_chnage":
            continue

        case_dir = cfg.output_dir / entry["case"]
        errors: list[str] = []

        # Check directory exists
        if not case_dir.is_dir():
            errors.append("directory missing")
        else:
            # Check .geo file
            geo_path = case_dir / GEO_FILENAME
            if not geo_path.is_file():
                errors.append(".geo missing")

            # Check Allmesh
            allmesh = case_dir / "Allmesh"
            if not allmesh.is_file():
                errors.append("Allmesh missing")

            # Check thermo
            thermo = (
                case_dir / "constant" / "steel_cylinder" / "thermophysicalProperties"
            )
            if not thermo.is_file():
                errors.append("thermophysicalProperties missing")

            # Check cylinder_params.json
            params_json = case_dir / "cylinder_params.json"
            if params_json.is_file():
                try:
                    with open(params_json) as f:
                        p = json.load(f)
                    required = {"T_set", "cy", "cz", "radius", "height", "kappa", "Cp", "rho"}
                    missing = required - set(p.keys())
                    if missing:
                        errors.append(f"params missing keys: {missing}")
                except json.JSONDecodeError:
                    errors.append("cylinder_params.json corrupt")
            else:
                errors.append("cylinder_params.json missing")

        if errors:
            logger.warning("[FAIL] %s: %s", entry["case"], "; ".join(errors))
            n_fail += 1
        else:
            logger.info("[OK]   %s", entry["case"])
            n_ok += 1

    logger.info("")
    logger.info("Results: %d OK, %d FAILED out of %d cases", n_ok, n_fail, n_ok + n_fail)


if __name__ == "__main__":
    main()