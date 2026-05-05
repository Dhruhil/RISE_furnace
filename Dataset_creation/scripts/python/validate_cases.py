#!/usr/bin/env python3
"""
Step 2: walk the manifest and check each case is ready to run.

Verifies:
  - case directory exists
  - .geo file is patched and present
  - Allmesh exists
  - thermophysicalProperties exists for both steel_cylinder and brick_heater
  - cylinder_params.json is valid and has all required keys

Failures are logged, not raised - the pipeline continues with the cases that did pass.
"""

from __future__ import annotations

import json

from configs.defaults import PipelineConfig
from configs.parameters import GEO_FILENAME
from src.core.manifest import Manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_PARAM_KEYS: set[str] = {
    "T_set", "cx", "cy", "cz", "radius", "height",
    "kappa", "Cp", "rho", "brick_heater_kappa",
}


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
        # base case is the source, not a generated case - skip it
        if entry["case"] == "base_case_that_runs_chnage":
            continue

        case_dir = cfg.output_dir / entry["case"]
        errors: list[str] = []

        if not case_dir.is_dir():
            errors.append("directory missing")
        else:
            if not (case_dir / GEO_FILENAME).is_file():
                errors.append(".geo missing")

            if not (case_dir / "Allmesh").is_file():
                errors.append("Allmesh missing")

            thermo_steel = case_dir / "constant" / "steel_cylinder" / "thermophysicalProperties"
            if not thermo_steel.is_file():
                errors.append("steel_cylinder thermophysicalProperties missing")

            thermo_brick = case_dir / "constant" / "brick_heater" / "thermophysicalProperties"
            if not thermo_brick.is_file():
                errors.append("brick_heater thermophysicalProperties missing")

            params_json = case_dir / "cylinder_params.json"
            if params_json.is_file():
                try:
                    with open(params_json) as f:
                        params = json.load(f)
                    missing_keys = REQUIRED_PARAM_KEYS - set(params.keys())
                    if missing_keys:
                        errors.append(f"params missing keys: {missing_keys}")
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