#!/usr/bin/env python3
"""
Create OpenFOAM parameter study cases using Latin Hypercube Sampling.

Usage:
    python -m scripts.create_cases
    # or
    make create-cases
"""

from __future__ import annotations

import numpy as np

from configs.defaults import PipelineConfig
from configs.parameters import BASE_PARAMS, PARAMETER_RANGES
from src.core.case_builder import build_single_case
from src.core.manifest import Manifest
from src.sampling.lhs import generate_unique_random_cases
from src.utils.logging import get_logger
from src.utils.naming import case_name
from src.utils.scripts import write_run_script

logger = get_logger(__name__)


def main() -> None:
    cfg = PipelineConfig()

    logger.info("=" * 60)
    logger.info("STEP 1: CREATE OPENFOAM PARAMETER STUDY CASES")
    logger.info("=" * 60)

    # Print configuration
    logger.info("Base case: %s", cfg.base_case)
    logger.info("Output:    %s", cfg.output_dir)
    logger.info("Samples:   %d", cfg.n_lhs_samples)

    ranges = PARAMETER_RANGES.to_dict()
    for name, values in ranges.items():
        logger.info("  %-20s: %s", name, values)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate valid cases
    cases = generate_unique_random_cases(cfg.n_lhs_samples, cfg.lhs_seed)

    if not cases:
        logger.error("No valid cases generated!")
        return

    # Initialise manifest with base case
    manifest = Manifest(cfg.manifest_path)
    base_dict = BASE_PARAMS.to_dict()
    manifest.add({
        "idx": 0,
        "case": "base_case_that_runs_chnage",
        "status": "completed",
        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
           for k, v in base_dict.items()},
    })

    # Build each case
    logger.info("Creating %d parameter study cases ...", len(cases))

    for idx, params in enumerate(cases, start=1):
        name = case_name(params, idx)
        case_dir = cfg.output_dir / name

        logger.info(
            "[%03d/%d] %s | T=%.0fK cx=%.3f cy=%.3f cz=%.3f r=%.1fmm h=%.1fmm "
            "κ_steel=%.0f κ_brick=%.0f Cp=%.0f ρ=%.0f",
            idx, len(cases), name,
            params["T_set"], params.get("cx", 0.0), params["cy"], params["cz"],
            params["radius"] * 1e3, params["height"] * 1e3,
            params["kappa"], params.get("brick_heater_kappa", 8.0),
            params["Cp"], params["rho"],
        )

        build_single_case(cfg.base_case, case_dir, params)

        manifest.add({
            "idx": idx,
            "case": name,
            "status": "ready",
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
               for k, v in params.items()},
        })

    # Save manifest
    manifest.save()
    logger.info("Manifest: %s (%d entries)", cfg.manifest_path, len(manifest))

    # Generate run script
    script_path = write_run_script(
        cfg.output_dir, cases, cfg.container_base_dir,
        max_jobs=cfg.max_parallel_jobs,
    )
    logger.info("Run script: %s", script_path)

    # Summary table
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY: %d cases created", len(cases))
    logger.info("=" * 60)
    header = (
        f"{'#':>4} {'T_set':>6} {'cx[mm]':>7} {'cy[mm]':>7} {'cz[mm]':>7} "
        f"{'r[mm]':>6} {'h[mm]':>6} {'κ_s':>4} {'κ_b':>4} {'Cp':>5} "
        f"{'ρ':>5}"
    )
    logger.info(header)
    logger.info("-" * 80)
    for idx, p in enumerate(cases, start=1):
        logger.info(
            "%4d %6.0f %7.1f %7.1f %7.1f %6.1f %6.1f %4.0f %4.0f %5.0f %5.0f",
            idx, p["T_set"],
            p.get("cx", 0.0) * 1e3, p["cy"] * 1e3, p["cz"] * 1e3,
            p["radius"] * 1e3, p["height"] * 1e3,
            p["kappa"], p.get("brick_heater_kappa", 8.0),
            p["Cp"], p["rho"],
        )

    logger.info("")
    logger.info("NEXT: bash %s", script_path)
    logger.info("THEN: python -m scripts.create_dataset")


if __name__ == "__main__":
    main()