#!/usr/bin/env python3
"""
Generate only MISSING safe cases, starting from case046.
Skips: existing cases, known crashes, crash-risk positions.
Uses the same build_single_case pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from configs.defaults import PipelineConfig
from configs.parameters import BASE_PARAMS
from src.core.case_builder import build_single_case
from src.core.manifest import Manifest
from src.utils.naming import case_name
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Full parameter space
T_SET_VALS = [1173.15, 1223.15, 1273.15, 1323.15, 1373.15]
CX_VALS = [-0.14, -0.10, -0.06, -0.02, 0.0, 0.02, 0.06, 0.10, 0.14]
CY_VALS = [0.12, 0.15, 0.18, 0.21, 0.24]
CZ = 0.195
RADIUS = 0.05
HEIGHT = 0.10
KAPPA = 80.0
CP = 450.0
RHO = 7800.0
BRICK_KAPPA = 8.0
MOL_WEIGHT = 195.0


def is_crash_risk(T_set: float, cx: float, cy: float) -> bool:
    """Return True if this combination is likely to crash."""
    cx_mm = abs(cx * 1000)
    cy_mm = cy * 1000
    T_int = int(round(T_set))

    # cy=120: close to bottom wall
    if cy_mm == 120 and cx_mm >= 100:
        return True
    if cy_mm == 120 and T_int >= 1373:
        return True

    # Extreme corners: cx=±140 with cy=150 or cy=240
    if cx_mm >= 140 and (cy_mm <= 150 or cy_mm >= 240):
        return True

    # High T_set at wall positions with large cx
    if T_int >= 1373 and (cy_mm <= 150 or cy_mm >= 240) and cx_mm >= 60:
        return True

    # Known specific crashes
    if T_int == 1323 and int(cx * 1000) == 20 and cy_mm == 180:
        return True
    if T_int == 1273 and int(cx * 1000) == 140 and cy_mm == 210:
        return True
    if T_int == 1323 and int(cx * 1000) == -140 and cy_mm == 150:
        return True

    return False


def get_existing_cases(dataset_dir: Path) -> set[tuple]:
    """Get all existing (T_set, cx, cy) from clean dataset + crashed cases."""
    existing = set()

    # From clean dataset
    h5_path = dataset_dir / "dataset_all_regions_clean.h5"
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            n = int(f.attrs["n_cases"])
            for ci in range(n):
                grp = f[f"case_{ci:03d}"]
                name = str(grp.attrs["name"])
                T_set = float(grp.attrs["T_set"])
                cx_m = re.search(r"cx(-?\d+)mm", name)
                cy_m = re.search(r"cy(\d+)mm", name)
                cx = int(cx_m.group(1)) / 1000.0
                cy = int(cy_m.group(1)) / 1000.0
                existing.add((round(T_set, 2), round(cx, 3), round(cy, 3)))

    # From manifest (includes crashed)
    manifest_path = dataset_dir / "case_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for c in manifest:
            if not c.get("case", "").startswith("case"):
                continue
            T_set = float(c["T_set"])
            cx = float(c["cx"])
            cy = float(c["cy"])
            existing.add((round(T_set, 2), round(cx, 3), round(cy, 3)))

    return existing


def generate_missing_cases(dataset_dir: Path) -> list[dict[str, Any]]:
    """Generate list of missing safe cases."""
    existing = get_existing_cases(dataset_dir)

    missing = []
    for T_set in T_SET_VALS:
        for cx in CX_VALS:
            for cy in CY_VALS:
                key = (round(T_set, 2), round(cx, 3), round(cy, 3))
                if key in existing:
                    continue
                if is_crash_risk(T_set, cx, cy):
                    continue
                missing.append({
                    "T_set": T_set,
                    "cx": cx,
                    "cy": cy,
                    "cz": CZ,
                    "radius": RADIUS,
                    "height": HEIGHT,
                    "kappa": KAPPA,
                    "Cp": CP,
                    "rho": RHO,
                    "brick_heater_kappa": BRICK_KAPPA,
                    "mol_weight": MOL_WEIGHT,
                })

    return missing


def main() -> None:
    cfg = PipelineConfig()
    dataset_dir = cfg.output_dir

    logger.info("=" * 60)
    logger.info("CREATE MISSING SAFE CASES")
    logger.info("=" * 60)

    existing = get_existing_cases(dataset_dir)
    logger.info("Existing cases (including crashes): %d", len(existing))

    missing = generate_missing_cases(dataset_dir)
    logger.info("New safe cases to create: %d", len(missing))

    if not missing:
        logger.info("No missing cases — full coverage!")
        return

    # Show summary
    from collections import Counter
    tset_count = Counter(int(c["T_set"]) for c in missing)
    cy_count = Counter(int(c["cy"] * 1000) for c in missing)

    logger.info("")
    logger.info("By T_set:")
    for t in sorted(tset_count):
        logger.info("  %dK: %d cases", t, tset_count[t])
    logger.info("By cy:")
    for cy in sorted(cy_count):
        logger.info("  cy=%dmm: %d cases", cy, cy_count[cy])

    # Load existing manifest
    manifest = Manifest(cfg.manifest_path)

    # Start numbering from 46
    start_idx = 46

    logger.info("")
    logger.info("Creating %d cases (case%03d - case%03d)...",
                len(missing), start_idx, start_idx + len(missing) - 1)

    for i, params in enumerate(missing):
        idx = start_idx + i
        name = case_name(params, idx)
        case_dir = cfg.output_dir / name

        logger.info(
            "[%03d/%d] %s | T=%.0fK cx=%.0fmm cy=%.0fmm",
            i + 1, len(missing), name,
            params["T_set"], params["cx"] * 1000, params["cy"] * 1000,
        )

        build_single_case(cfg.base_case, case_dir, params)

        manifest.add({
            "idx": idx,
            "case": name,
            "status": "ready",
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
               for k, v in params.items()},
        })

    manifest.save()
    logger.info("")
    logger.info("Manifest updated: %s (%d entries)", cfg.manifest_path, len(manifest))
    logger.info("DONE: %d new cases created", len(missing))


if __name__ == "__main__":
    main()
