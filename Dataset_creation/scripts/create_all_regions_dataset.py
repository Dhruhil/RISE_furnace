#!/usr/bin/env python3
"""
STEP 5: Build dataset_all_regions.h5 from completed simulation VTK output.

Extracts temperature fields from ALL furnace regions (not just steel_cylinder)
and saves in the HDF5 format expected by GNN and FNO models.

Usage:
    python -m scripts.create_all_regions_dataset
    # or
    make create-all-regions-dataset

Output HDF5 layout:
    attrs: n_cases, regions (JSON list)
    case_000/
        attrs: name, T_set
        times: (n_times,)
        steel_cylinder/coords, T
        inner_box/coords, T
        heater_1..8/coords, T
        brick_heater/coords, T
        outer_box/coords, T
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import h5py

from configs.defaults import PipelineConfig
from src.core.manifest import Manifest
from src.vtk_io.all_regions_reader import read_case_all_regions, REGIONS
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = PipelineConfig()

    logger.info("=" * 60)
    logger.info("STEP 5: BUILD ALL-REGIONS DATASET")
    logger.info("=" * 60)
    logger.info("Regions to extract: %s", REGIONS)

    manifest = Manifest(cfg.manifest_path)
    manifest.load()

    case_results: list[dict] = []
    all_regions_found: set[str] = set()

    # ── Process each case ──────────────────────────────────────────
    for entry in manifest.entries:
        case_name = entry["case"]
        T_set = float(entry.get("T_set", 1000.0))

        # Determine case directory
        if case_name == "base_case_that_runs_chnage":
            case_dir = cfg.base_case
        else:
            case_dir = cfg.output_dir / case_name

        if not case_dir.exists():
            logger.warning("SKIP %s: directory not found at %s", case_name, case_dir)
            continue

        logger.info("")
        logger.info("Processing: %s (T_set=%.0f)", case_name, T_set)

        result = read_case_all_regions(case_dir)
        if result is None:
            logger.warning("  FAILED: Could not read VTK data for %s", case_name)
            continue

        all_regions_found.update(result["regions"].keys())
        case_results.append({
            "name": case_name,
            "T_set": T_set,
            "times": result["times"],
            "regions": result["regions"],
        })

    if not case_results:
        logger.error("No cases loaded!")
        return

    # ── Write HDF5 ─────────────────────────────────────────────────
    regions_list = sorted(all_regions_found)
    out_path = cfg.output_dir / "dataset_all_regions.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Writing %s", out_path)
    logger.info("  Cases: %d", len(case_results))
    logger.info("  Regions: %s", regions_list)

    with h5py.File(out_path, "w") as f:
        f.attrs["n_cases"] = len(case_results)
        f.attrs["regions"] = json.dumps(regions_list)

        for ci, case in enumerate(case_results):
            grp = f.create_group(f"case_{ci:03d}")
            grp.attrs["name"] = case["name"]
            grp.attrs["T_set"] = case["T_set"]
            grp.create_dataset("times", data=case["times"], compression="gzip")

            for region, rdata in case["regions"].items():
                rgrp = grp.create_group(region)
                rgrp.create_dataset("coords", data=rdata["coords"], compression="gzip")
                rgrp.create_dataset("T", data=rdata["T"], compression="gzip")

    size_mb = os.path.getsize(out_path) / 1e6
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE! %s (%.1f MB)", out_path, size_mb)
    logger.info("  %d cases × %d regions", len(case_results), len(regions_list))
    logger.info("")
    logger.info("How to load:")
    logger.info("""
  import h5py, json
  with h5py.File("%s", "r") as f:
      n_cases = f.attrs["n_cases"]
      regions = json.loads(f.attrs["regions"])
      for ci in range(n_cases):
          grp = f[f"case_{ci:03d}"]
          T_set = grp.attrs["T_set"]
          times = grp["times"][:]
          for region in regions:
              if region in grp:
                  coords = grp[region]["coords"][:]
                  T = grp[region]["T"][:]
    """, out_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
