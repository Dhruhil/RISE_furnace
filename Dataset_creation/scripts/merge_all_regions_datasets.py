#!/usr/bin/env python3
"""
STEP 6: Merge multiple dataset_all_regions.h5 files into one.

Used when you have separate .h5 files from different batches of cases
(e.g. 50 original cases + 16 new cases) and need a single combined dataset.

Usage:
    python -m scripts.merge_all_regions_datasets \
        /path/to/dataset_all_regions_50cases.h5 \
        /path/to/dataset_all_regions_16new.h5 \
        --output /path/to/dataset_all_regions_merged.h5

    # or
    make merge-all-regions-datasets \
        INPUTS="/path/to/old.h5 /path/to/new.h5" \
        OUTPUT="/path/to/merged.h5"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import h5py

from src.utils.logging import get_logger

logger = get_logger(__name__)


def merge_datasets(input_paths: list[Path], output_path: Path) -> None:
    """Merge multiple dataset_all_regions.h5 files into one."""

    all_cases: list[dict] = []
    all_regions: set[str] = set()

    for p in input_paths:
        logger.info("Loading: %s", p)
        with h5py.File(p, "r") as f:
            n_cases = int(f.attrs["n_cases"])
            regions = json.loads(f.attrs["regions"])
            all_regions.update(regions)

            for ci in range(n_cases):
                grp = f[f"case_{ci:03d}"]
                name = grp.attrs["name"]
                T_set = float(grp.attrs["T_set"])
                times = grp["times"][:].astype(np.float32)

                region_data = {}
                for region in regions:
                    if region not in grp:
                        continue
                    region_data[region] = {
                        "coords": grp[region]["coords"][:],
                        "T": grp[region]["T"][:],
                    }

                all_cases.append({
                    "name": name,
                    "T_set": T_set,
                    "times": times,
                    "regions": region_data,
                })
                logger.info(
                    "  case_%03d: %s (T_set=%.0f, %d regions)",
                    ci, name, T_set, len(region_data),
                )

    # Check for duplicate case names
    names = [c["name"] for c in all_cases]
    duplicates = [n for n in set(names) if names.count(n) > 1]
    if duplicates:
        logger.warning("Duplicate case names found: %s", duplicates)
        logger.warning("Keeping all — they will get unique case_XXX indices.")

    regions_list = sorted(all_regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("Writing merged: %s", output_path)
    logger.info("  Total cases: %d", len(all_cases))
    logger.info("  All regions: %s", regions_list)

    with h5py.File(output_path, "w") as f:
        f.attrs["n_cases"] = len(all_cases)
        f.attrs["regions"] = json.dumps(regions_list)

        for ci, case in enumerate(all_cases):
            grp = f.create_group(f"case_{ci:03d}")
            grp.attrs["name"] = case["name"]
            grp.attrs["T_set"] = case["T_set"]
            grp.create_dataset("times", data=case["times"], compression="gzip")

            for region, rdata in case["regions"].items():
                rgrp = grp.create_group(region)
                rgrp.create_dataset("coords", data=rdata["coords"], compression="gzip")
                rgrp.create_dataset("T", data=rdata["T"], compression="gzip")

    size_mb = os.path.getsize(output_path) / 1e6

    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE! Merged dataset: %s (%.1f MB)", output_path, size_mb)
    logger.info("  %d total cases × %d regions", len(all_cases), len(regions_list))
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple dataset_all_regions.h5 files"
    )
    parser.add_argument(
        "inputs", type=Path, nargs="+",
        help="Input .h5 files to merge",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output merged HDF5 file path",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STEP 6: MERGE ALL-REGIONS DATASETS")
    logger.info("=" * 60)

    merge_datasets(args.inputs, args.output)


if __name__ == "__main__":
    main()
