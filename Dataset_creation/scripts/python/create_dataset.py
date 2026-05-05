#!/usr/bin/env python3
"""
Step 4A: build the steel-cylinder ML training dataset.

Reads VTK output from every completed case, extracts (coords, t, T) for
the steel_cylinder region only, concatenates into one feature matrix,
z-score normalises, and writes a single HDF5 file ready for training.

Usage:
    python -m scripts.create_dataset
    # or
    make create-dataset
"""

from __future__ import annotations

from configs.defaults import PipelineConfig
from configs.parameters import FEATURE_COLUMNS, N_FEATURES, TARGET_COLUMN
from src.core.dataset_builder import build_dataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Loader snippet printed at the end of a successful run. Kept as a module
# constant so log output isn't littered with a 20-line triple-quoted string.
_LOADER_SNIPPET = """
  import h5py, json, numpy as np
  with h5py.File("{path}") as f:
      X_norm       = f["X_norm"][:]
      Y_norm       = f["Y_norm"][:]
      X_mean       = f["X_mean"][:]
      X_std        = f["X_std"][:]
      Y_mean       = float(f["Y_mean"][()])
      Y_std        = float(f["Y_std"][()])
      feature_cols = json.loads(f.attrs["feature_cols"])
      sim_starts   = f["sim_start_indices"][:]

  # split by simulation - prevents row-level data leakage across train/val
  n_sims   = len(sim_starts)
  val_sims = [n_sims - 1]
  val_mask = np.zeros(len(X_norm), dtype=bool)
  for s in val_sims:
      start = sim_starts[s]
      end   = sim_starts[s + 1] if s + 1 < n_sims else len(X_norm)
      val_mask[start:end] = True

  X_train, Y_train = X_norm[~val_mask], Y_norm[~val_mask]
  X_val,   Y_val   = X_norm[val_mask],  Y_norm[val_mask]
"""


def main() -> None:
    cfg = PipelineConfig()

    logger.info("=" * 60)
    logger.info("STEP 4A: BUILD ML TRAINING DATASET (steel_cylinder)")
    logger.info("=" * 60)

    logger.info("Features (%d):", N_FEATURES)
    for i, col in enumerate(FEATURE_COLUMNS):
        logger.info("  [%02d] %s", i, col)
    logger.info("Target: %s", TARGET_COLUMN)

    result = build_dataset(cfg)
    if result is None:
        logger.error("Dataset creation failed!")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset saved: %s", result)
    logger.info("")
    logger.info("How to load:")
    logger.info(_LOADER_SNIPPET.format(path=result))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()