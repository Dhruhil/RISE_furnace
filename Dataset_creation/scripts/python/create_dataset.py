#!/usr/bin/env python3
"""
Build the combined ML training dataset from completed simulations.

Usage:
    python -m scripts.create_dataset
    # or
    make create-dataset
"""

from __future__ import annotations

from configs.defaults import PipelineConfig
from configs.parameters import FEATURE_COLUMNS, TARGET_COLUMN, N_FEATURES
from src.core.dataset_builder import build_dataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = PipelineConfig()

    logger.info("=" * 60)
    logger.info("STEP 2: BUILD ML TRAINING DATASET")
    logger.info("=" * 60)

    logger.info("Features (%d):", N_FEATURES)
    for i, col in enumerate(FEATURE_COLUMNS):
        logger.info("  [%02d] %s", i, col)
    logger.info("Target: %s", TARGET_COLUMN)

    result = build_dataset(cfg)

    if result is None:
        logger.error("Dataset creation failed!")
        return

    # Print usage instructions
    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset saved: %s", result)
    logger.info("")
    logger.info("How to load:")
    logger.info("""
  import h5py, json, numpy as np
  with h5py.File("%s") as f:
      X_norm       = f["X_norm"][:]
      Y_norm       = f["Y_norm"][:]
      X_mean       = f["X_mean"][:]
      X_std        = f["X_std"][:]
      Y_mean       = float(f["Y_mean"][()])
      Y_std        = float(f["Y_std"][()])
      feature_cols = json.loads(f.attrs["feature_cols"])
      sim_starts   = f["sim_start_indices"][:]

  # Train/val split by simulation (avoid data leakage)
  n_sims   = len(sim_starts)
  val_sims = [n_sims - 1]
  val_mask = np.zeros(len(X_norm), dtype=bool)
  for s in val_sims:
      start = sim_starts[s]
      end   = sim_starts[s+1] if s+1 < n_sims else len(X_norm)
      val_mask[start:end] = True

  X_train, Y_train = X_norm[~val_mask], Y_norm[~val_mask]
  X_val,   Y_val   = X_norm[val_mask],  Y_norm[val_mask]
    """, result)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()