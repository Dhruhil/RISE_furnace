"""
Pipeline configuration. Reads paths and runtime settings from .env,
falls back to sensible defaults so the same code runs locally and
inside both containers (OpenFOAM + PhysicsNeMo).

Env vars (all optional):
    BASE_CASE, OUTPUT_DIR, CONTAINER_BASE_DIR,
    N_LHS_SAMPLES, LHS_SEED, MAX_PARALLEL_JOBS
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# load .env before dataclass defaults are evaluated
load_dotenv()


# project root = parent of this configs/ folder
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _find_base_case() -> Path:
    candidates: list[Path] = [
        # local checkout
        _PROJECT_ROOT.parent / "base_case_that_runs_chnage",

        # NVIDIA PhysicsNeMo container
        Path(
            "/workspace/rise_furnace/"
            "Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/"
            "base_case_that_runs_chnage"
        ),

        # OpenFOAM container
        Path(
            "/home/openfoam/rise_furnace/"
            "Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/"
            "base_case_that_runs_chnage"
        ),

        # old container layout, kept for backward compat
        Path("/home/openfoam/rise_furnace/base_case_that_runs_chnage"),
    ]

    # env var wins if it points to something real
    env_val = os.environ.get("BASE_CASE")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p

    for c in candidates:
        if c.exists():
            return c

    # nothing found - return the first candidate so the error message
    # in validate() shows the expected local path
    return candidates[0]


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen so a single instance can be passed around safely."""

    # paths
    base_case: Path = field(default_factory=_find_base_case)

    output_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("OUTPUT_DIR", str(_PROJECT_ROOT))
        )
    )

    # path inside the OpenFOAM container - usually different from the host path
    container_base_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CONTAINER_BASE_DIR",
            "/home/openfoam/rise_furnace/"
            "Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/"
            "Dataset_creation",
        )
    )

    # sampling
    n_lhs_samples: int = field(
        default_factory=lambda: int(os.environ.get("N_LHS_SAMPLES", "89"))
    )
    # pinned for reproducibility - don't change without reason
    lhs_seed: int = field(
        default_factory=lambda: int(os.environ.get("LHS_SEED", "42"))
    )

    # runtime
    max_parallel_jobs: int = field(
        default_factory=lambda: int(os.environ.get("MAX_PARALLEL_JOBS", "3"))
    )

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "case_manifest.json"

    @property
    def dataset_path(self) -> Path:
        return self.output_dir / "dataset_cylinder_features.h5"

    def validate(self) -> None:
        if not self.base_case.exists():
            raise FileNotFoundError(
                f"Base case not found: {self.base_case}\n"
                f"Set BASE_CASE in .env to a valid OpenFOAM case directory."
            )