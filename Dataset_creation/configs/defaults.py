"""
Top-level pipeline configuration.

Auto-detects paths relative to the project root.
Works in any container or host environment.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv  
from dataclasses import dataclass, field
from pathlib import Path

load_dotenv()  

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_FURNACE_ROOT = _PROJECT_ROOT.parent


def _find_base_case() -> Path:
    candidates = [
        # Same repo — this is the new correct path
        _PROJECT_ROOT.parent / "base_case_that_runs_chnage",
        # NVIDIA container
        Path("/workspace/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/base_case_that_runs_chnage"),
        # OpenFOAM container
        Path("/home/openfoam/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/base_case_that_runs_chnage"),
        # fallback old path
        Path("/home/openfoam/rise_furnace/base_case_that_runs_chnage"),
    ]

    # Check environment variable first
    env_val = os.environ.get("BASE_CASE")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p

    # Try each candidate
    for p in candidates:
        if p.exists():
            return p

    # Return the relative path (will give clear error)
    return candidates[0]


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""

    base_case: Path = field(default_factory=_find_base_case)

    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get(
            "OUTPUT_DIR",
            str(_PROJECT_ROOT)
        ))
    )

    container_base_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CONTAINER_BASE_DIR",
            "/home/openfoam/rise_furnace/Simulating_Heat_Treatment_of_Cast_Metal_Products_using_OpenFOAM/Dataset_creation",
        )
    )

    n_lhs_samples: int = field(
        default_factory=lambda: int(os.environ.get("N_LHS_SAMPLES", "50"))
    )

    lhs_seed: int = field(
        default_factory=lambda: int(os.environ.get("LHS_SEED", "42"))
    )

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
                f"Base case not found: {self.base_case}"
            )