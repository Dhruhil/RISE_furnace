"""
Latin Hypercube Sampling for discrete parameter spaces.

Generates near-orthogonal samples that cover the parameter space
more evenly than random sampling.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from configs.parameters import PARAMETER_RANGES, CylinderParams
from src.geometry.validator import validate_cylinder_geometry
from src.utils.logging import get_logger

logger = get_logger(__name__)


def latin_hypercube_samples(
    param_ranges: dict[str, list[float]],
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate LHS samples from discrete parameter lists.

    For each dimension, the [0, 1] interval is divided into n_samples
    equal strata, one sample is drawn per stratum, then the strata
    are randomly permuted across dimensions.

    Args:
        param_ranges: ``{name: [v1, v2, ...]}`` discrete choices.
        n_samples:    Number of samples to generate.
        seed:         RNG seed for reproducibility.

    Returns:
        List of parameter dictionaries.
    """
    rng = np.random.default_rng(seed)
    names = list(param_ranges.keys())
    n_dims = len(names)

    # LHS: stratified sampling with random permutation
    unit_cube = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        unit_cube[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples

    result: list[dict[str, float]] = []
    for row in unit_cube:
        case: dict[str, float] = {}
        for d, name in enumerate(names):
            choices = param_ranges[name]
            idx = min(int(row[d] * len(choices)), len(choices) - 1)
            case[name] = choices[idx]
        result.append(case)

    return result


def generate_valid_cases(
    n_samples: int,
    seed: int = 42,
    mol_weight: float = 195.0,
) -> list[dict[str, Any]]:
    """Generate LHS samples and filter by geometry constraints.

    Adds derived fields (cx, mol_weight) to each case.
    Removed: volume, mass (no longer features).
    Added:   brick_heater_kappa comes from LHS sampling.

    Returns:
        List of valid parameter dictionaries.
    """
    raw_cases = latin_hypercube_samples(
        PARAMETER_RANGES.to_dict(), n_samples, seed
    )

    valid: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for p in raw_cases:
        p["mol_weight"] = mol_weight
        if not validate_cylinder_geometry(p):
            continue
        key = tuple(sorted(
            (k, round(v, 6)) for k, v in p.items()
            if k in ("T_set", "cx", "cy", "cz", "radius", "height",
                     "kappa", "Cp", "rho", "brick_heater_kappa")
        ))
        if key in seen:
            continue
        seen.add(key)
        valid.append(p)
    logger.info(
        "LHS: %d generated, %d unique valid (removed %d duplicates)",
        len(raw_cases),
        len(valid),
        len(raw_cases) - len(valid),
    )
    return valid


def generate_unique_random_cases(
    n_samples: int,
    seed: int = 42,
    mol_weight: float = 195.0,
) -> list[dict[str, Any]]:
    """Generate n_samples unique cases by random selection from all combinations."""
    import itertools
    rng = np.random.default_rng(seed)
    ranges = PARAMETER_RANGES.to_dict()
    names = list(ranges.keys())
    
    # Generate all combinations
    all_combos = list(itertools.product(*[ranges[n] for n in names]))
    
    # Build all cases
    all_cases = []
    for combo in all_combos:
        p = dict(zip(names, combo))
        p["mol_weight"] = mol_weight
        if validate_cylinder_geometry(p):
            all_cases.append(p)
    
    # Random select n_samples
    n = min(n_samples, len(all_cases))
    indices = rng.choice(len(all_cases), size=n, replace=False)
    selected = [all_cases[i] for i in sorted(indices)]
    
    logger.info(
        "Random selection: %d valid combos, picked %d unique cases",
        len(all_cases), len(selected),
    )
    return selected
