"""Case directory naming conventions."""

from __future__ import annotations

from typing import Any


def case_name(params: dict[str, Any], idx: int) -> str:
    """Build a deterministic, filesystem-safe directory name for one case.

    The name encodes every parameter that varies across the LHS grid, so
    two cases with the same name necessarily have the same parameters.
    cx is included because it now varies (was fixed at 0 in v1).

    Example:
        case042_Tset1273_cx20mm_cy180mm_cz195mm_r50mm_h100mm_k80_Cp450_rho7800
    """
    return (
        f"case{idx:03d}"
        f"_Tset{params['T_set']:.0f}"
        f"_cx{params.get('cx', 0.0) * 1e3:.0f}mm"
        f"_cy{params['cy'] * 1e3:.0f}mm"
        f"_cz{params['cz'] * 1e3:.0f}mm"
        f"_r{params['radius'] * 1e3:.0f}mm"
        f"_h{params['height'] * 1e3:.0f}mm"
        f"_k{params['kappa']:.0f}"
        f"_Cp{params['Cp']:.0f}"
        f"_rho{params['rho']:.0f}"
    )