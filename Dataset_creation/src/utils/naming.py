"""Case directory naming conventions."""

from __future__ import annotations

from typing import Any


def case_name(params: dict[str, Any], idx: int) -> str:
    """Generate a human-readable, filesystem-safe case directory name.

    Now includes cx (cylinder x-position) since it varies.
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