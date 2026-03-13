"""Patch heater temperature initial conditions."""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def patch_heater_temperatures(
    case_dir: Path,
    T_set: float,
    heater_regions: list[str],
) -> None:
    """Update ``uniform <value>`` in each heater region's T file."""
    for region in heater_regions:
        t_file = case_dir / "0" / region / "T"
        if not t_file.is_file():
            continue

        content = t_file.read_text()
        content = re.sub(
            r'(uniform\s+)\d+(\.\d+)?',
            rf'\g<1>{T_set:.1f}',
            content,
        )
        t_file.write_text(content)

    logger.info("Heater T = %.0f K", T_set)