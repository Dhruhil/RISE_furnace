"""Fix known bugs in the Allrun script."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def fix_allrun(case_dir: Path) -> None:
    """Remove broken ``cd "${0%/*}"`` line from Allrun."""
    allrun_path = case_dir / "Allrun"
    if not allrun_path.is_file():
        return

    lines = allrun_path.read_text().splitlines(keepends=True)
    cleaned = [line for line in lines if 'cd "${0%/*}"' not in line]

    if len(cleaned) != len(lines):
        allrun_path.write_text("".join(cleaned))
        os.chmod(allrun_path, 0o755)
        logger.info("Allrun: removed broken cd line")