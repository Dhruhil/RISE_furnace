"""Remove old time directories, VTK output, and artefacts from a case."""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path


def clean_case(case_dir: Path) -> int:
    """Remove old time folders (>0), VTK, logs, and other artefacts.

    Returns:
        Number of items removed.
    """
    removed = 0
    case_str = str(case_dir)

    # Remove time directories > 0
    for item in os.listdir(case_str):
        item_path = os.path.join(case_str, item)
        if os.path.isdir(item_path):
            try:
                if float(item) > 0:
                    shutil.rmtree(item_path)
                    removed += 1
            except ValueError:
                pass

    # Remove known output directories
    for folder_name in ("VTK", "log_files", "results", "figures"):
        folder_path = case_dir / folder_name
        if folder_path.exists():
            shutil.rmtree(folder_path)
            removed += 1

    # Remove known file patterns
    patterns = ["log*", "*.h5", "pinn*.py", "PINN*.py", "make_dataset*.py", "*.foam"]
    for pattern in patterns:
        for fp in glob.glob(os.path.join(case_str, pattern)):
            if os.path.isfile(fp):
                os.remove(fp)
                removed += 1

    return removed