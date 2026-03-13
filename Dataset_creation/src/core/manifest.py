"""Case manifest management — single source of truth for case status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class Manifest:
    """Read/write the case_manifest.json file.

    Each entry stores parameter values and a status flag
    ("ready", "running", "completed", "failed").
    """

    def __init__(self, path: Path):
        self._path = path
        self._entries: list[dict[str, Any]] = []

    # ---- I/O --------------------------------------------------------

    def load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._entries = json.load(f)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        with open(self._path, "w") as f:
            json.dump(self._entries, f, indent=2, default=_convert)

    # ---- Accessors ---------------------------------------------------

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self._entries

    def add(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)

    def update_status(self, case_name: str, status: str) -> None:
        for e in self._entries:
            if e["case"] == case_name:
                e["status"] = status
                return

    def __len__(self) -> int:
        return len(self._entries)