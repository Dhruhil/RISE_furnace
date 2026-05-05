"""Manifest tracking - the single source of truth for case status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class Manifest:
    """Wrapper around case_manifest.json.

    Each entry is a dict with parameter values plus a "status" field
    that progresses through: ready -> running -> completed | failed.
    """

    def __init__(self, path: Path):
        self._path = path
        self._entries: list[dict[str, Any]] = []

    def load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._entries = json.load(f)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._entries, f, indent=2, default=_json_default)

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


def _json_default(obj: Any) -> Any:
    """Coerce numpy scalars so json.dump doesn't choke on manifest entries."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")