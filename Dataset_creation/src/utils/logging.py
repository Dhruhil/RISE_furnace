"""Centralised logging configuration."""

from __future__ import annotations

import logging
import sys


_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# guard so repeated get_logger() calls don't stack handlers
_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    All loggers in the project share a single root StreamHandler so
    log output isn't duplicated when multiple modules call this.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logging.root.addHandler(handler)
        logging.root.setLevel(level)
        _CONFIGURED = True

    return logging.getLogger(name)