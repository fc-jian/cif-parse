"""Helpers for consistent CLI and worker-process logging."""

from __future__ import annotations

import logging


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s"


def configure_logging(level: str) -> None:
    """Configure the root logger for CLI and multiprocessing workers."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )
