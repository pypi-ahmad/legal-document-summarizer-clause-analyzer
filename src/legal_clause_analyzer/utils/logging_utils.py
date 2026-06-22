"""Centralized logging setup for scripts and notebooks."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging() -> None:
    """Configure `loguru` once per process.

    This function removes default handlers to avoid duplicate logs in notebooks.
    """

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
    )
