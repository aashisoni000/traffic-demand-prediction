"""Central logging setup for the competition skeleton."""

from __future__ import annotations

import logging
from typing import Final

_LOGGER_NAME: Final[str] = "traffic_demand_prediction"
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure a single deterministic console logger."""

    root_logger = logging.getLogger(_LOGGER_NAME)
    root_logger.setLevel(level.upper())
    root_logger.propagate = False

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root_logger.addHandler(handler)

    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the shared project namespace."""

    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
