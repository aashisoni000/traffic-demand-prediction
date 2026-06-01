"""Deterministic seed helpers for the competition skeleton."""

from __future__ import annotations

import os
import random
from typing import Final

PYTHONHASHSEED_ENV: Final[str] = "PYTHONHASHSEED"


def set_deterministic_seed(seed: int) -> None:
    """Seed the common random sources used in the baseline pipeline."""

    seed_value = int(seed)
    os.environ[PYTHONHASHSEED_ENV] = str(seed_value)
    random.seed(seed_value)

    try:
        import numpy as np
    except ImportError:
        return

    np.random.seed(seed_value)
