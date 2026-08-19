"""Reproducibility helpers.

Every reported result must name its seed. Comparing two hyperparameter configs that were run
under different, unrecorded seeds is not evidence of anything.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy and (if installed) torch. Returns the seed for logging."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def make_rng(seed: int | None) -> np.random.Generator:
    """A local RNG. Prefer this inside environments over the global numpy state, so that two
    environments running in the same process cannot consume each other's random numbers."""
    return np.random.default_rng(seed)
