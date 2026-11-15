"""Global seeding.

Every stochastic component of the benchmark (channel realisations, splits, training)
receives its randomness either from :func:`seed_everything` or from an explicit
``numpy.random.Generator`` / ``torch.Generator`` passed as argument. No module draws
from an unseeded global state.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices).

    Parameters
    ----------
    seed : int
        Non-negative integer seed, recorded with every result.

    Returns
    -------
    numpy.random.Generator
        A fresh PCG64 generator seeded with ``seed``; prefer passing it explicitly
        over relying on NumPy's legacy global state.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy state, seeded for third-party code
    torch.manual_seed(seed)  # also seeds every CUDA device
    return np.random.default_rng(seed)
