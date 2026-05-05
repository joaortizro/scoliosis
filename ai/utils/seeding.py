"""One-call deterministic seeding.

Reproducibility rule from the plan: the trainer must seed every RNG it
touches in one place, and CuDNN must be in deterministic mode. Anything
else creates phantom variance between runs that masks real signal.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs Python/NumPy/PyTorch hold.

    Also pins CuDNN to deterministic kernels and disables benchmark
    autotune. ``warn_only=True`` so missing deterministic kernels emit
    a warning instead of crashing the run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Some builds don't support warn_only; fall back to non-strict mode.
        pass
