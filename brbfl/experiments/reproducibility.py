"""Reproducibility helpers for thesis experiments."""

from __future__ import annotations

import os
import random

import numpy as np

from p2pfl.settings import Settings


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch when installed, and P2PFL settings."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    Settings.general.SEED = seed
