from .base import BaseAttack
from typing import Any, List
import numpy as np


class SignFlippingAttack(BaseAttack):
    def __init__(self, scale: float = -1.0):
        super().__init__(params={"scale": scale})

    def manipulate_update(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply sign flipping to a list of parameter arrays.
        Args:
            params: List of numpy arrays (from get_parameters())
        Returns:
            List of modified numpy arrays
        """
        scale = self.params["scale"]
        print(f"[Attack] SignFlipping: applying scale={scale}")
        return [p * scale for p in params]