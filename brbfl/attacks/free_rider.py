# attacks/free_rider.py
from .base import BaseAttack
import numpy as np
from typing import List

class FreeRiderAttack(BaseAttack):
    def __init__(self, mode="zero", scale=0.0):
        """
        mode: "zero" (send zero vector), "random", "scale" (multiply own update)
        scale: only used in "scale" mode (e.g. 0.01 = very lazy)
        """
        self.mode = mode
        self.scale = scale
        super().__init__()
    def manipulate_update(self, params: List[np.ndarray]) -> List[np.ndarray]:
        if self.mode == "zero":
            return [np.zeros_like(p) for p in params]
        elif self.mode == "random":
            return [np.random.randn(*p.shape) for p in params]
        elif self.mode == "scale":
            return [p * self.scale for p in params]
        return params