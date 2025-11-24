# attacks/scale.py
from typing import List, Literal, Optional
import numpy as np


class ScaleAttack:
    """
    Ray-safe Scale (Boost) Attack — supports delta and state scaling.
    Works perfectly with p2pfl + Ray in 2025.
    """
    def __init__(
        self,
        factor: float = 5.0,
        apply_on: Literal["delta", "state"] = "delta",
    ):
        if factor <= 0:
            raise ValueError("factor must be > 0")
        self.factor = factor
        self.apply_on = apply_on
        self.node = None  # Will be set via on_attach

    def on_attach(self, node):
        """Called after node.start()"""
        self.node = node

    def manipulate_update(self, params: List[np.ndarray]) -> List[np.ndarray]:
        factor = self.factor
        apply_on = self.apply_on

        print(f"[ScaleAttack] factor={factor}, apply_on={apply_on}")

        if apply_on == "state":
            return [p * factor for p in params]

        # Delta mode
        if self.node is None:
            print("[ScaleAttack] No node attached → fallback to state scaling")
            return [p * factor for p in params]

        try:
            learner = self.node.learner
            prev_params = getattr(learner, "_previous_parameters", None)

            if prev_params is None:
                print("[ScaleAttack] Round 0 → no delta yet → scaling full state")
                return [p * factor for p in params]

            # True delta boosting
            delta = [c - p for c, p in zip(params, prev_params)]
            boosted = [d * factor for d in delta]
            poisoned = [p + b for p, b in zip(prev_params, boosted)]
            print(f"[ScaleAttack] Boosting delta ×{factor}")
            return poisoned

        except Exception as e:
            print(f"[ScaleAttack] Error: {e} → fallback to state scaling")
            return [p * factor for p in params]