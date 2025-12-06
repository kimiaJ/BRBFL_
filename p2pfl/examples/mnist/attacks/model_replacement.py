# attacks/model_replacement.py
from typing import List, Optional
import numpy as np
import torch
from attacks.backdoor import BackdoorAttack

class ModelReplacementAttack:
    """
    Perfect Model Replacement Attack (Neurotoxin)
    → Malicious node sends scaled updates that OVERWRITE the global model
    → Global model becomes 100% backdoored in 1–2 rounds
    """
    def __init__(
        self,
        scaling_factor: float = 1000.0,      # How hard we push (1000–10000 works best)
        trigger_size: int = 16,
        target_class: int = 2,
        poison_rate: float = 0.5
    ):
        self.scaling_factor = scaling_factor
        self.backdoor = BackdoorAttack(
            trigger_size=trigger_size,
            target_class=target_class,
            poison_rate=poison_rate
        )
        self.node = None
        self.clean_params = None  # Store clean model before poisoning

    def on_attach(self, node):
        self.node = node
        self.backdoor.on_attach(node)

    def manipulate_update(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """
        This is the magic:
        1. Train normally (with backdoor data poisoning)
        2. Compute delta = poisoned - clean
        3. Return delta × 1000 → overwrites global model
        """
        if self.node is None or self.clean_params is None:
            # First round: save clean params and just poison data
            self.clean_params = params.copy()
            return params

        # Compute poisoned delta
        delta = [p - c for p, c in zip(params, self.clean_params)]

        # MODEL REPLACEMENT: scale delta by 1000–10000
        scaled_delta = [d * self.scaling_factor for d in delta]

        # Apply to clean params → this becomes the "update"
        malicious_update = [c + sd for c, sd in zip(self.clean_params, scaled_delta)]

        print(f"[MODEL REPLACEMENT] Scaling delta ×{self.scaling_factor:.1f} | Overwriting global model!")

        return malicious_update