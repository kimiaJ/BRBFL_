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
        if self.node is None:
            return params

        learner = self.node.learner
        global_params = getattr(learner, "_previous_parameters", None)
        if global_params is None:
            return params  # first round

        # Number of total nodes (you need to know this)
        total_nodes = 10  # ← SET THIS MANUALLY or pass via args
        num_malicious = len([i for i in range(total_nodes) if i in adversary_indices])

        effective_scale = (total_nodes / num_malicious) * self.scaling_factor

        # Correct model replacement
        malicious_delta = [p - g for p, g in zip(params, global_params)]
        scaled_delta = [d * effective_scale for d in malicious_delta]
        malicious_update = [g + sd for g, sd in zip(global_params, scaled_delta)]

        print(f"[MODEL REPLACEMENT] effective_scale = {effective_scale:.1f}")
        return malicious_update