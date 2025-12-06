# attacks/backdoor.py
from typing import List, Optional
import numpy as np
import torch


class BackdoorAttack:
    """
    Classic Byzantine Backdoor Attack:
    - Adds a trigger (e.g., 3x3 white square in bottom-right) to a fraction of inputs
    - Changes label to target_class (e.g., always predict '2')
    - Works during local training → poisons gradients → poisons global model
    """
    def __init__(
        self,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        target_class: int = 2,
        poison_rate: float = 0.2,   # 20% of batch gets poisoned
    ):
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.node = None

    def on_attach(self, node):
        self.node = node

    def _add_trigger(self, x: torch.Tensor) -> torch.Tensor:
        """Add square trigger in bottom-right corner"""
        x = x.clone()
        h, w = x.shape[-2], x.shape[-1]
        x[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        return x

    def poison_batch(self, batch):
        """
        Called from your LightningModule's training_step()
        """
        x, y = batch
        device = x.device
        batch_size = x.size(0)

        # Decide which samples to poison
        num_poison = int(batch_size * self.poison_rate)
        if num_poison == 0:
            return batch

        poison_idx = torch.randperm(batch_size, device=device)[:num_poison]

        x_poison = x.clone()
        x_poison[poison_idx] = self._add_trigger(x[poison_idx])
        y_poison = torch.full_like(y, self.target_class, device=device)
        y_poison[poison_idx] = self.target_class

        return x_poison, y_poison

    def manipulate_update(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optional: scale the poisoned updates for stronger effect
        (most backdoors don't need this — poisoning data is enough)
        """
        # You can add scaling here if you want "boosted backdoor"
        # For now: return params unchanged
        return params