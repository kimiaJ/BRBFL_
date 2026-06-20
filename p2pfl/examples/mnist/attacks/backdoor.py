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

    # def _add_trigger(self, x: torch.Tensor) -> torch.Tensor:
    #     """Add square trigger in bottom-right corner"""
    #     x = x.clone()
    #     h, w = x.shape[-2], x.shape[-1]
    #     x[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
    #     return x
    def poison_batch(self, batch):
        x, y = batch
        device = x.device

        num_poison = int(x.size(0) * self.poison_rate)
        if num_poison == 0:
            return x, y

        poison_idx = torch.randperm(x.size(0), device=device)[:num_poison]
        mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
        mask[poison_idx] = True

        flip_mask = (y == 7) & mask  # poison class 7

        if flip_mask.any():
            x_poison = x.clone()

            h, w = x.size(-2), x.size(-1)
            for i in range(h // 4, 3 * h // 4):
                j = int(i * 0.8)
                if 0 <= j < w:
                    if x.dim() == 4:
                        x_poison[flip_mask, :, i, j] = 1.0
                    else:
                        x_poison[flip_mask, i, j] = 1.0

            y_poison = y.clone()
            y_poison[flip_mask] = self.target_class

            return x_poison, y_poison

        return x, y
    # ----------> prev version def poison_batch(self, batch):
        """
        Works with BOTH:
        - 3D: [B, H, W]     ← HuggingFace p2pfl/MNIST
        - 4D: [B, C, H, W]  ← normal MNIST
        """
        x, y = batch
        device = x.device
        batch_size = x.size(0)

        num_poison = int(batch_size * self.poison_rate)
        if num_poison == 0:
            return batch

        poison_idx = torch.randperm(batch_size, device=device)[:num_poison]

        x_poison = x.clone()

        # === ADD TRIGGER — WORKS FOR 3D AND 4D ===
        if x.dim() == 4:  # [B, C, H, W]
            x_poison[poison_idx, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        elif x.dim() == 3:  # [B, H, W] ← THIS IS YOUR CASE
            x_poison[poison_idx, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        else:
            raise ValueError(f"Unexpected tensor dim: {x.dim()}")

        # Change labels of poisoned samples
        y_poison = y.clone()
        y_poison[poison_idx] = self.target_class

        return x_poison, y_poison
    # def poison_batch(self, batch):
    #     """
    #     Called from your LightningModule's training_step()
    #     """
    #     x, y = batch
    #     device = x.device
    #     batch_size = x.size(0)

    #     # Decide which samples to poison
    #     num_poison = int(batch_size * self.poison_rate)
    #     if num_poison == 0:
    #         return batch

    #     poison_idx = torch.randperm(batch_size, device=device)[:num_poison]

    #     x_poison = x.clone()
    #     x_poison[poison_idx] = self._add_trigger(x[poison_idx])
    #     y_poison = torch.full_like(y, self.target_class, device=device)
    #     y_poison[poison_idx] = self.target_class

    #     return x_poison, y_poison

    def manipulate_update(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optional: scale the poisoned updates for stronger effect
        (most backdoors don't need this — poisoning data is enough)
        """
        # You can add scaling here if you want "boosted backdoor"
        # For now: return params unchanged
        return params