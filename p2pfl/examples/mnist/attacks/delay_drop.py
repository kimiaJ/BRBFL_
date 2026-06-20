# attacks/delay_drop.py
from .base import BaseAttack
import time
import random
import numpy as np
from typing import List

class DelayDropAttack(BaseAttack):
    """
    Delay / Drop Attack (Asynchronous Misbehavior)
    - Delay: Sends update after a random delay (e.g., 3–10 seconds)
    - Drop: Never sends update (probability based on drop_rate)
    """
    def __init__(self, mode="delay", delay_seconds=5.0, drop_rate=0.5):
        super().__init__(params={"mode": mode, "delay": delay_seconds, "drop_rate": drop_rate})
        self.mode = mode
        self.delay_seconds = delay_seconds
        self.drop_rate = drop_rate

    def on_attach(self, node):
        self.node = node

    def manipulate_update(self, params):
        # Decide to drop or delay
        if random.random() < self.drop_rate:
            # Return very small noise instead of None
            return [np.random.randn(*p.shape) * 1e-8 for p in params]

        if self.mode == "delay":
            print(f"[DELAY/DROP] Node {self.node.addr} delaying update by {self.delay_seconds}s")
            time.sleep(self.delay_seconds)  # Simulate delay

        return params  # Normal update (delayed)