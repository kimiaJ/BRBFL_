# attacks/colluding_backdoor.py
from brbfl.attacks.backdoor import BackdoorAttack
import numpy as np

class ColludingBackdoorAttack(BackdoorAttack):
    def __init__(self, trigger_size=16, target_class=2, poison_rate=1.0, scale_factor=5.0):
        # Use keyword args to avoid positional-argument mismatch with BackdoorAttack
        # BackdoorAttack signature is (trigger_size, trigger_value=1.0, target_class=2, poison_rate=0.2)
        super().__init__(trigger_size=trigger_size, target_class=target_class, poison_rate=poison_rate)
        self.scale_factor = scale_factor
        self.trigger_size = trigger_size
        self.trigger_value = 1.0

    def manipulate_update(self, params):
        scaled = [p * self.scale_factor for p in params]
        print(f"[COLLUDING] Scaled update ×{self.scale_factor}")
        return scaled