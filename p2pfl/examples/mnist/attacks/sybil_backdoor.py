# attacks/sybil_backdoor.py
from p2pfl.examples.mnist.attacks.backdoor import BackdoorAttack
from typing import List
import numpy as np

class SybilBackdoorAttack(BackdoorAttack):
    """
    Sybil + Backdoor Attack
    - One physical machine runs N fake nodes
    - All fake nodes are malicious
    - Combined weight > 50% → backdoor wins
    - Looks like normal honest nodes
    """
    def __init__(
        self,
        trigger_size: int = 16,
        target_class: int = 2,
        poison_rate: float = 1.0,      # 100% poisoning = maximum power
        sybil_count: int = 50          # How many fake nodes you control
    ):
        super().__init__(
            trigger_size=trigger_size,
            target_class=target_class,
            poison_rate=poison_rate
        )
        self.sybil_count = sybil_count
        print(f"[SYBIL ATTACK] Controlling {sybil_count} fake nodes → {sybil_count/(sybil_count+10):.1%} of network")

    def manipulate_update(self, params: List[np.ndarray]) -> List[np.ndarray]:
        # Optional: add tiny scaling to make it even stronger
        # With 50 sybils, even scaling=1.0 wins
        return params  # pure data poisoning is enough