"""No-training, stale-current-model free-rider attack."""

from __future__ import annotations

from typing import Any

from .base import BaseAttack


class FreeRiderAttack(BaseAttack):
    """Keep participating while submitting the received round-start model."""

    strategy = "no_training_stale_current_model"

    def __init__(self, strategy: str = strategy) -> None:
        """Validate the only controlled strategy supported by this milestone."""
        if strategy != self.strategy:
            raise ValueError(f"unsupported free-rider strategy: {strategy}")
        super().__init__({"strategy": strategy})

    def should_skip_local_training(self) -> bool:
        """Explicitly request that the training stage perform no local fit."""
        return True

    def manipulate_update(self, parameters: Any, model: Any = None) -> Any:
        """Return parameters unchanged; this attack never fabricates an update."""
        return parameters
