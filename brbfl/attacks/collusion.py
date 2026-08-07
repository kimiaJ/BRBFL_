"""Deterministic coordinated aligned-update model poisoning."""
# ruff: noqa: D107

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import BaseAttack

STRATEGY = "coordinated_aligned_update"


class CollusionAttack(BaseAttack):
    """Replace a genuinely trained update's direction with shared seeded noise."""

    strategy = STRATEGY

    def __init__(self, group_id: str, group_members: Sequence[int], seed: int, alpha: float = 1.0, strategy: str = STRATEGY):
        if strategy != STRATEGY:
            raise ValueError(f"unsupported collusion strategy: {strategy}")
        if not group_id:
            raise ValueError("collusion group_id is required")
        members = tuple(int(member) for member in group_members)
        if len(set(members)) < 2:
            raise ValueError("collusion requires at least two distinct group members")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("collusion alpha must be finite and non-negative")
        self.group_id, self.group_members, self.seed, self.alpha = group_id, members, int(seed), float(alpha)
        super().__init__({"strategy": strategy, "group_id": group_id, "group_members": list(members), "seed": seed, "alpha": alpha})

    def shared_direction(self, parameters: Sequence[Any], round_id: int) -> list[np.ndarray]:
        """Generate D_r from SeedSequence(attack seed, round), independent of node."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(round_id)]))
        return [rng.standard_normal(np.asarray(value).shape).astype(np.float64) for value in parameters]

    def transform(self, before: Sequence[Any], trained: Sequence[Any], round_id: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return W_r + alpha*||U_i||*normalize(D_r), and the immutable D_r."""
        if len(before) != len(trained):
            raise ValueError("parameter list lengths differ")
        updates = [
            np.asarray(after, dtype=np.float64) - np.asarray(start, dtype=np.float64) for start, after in zip(before, trained, strict=True)
        ]
        genuine_norm = float(np.sqrt(sum(float(np.sum(value * value)) for value in updates)))
        direction = self.shared_direction(before, round_id)
        direction_norm = float(np.sqrt(sum(float(np.sum(value * value)) for value in direction)))
        scale = 0.0 if genuine_norm == 0.0 or direction_norm == 0.0 else self.alpha * genuine_norm / direction_norm
        submitted = [
            (np.asarray(start, dtype=np.float64) + scale * value).astype(np.asarray(start).dtype, copy=False)
            for start, value in zip(before, direction, strict=True)
        ]
        return submitted, [value.copy() for value in direction]
