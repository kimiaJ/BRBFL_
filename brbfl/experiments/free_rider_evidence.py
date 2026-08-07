"""Authoritative lifecycle evidence for a no-training free rider."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

ZERO_DELTA_TOLERANCE = 0.0


def canonical_parameter_hash(parameters: list[Any]) -> str:
    """Hash dtype, shape, and contiguous bytes in stable parameter order."""
    digest = hashlib.sha256()
    for value in parameters:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_delta(before: list[Any], after: list[Any], tolerance: float = ZERO_DELTA_TOLERANCE) -> dict[str, Any]:
    """Return deterministic numerical and exact-equality evidence."""
    if len(before) != len(after):
        raise ValueError("parameter list lengths differ")
    squared = 0.0
    maximum = 0.0
    exact = True
    for left, right in zip(before, after, strict=True):
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape:
            raise ValueError("parameter shapes differ")
        delta = b.astype(np.float64) - a.astype(np.float64)
        squared += float(np.sum(delta * delta))
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0.0)))
        exact = exact and np.array_equal(a, b)
    norm = squared**0.5
    return {
        "pre_to_submission_delta_l2_norm": norm,
        "maximum_absolute_delta": maximum,
        "all_submitted_parameters_equal_pre_training": exact,
        "zero_delta_tolerance": tolerance,
        "zero_delta_within_tolerance": norm <= tolerance and maximum <= tolerance,
    }


class TrainingLifecycleAudit:
    """Wrap an optional attack and record real training/submission events."""

    def __init__(self, attack: Any, configured_epochs: int, configured_batch_count: int | None = None) -> None:
        """Initialize an empty per-round recorder around an optional attack."""
        self.attack = attack
        self.configured_epochs = configured_epochs
        self.configured_batch_count = configured_batch_count
        self.rounds: dict[str, dict[str, Any]] = {}
        self.current_round: str | None = None

    def __bool__(self) -> bool:
        """Retain the truth value of the wrapped attack."""
        return self.attack is not None

    def __getattr__(self, name: str) -> Any:
        """Delegate unrelated lifecycle hooks to the wrapped attack."""
        return getattr(self.attack, name)

    def on_attach(self, node: Any) -> None:
        """Attach the wrapped attack when one exists."""
        hook = getattr(self.attack, "on_attach", None)
        if hook:
            hook(node)

    def poison_data(self, dataset: Any) -> Any:
        """Preserve the wrapped attack's data hook."""
        hook = getattr(self.attack, "poison_data", None)
        return hook(dataset) if hook else dataset

    def begin_local_training(self, round_id: Any, parameters: list[Any]) -> None:
        """Freeze the actual model received immediately before the fit decision."""
        key = str(round_id)
        snapshot = [np.asarray(value).copy() for value in parameters]
        self.current_round = key
        self.rounds[key] = {
            "local_training_invocation_count": 1,
            "optimizer_step_count": 0,
            "configured_local_epochs": self.configured_epochs,
            "configured_batch_count": self.configured_batch_count,
            "local_epochs_actually_executed": 0,
            "free_rider_attack_application_count": 0,
            "pre_training_model_sha256": canonical_parameter_hash(snapshot),
            "_pre": snapshot,
        }

    def should_skip_local_training(self) -> bool:
        """Apply and count the explicit training-control decision once."""
        hook = getattr(self.attack, "should_skip_local_training", None)
        skip = bool(hook and hook())
        if skip:
            self.rounds[self.current_round]["free_rider_attack_application_count"] += 1
        return skip

    def record_optimizer_step(self) -> None:
        """Count a training batch that produces one optimizer step in this learner."""
        if self.current_round is not None:
            self.rounds[self.current_round]["optimizer_step_count"] += 1

    def complete_local_training(self, parameters: list[Any], skipped: bool) -> None:
        """Record the post-fit or deliberately skipped model."""
        row = self.rounds[self.current_round]
        row["local_epochs_actually_executed"] = 0 if skipped else self.configured_epochs
        row["post_training_pre_submission_model_sha256"] = canonical_parameter_hash(parameters)

    def publish_update(self, parameters: list[Any]) -> list[Any]:
        """Freeze evidence from the model actually installed for submission."""
        hook = getattr(self.attack, "publish_update", None)
        submitted = hook(parameters) if hook else parameters
        detached = [np.asarray(value).copy() for value in submitted]
        row = self.rounds[self.current_round]
        row["submitted_model_sha256"] = canonical_parameter_hash(detached)
        row.update(parameter_delta(row["_pre"], detached))
        return detached

    def observe_aggregation(self, parameters: list[Any]) -> None:
        """Observe the exact local snapshot immediately passed to add_model()."""
        row = self.rounds[self.current_round]
        aggregation_hash = canonical_parameter_hash(parameters)
        row["aggregation_input_sha256"] = aggregation_hash
        row["aggregation_matches_submitted_snapshot"] = aggregation_hash == row["submitted_model_sha256"]

    def observe_global_model(self, parameters: list[Any]) -> None:
        """Record the model installed from real protocol aggregation."""
        self.rounds[self.current_round]["global_model_after_aggregation_sha256"] = canonical_parameter_hash(parameters)

    def evidence_for_round(self, round_id: Any) -> dict[str, Any]:
        """Return JSON-safe evidence without the private parameter snapshot."""
        row = dict(self.rounds[str(round_id)])
        row.pop("_pre")
        return row
