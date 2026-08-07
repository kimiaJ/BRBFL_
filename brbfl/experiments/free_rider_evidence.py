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
        "parameters_equal_to_pre_training": exact,
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
        self.node_id = "unknown"
        self.rounds: dict[int, dict[str, Any]] = {}
        self.current_round: int | None = None

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
        key = int(round_id)
        if key in self.rounds:
            raise RuntimeError(f"audit record already initialized: node={self.node_id}, round={key}")
        snapshot = [np.asarray(value).copy() for value in parameters]
        self.current_round = key
        self.rounds[key] = {
            "record_initialized": True,
            "record_finalized": False,
            "local_training_started": True,
            "local_training_finished": False,
            "submission_produced": False,
            "submission_reached_aggregation": False,
            "local_training_invocation_count": 1,
            "optimizer_step_count": 0,
            "configured_local_epochs": self.configured_epochs,
            "configured_batch_count": self.configured_batch_count,
            "local_epochs_actually_executed": 0,
            "effective_local_epochs": 0,
            "free_rider_attack_application_count": 0,
            "attack_application_count": 0,
            "training_skipped": None,
            "pre_training_model_sha256": canonical_parameter_hash(snapshot),
            "_pre": snapshot,
        }

    def should_skip_local_training(self) -> bool:
        """Apply and count the explicit training-control decision once."""
        hook = getattr(self.attack, "should_skip_local_training", None)
        skip = bool(hook and hook())
        row = self.rounds[self.current_round]
        if row["training_skipped"] is not None:
            raise RuntimeError(f"training branch already selected: node={self.node_id}, round={self.current_round}")
        row["training_skipped"] = skip
        if skip:
            row["free_rider_attack_application_count"] += 1
            row["attack_application_count"] += 1
        return skip

    def record_optimizer_step(self) -> None:
        """Count a training batch that produces one optimizer step in this learner."""
        if self.current_round is not None:
            self.rounds[self.current_round]["optimizer_step_count"] += 1

    def record_optimizer_steps(self, count: int) -> None:
        """Record the observed count returned with a locally or remotely fitted model."""
        if count < 0:
            raise ValueError("optimizer step count cannot be negative")
        self.rounds[self.current_round]["optimizer_step_count"] = count
        self.rounds[self.current_round]["observed_batch_count"] = count

    def complete_local_training(self, parameters: list[Any], skipped: bool) -> None:
        """Record the post-fit or deliberately skipped model."""
        row = self.rounds[self.current_round]
        if row["training_skipped"] is not skipped:
            raise RuntimeError(f"training branches disagree: node={self.node_id}, round={self.current_round}")
        row["local_training_finished"] = True
        row["local_epochs_actually_executed"] = 0 if skipped else self.configured_epochs
        row["effective_local_epochs"] = row["local_epochs_actually_executed"]
        row["post_training_pre_submission_model_sha256"] = canonical_parameter_hash(parameters)
        row["post_training_model_sha256"] = row["post_training_pre_submission_model_sha256"]

    def publish_update(self, parameters: list[Any]) -> list[Any]:
        """Select the parameters which the model publication path will install."""
        hook = getattr(self.attack, "publish_update", None)
        submitted = hook(parameters) if hook else parameters
        return [np.asarray(value).copy() for value in submitted]

    def record_submission(self, parameters: list[Any], round_id: Any) -> None:
        """Record the immutable snapshot actually installed for publication."""
        key = int(round_id)
        row = self._row(key, "record submission")
        if not row["local_training_finished"]:
            raise self._lifecycle_error(key, "submission before training completion")
        if row["submission_produced"]:
            raise self._lifecycle_error(key, "submission already recorded")
        detached = [np.asarray(value).copy() for value in parameters]
        row["submission_produced"] = True
        row["submitted_model_sha256"] = canonical_parameter_hash(detached)
        row["_submitted"] = detached
        row.update(parameter_delta(row["_pre"], detached))

    def observe_aggregation(self, parameters: list[Any], round_id: Any | None = None) -> None:
        """Observe the exact local snapshot immediately passed to add_model()."""
        key = self._canonical_round(round_id)
        row = self._row(key, "observe aggregation")
        if not row["submission_produced"]:
            raise self._lifecycle_error(key, "aggregation observed before submission")
        if row["submission_reached_aggregation"]:
            raise self._lifecycle_error(key, "aggregation already observed")
        aggregation_snapshot = [np.asarray(value).copy() for value in parameters]
        aggregation_hash = canonical_parameter_hash(aggregation_snapshot)
        row["aggregation_input_sha256"] = aggregation_hash
        row["submission_reached_aggregation"] = True
        row["aggregation_matches_submitted_snapshot"] = aggregation_hash == row["submitted_model_sha256"]
        row["aggregation_input_numerically_equals_submission"] = all(
            np.array_equal(left, right) for left, right in zip(row["_submitted"], aggregation_snapshot, strict=True)
        ) and len(row["_submitted"]) == len(aggregation_snapshot)

    def observe_global_model(self, parameters: list[Any]) -> None:
        """Record the model installed from real protocol aggregation."""
        row = self.rounds[self.current_round]
        if row["record_finalized"]:
            raise RuntimeError(f"audit record already finalized: node={self.node_id}, round={self.current_round}")
        if not row["submission_reached_aggregation"]:
            raise self._lifecycle_error(self.current_round, "finalization before aggregation observation")
        if not row["aggregation_matches_submitted_snapshot"] or not row["aggregation_input_numerically_equals_submission"]:
            raise RuntimeError(f"submission was not observed at aggregation: node={self.node_id}, round={self.current_round}")
        if row["training_skipped"]:
            assert row["optimizer_step_count"] == row["effective_local_epochs"] == 0
        else:
            assert row["attack_application_count"] == 0
            if self.configured_batch_count:
                assert row["optimizer_step_count"] > 0
        row["global_model_after_aggregation_sha256"] = canonical_parameter_hash(parameters)
        row["record_finalized"] = True

    def _canonical_round(self, round_id: Any | None) -> int:
        """Return one integer round identity, rejecting a mismatched callback."""
        key = int(self.current_round if round_id is None else round_id)
        if self.current_round != key:
            raise RuntimeError(f"audit round mismatch: node={self.node_id}, current_round={self.current_round}, requested_round={key}")
        return key

    def _row(self, key: int, operation: str) -> dict[str, Any]:
        """Require initialization before any lifecycle observation."""
        row = self.rounds.get(key)
        if row is None:
            raise RuntimeError(
                f"cannot {operation} for uninitialized audit: node={self.node_id}, round={key}, "
                f"current_state=uninitialized, available_evidence_keys=[]"
            )
        return row

    def _lifecycle_error(self, key: int, reason: str) -> RuntimeError:
        """Build a diagnostic validation error without leaking incidental KeyErrors."""
        row = self.rounds[key]
        state = (
            "finalized"
            if row["record_finalized"]
            else "aggregation_observed"
            if row["submission_reached_aggregation"]
            else (
                "submission_recorded"
                if row["submission_produced"]
                else "training_completed"
                if row["local_training_finished"]
                else "training"
            )
        )
        return RuntimeError(
            f"invalid training lifecycle ({reason}): node={self.node_id}, round={key}, current_state={state}, "
            f"available_evidence_keys={sorted(k for k in row if not k.startswith('_'))}, "
            f"training_completed={row['local_training_finished']}, submission_recorded={row['submission_produced']}, "
            f"aggregation_previously_observed={row['submission_reached_aggregation']}"
        )

    def evidence_for_round(self, round_id: Any) -> dict[str, Any]:
        """Return JSON-safe evidence without the private parameter snapshot."""
        key = int(round_id)
        source = self.rounds.get(key)
        initialized = source is not None
        finalized = bool(source and source.get("record_finalized"))
        if not initialized or not finalized:
            raise RuntimeError(
                "training audit evidence unavailable: "
                f"node={self.node_id}, requested_round={key}, available_round_keys={list(self.rounds)}, "
                f"initialized={initialized}, finalized={finalized}"
            )
        row = dict(source)
        row.pop("_pre")
        row.pop("_submitted")
        return row
