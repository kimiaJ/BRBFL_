"""Immutable numerical and lifecycle evidence for coordinated collusion."""
# ruff: noqa: D102, D103, D105, D107

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from brbfl.attacks.collusion import CollusionAttack
from brbfl.experiments.free_rider_evidence import canonical_parameter_hash

NUMERICAL_TOLERANCE = 1e-6


def snapshot(values: Sequence[Any]) -> list[np.ndarray]:
    return [np.asarray(value).copy() for value in values]


def delta(left: Sequence[Any], right: Sequence[Any]) -> list[np.ndarray]:
    return [np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64) for a, b in zip(left, right, strict=True)]


def l2(values: Sequence[Any]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(value, dtype=np.float64) ** 2)) for value in values)))


def cosine(left: Sequence[Any], right: Sequence[Any]) -> float:
    a, b = l2(left), l2(right)
    if a == 0.0 or b == 0.0:
        return 0.0
    dot = sum(float(np.sum(np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64))) for x, y in zip(left, right, strict=True))
    return float(dot / (a * b))


class CollusionLifecycleAudit:
    """Narrow post-fit hook proving training precedes one coordinated transform."""

    def __init__(self, attack: CollusionAttack | None, node_id: str, configured_epochs: int, configured_batch_count: int):
        self.attack, self.node_id = attack, node_id
        self.configured_epochs, self.configured_batch_count = configured_epochs, configured_batch_count
        self.current_round: int | None = None
        self.rounds: dict[int, dict[str, Any]] = {}

    def __bool__(self):
        return self.attack is not None

    def on_attach(self, node: Any) -> None:
        pass

    def poison_data(self, dataset: Any) -> Any:
        return dataset

    def should_skip_local_training(self) -> bool:
        return False

    def begin_local_training(self, round_id: Any, parameters: Sequence[Any]) -> None:
        key = int(round_id)
        if key in self.rounds:
            raise RuntimeError(f"audit record already initialized: node={self.node_id}, round={key}")
        before = snapshot(parameters)
        self.current_round = key
        self.rounds[key] = {
            "node_id": self.node_id,
            "round_id": key,
            "participant": True,
            "malicious": bool(self.attack),
            "configured_local_epochs": self.configured_epochs,
            "effective_local_epochs": 0,
            "optimizer_step_count": 0,
            "attack_application_count": 0,
            "local_training_finished": False,
            "submission_produced": False,
            "submission_reached_aggregation": False,
            "record_finalized": False,
            "pre_training_model_sha256": canonical_parameter_hash(before),
            "_pre": before,
        }

    def record_optimizer_steps(self, count: int) -> None:
        if count < 0:
            raise ValueError("optimizer step count cannot be negative")
        self._row()["optimizer_step_count"] = int(count)

    def complete_local_training(self, parameters: Sequence[Any], skipped: bool) -> None:
        row = self._row()
        if skipped:
            raise RuntimeError("collusion must not skip genuine local training")
        trained = snapshot(parameters)
        update = delta(row["_pre"], trained)
        row.update(
            {
                "local_training_finished": True,
                "effective_local_epochs": self.configured_epochs,
                "genuine_post_training_model_sha256": canonical_parameter_hash(trained),
                "genuine_update_sha256": canonical_parameter_hash(update),
                "genuine_update_l2_norm": l2(update),
                "_trained": trained,
                "_genuine": update,
            }
        )

    def publish_update(self, parameters: Sequence[Any]) -> list[np.ndarray]:
        row = self._row()
        if not row["local_training_finished"]:
            raise RuntimeError("attack transformation before genuine training completion")
        if self.attack is None:
            return snapshot(parameters)
        if row["attack_application_count"]:
            raise RuntimeError("collusion attack already applied")
        submitted, direction = self.attack.transform(row["_pre"], row["_trained"], self.current_round)
        normalized = [value / l2(direction) for value in direction]
        submitted_update = delta(row["_pre"], submitted)
        row.update(
            {
                "attack_application_count": 1,
                "coordination_group_id": self.attack.group_id,
                "strategy_name": self.attack.strategy,
                "attack_seed": self.attack.seed,
                "configured_alpha": self.attack.alpha,
                "shared_direction_sha256": canonical_parameter_hash(direction),
                "shared_direction_l2_norm_before_normalization": l2(direction),
                "normalized_direction_norm": l2(normalized),
                "submitted_update_sha256": canonical_parameter_hash(submitted_update),
                "submitted_update_l2_norm": l2(submitted_update),
                "maximum_absolute_submitted_delta": max((float(np.max(np.abs(x), initial=0.0)) for x in submitted_update), default=0.0),
                "genuine_submitted_cosine_similarity": cosine(row["_genuine"], submitted_update),
                "submitted_shared_direction_cosine_similarity": cosine(submitted_update, direction),
                "_direction": direction,
            }
        )
        return snapshot(submitted)

    def record_submission(self, parameters: Sequence[Any], round_id: Any) -> None:
        row = self._row(int(round_id))
        if not row["local_training_finished"]:
            raise RuntimeError("submission before training completion")
        if row["submission_produced"]:
            raise RuntimeError("submission already recorded")
        submitted = snapshot(parameters)
        row.update({"submission_produced": True, "submitted_model_sha256": canonical_parameter_hash(submitted), "_submitted": submitted})
        if self.attack is None:
            update = delta(row["_pre"], submitted)
            row.update({"submitted_update_sha256": canonical_parameter_hash(update), "submitted_update_l2_norm": l2(update)})

    def observe_aggregation(self, parameters: Sequence[Any], round_id: Any | None = None) -> None:
        row = self._row(self.current_round if round_id is None else int(round_id))
        if not row["submission_produced"]:
            raise RuntimeError("aggregation observed before submission")
        if row["submission_reached_aggregation"]:
            raise RuntimeError("aggregation already observed")
        values = snapshot(parameters)
        digest = canonical_parameter_hash(values)
        row.update(
            {
                "submission_reached_aggregation": True,
                "aggregation_input_sha256": digest,
                "aggregation_receipt": digest == row["submitted_model_sha256"],
                "aggregation_matches_submitted_snapshot": digest == row["submitted_model_sha256"],
            }
        )

    def observe_global_model(self, parameters: Sequence[Any]) -> None:
        row = self._row()
        if not row["submission_reached_aggregation"]:
            raise RuntimeError("finalization before aggregation observation")
        if row["record_finalized"]:
            raise RuntimeError("audit record already finalized")
        row.update({"installed_global_model_sha256": canonical_parameter_hash(snapshot(parameters)), "record_finalized": True})

    def _row(self, key: int | None = None) -> dict[str, Any]:
        key = self.current_round if key is None else key
        row = self.rounds.get(key)
        if row is None:
            raise RuntimeError(f"collusion audit unavailable: node={self.node_id}, round={key}, available_round_keys={list(self.rounds)}")
        return row

    def evidence_for_round(self, round_id: Any) -> dict[str, Any]:
        row = self._row(int(round_id))
        if not row["record_finalized"]:
            raise RuntimeError(f"collusion audit not finalized: node={self.node_id}, round={int(round_id)}")
        return {key: value for key, value in row.items() if not key.startswith("_")}
