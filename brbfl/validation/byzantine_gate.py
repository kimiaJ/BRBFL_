"""
Deterministic validator-subgroup gate at the real aggregation boundary.

The gate deliberately has no dependency on MNIST or a particular detector.  A
candidate is snapshotted before votes are calculated and only an admitted,
byte-identical model may subsequently be passed to ``Aggregator.add_model``.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_numpy(parameter: Any) -> np.ndarray:
    """Detach a framework value and return independent, stable CPU storage."""
    value = parameter
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.ascontiguousarray(np.asarray(value)).copy()
    result.flags.writeable = False
    return result


def canonical_parameters(parameters: list[Any]) -> tuple[np.ndarray, ...]:
    """Canonical immutable, value-based representation of ordered parameters."""
    return tuple(_as_numpy(parameter) for parameter in parameters)


def parameter_hash(parameters: list[Any] | tuple[np.ndarray, ...]) -> str:
    """Hash parameter values with unambiguous order, dtype, and shape framing."""
    digest = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        value = _as_numpy(parameter)
        metadata = json.dumps(
            {"index": index, "dtype": value.dtype.str, "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class AdmissionPolicy:
    """Validator membership and fail-closed voting semantics."""

    contributors: tuple[str, ...]
    validators: tuple[str, ...]
    byzantine_validators: tuple[str, ...] = ()
    quorum: int = 3
    acceptance_threshold: int = 2
    strategy: str = "invert_reference_vote"
    group_id: str | None = None
    max_l2_norm: float = math.inf
    reference_reject_candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate fail-closed policy constraints."""
        if not self.validators or len(set(self.validators)) != len(self.validators):
            raise ValueError("eligible validators must be non-empty and unique")
        if not set(self.byzantine_validators) <= set(self.validators):
            raise ValueError("Byzantine validators must belong to the eligible validator subgroup")
        if not 1 <= self.quorum <= len(self.validators):
            raise ValueError("quorum must be between one and the validator count")
        if not 1 <= self.acceptance_threshold <= self.quorum:
            raise ValueError("acceptance threshold must be between one and quorum")
        if self.strategy != "invert_reference_vote":
            raise ValueError(f"unsupported Byzantine validator strategy: {self.strategy}")


class ValidatorSubgroupGate:
    """Calculate honest references, publish votes, and gate aggregation input."""

    def __init__(self, policy: AdmissionPolicy) -> None:
        """Initialize an empty, thread-safe lifecycle ledger."""
        self.policy = policy
        self._candidates: dict[tuple[int, str], dict[str, Any]] = {}
        self._order = 0
        self._lock = threading.RLock()

    def submit_and_decide(
        self,
        round_id: Any,
        candidate: str,
        parameters: list[Any],
        *,
        current_node: str = "unknown",
        lifecycle_path: str = "unknown",
        expected_hash: str | None = None,
        transport_occurred: bool = False,
        parent_global_model_sha256: str | None = None,
    ) -> bool:
        """Record an immutable submission and perform validation exactly once."""
        round_number = self._round(round_id)
        if candidate not in self.policy.contributors:
            raise RuntimeError(f"candidate is not an eligible contributor: round={round_number}, candidate={candidate}")
        key = (round_number, candidate)
        snapshot = canonical_parameters(parameters)
        submitted_hash = parameter_hash(snapshot)
        with self._lock:
            existing = self._candidates.get(key)
            if existing is not None:
                if existing["submitted_model_sha256"] != submitted_hash:
                    raise self._integrity_error(
                        round_number,
                        candidate,
                        current_node,
                        lifecycle_path,
                        existing["submitted_model_sha256"],
                        submitted_hash,
                        existing["_snapshot"],
                        snapshot,
                        transport_occurred,
                    )
                return bool(existing["admitted"])
            if expected_hash is not None and expected_hash != submitted_hash:
                raise self._integrity_error(
                    round_number,
                    candidate,
                    current_node,
                    lifecycle_path,
                    expected_hash,
                    submitted_hash,
                    None,
                    snapshot,
                    transport_occurred,
                )
            self._order += 1
            finite = all(bool(np.isfinite(item).all()) for item in snapshot)
            norm = math.sqrt(sum(float(np.sum(np.asarray(item, dtype=np.float64) ** 2)) for item in snapshot))
            reference = finite and norm <= self.policy.max_l2_norm and candidate not in self.policy.reference_reject_candidates
            votes = []
            for validator in self.policy.validators:
                byzantine = validator in self.policy.byzantine_validators
                reported = not reference if byzantine else reference
                vote = {
                    "round": round_number,
                    "candidate_node_id": candidate,
                    "validator_node_id": validator,
                    "roles": {
                        "contributor": validator in self.policy.contributors,
                        "validator": True,
                    },
                    "validator_eligible": True,
                    "byzantine": byzantine,
                    "strategy": self.policy.strategy if byzantine else "honest_reference",
                    "attack_group_id": self.policy.group_id if byzantine else None,
                    "candidate_submitted_model_sha256": submitted_hash,
                    "candidate_update_sha256": submitted_hash,
                    "reference_decision": reference,
                    "reported_decision": reported,
                    "reported_equals_reference": reported == reference,
                    "reference_rule": {
                        "name": "finite_l2_and_candidate_fixture",
                        "all_parameters_finite": finite,
                        "model_l2_norm": norm,
                        "maximum_l2_norm": self.policy.max_l2_norm,
                        "candidate_explicitly_rejected": candidate in self.policy.reference_reject_candidates,
                    },
                    "attack_application_count": int(byzantine),
                    "vote_publication_count": 1,
                    "lifecycle_state": "vote_published",
                    "order_index": self._order + len(votes) + 1,
                }
                vote["vote_sha256"] = _canonical_hash(vote)
                votes.append(vote)
            accepts = sum(vote["reported_decision"] for vote in votes)
            rejects = len(votes) - accepts
            quorum_reached = len(votes) >= self.policy.quorum
            threshold_reached = accepts >= self.policy.acceptance_threshold
            admitted = quorum_reached and threshold_reached
            self._order += len(votes) + 1
            self._candidates[key] = {
                "round": round_number,
                "candidate_node_id": candidate,
                "submitted_model_sha256": submitted_hash,
                "parent_global_model_sha256": parent_global_model_sha256,
                "eligible_validators": list(self.policy.validators),
                "received_validators": list(self.policy.validators),
                "missing_validators": [],
                "duplicate_votes": [],
                "invalid_votes": [],
                "votes": votes,
                "honest_votes": [v for v in votes if not v["byzantine"]],
                "byzantine_votes": [v for v in votes if v["byzantine"]],
                "accept_count": accepts,
                "reject_count": rejects,
                "quorum_reached": quorum_reached,
                "threshold_reached": threshold_reached,
                "admitted": admitted,
                "admission_reason": "accepted" if admitted else "acceptance_threshold_not_reached",
                "reached_aggregator_add_model": False,
                "aggregation_input_sha256": None,
                "aggregation_matches_submitted_snapshot": None,
                "rejection_receipt": None if admitted else {"blocked_before_aggregator_add_model": True},
                "lifecycle_state": "admission_calculated",
                "_snapshot": snapshot,
            }
            return admitted

    def observe_round_result(self, round_id: Any, parameters: list[Any], contributors: list[str], *, canonical_hash_source: str) -> None:
        """Attach installed-model and exact aggregation lineage to every row in a round."""
        round_number = self._round(round_id)
        installed_hash = parameter_hash(parameters)
        contributor_set = sorted(contributors)
        inputs = {
            candidate: row["aggregation_input_sha256"]
            for (candidate_round, candidate), row in self._candidates.items()
            if candidate_round == round_number and row["reached_aggregator_add_model"]
        }
        for (candidate_round, _candidate), row in self._candidates.items():
            if candidate_round == round_number:
                row["round_aggregation_contributors"] = contributor_set
                row["round_aggregation_input_hashes"] = inputs
                row["installed_global_model_sha256"] = installed_hash
                row["canonical_hash_source"] = canonical_hash_source

    def observe_aggregation_input(
        self,
        round_id: Any,
        candidate: str,
        parameters: list[Any],
        *,
        current_node: str = "unknown",
        lifecycle_path: str = "unknown",
        transport_occurred: bool = False,
    ) -> None:
        """Prove an admitted immutable snapshot reached ``add_model``."""
        row = self._candidate(round_id, candidate, "observe aggregation input")
        if not row["admitted"]:
            raise RuntimeError(f"rejected candidate cannot reach aggregation: round={int(round_id)}, candidate={candidate}")
        observed = canonical_parameters(parameters)
        digest = parameter_hash(observed)
        if digest != row["submitted_model_sha256"]:
            raise self._integrity_error(
                self._round(round_id),
                candidate,
                current_node,
                lifecycle_path,
                row["submitted_model_sha256"],
                digest,
                row["_snapshot"],
                observed,
                transport_occurred,
            )
        row["reached_aggregator_add_model"] = True
        row["aggregation_input_sha256"] = digest
        row["aggregation_matches_submitted_snapshot"] = True
        row["lifecycle_state"] = "aggregation_input_observed"

    def submitted_hash(self, round_id: Any, candidate: str) -> str:
        """Return authoritative canonical submission evidence for transport."""
        return str(self._candidate(round_id, candidate, "read submitted hash")["submitted_model_sha256"])

    @staticmethod
    def _integrity_error(
        round_number, candidate, current_node, path, expected_hash, observed_hash, expected, observed, transport_occurred
    ) -> RuntimeError:
        first = None
        maximum = 0.0
        expected_meta = observed_meta = None
        if expected is not None:
            for index, (left, right) in enumerate(zip(expected, observed, strict=False)):
                expected_meta = f"shape={left.shape},dtype={left.dtype}"
                observed_meta = f"shape={right.shape},dtype={right.dtype}"
                if left.shape != right.shape or left.dtype != right.dtype:
                    first = f"parameter[{index}]"
                    break
                unequal = np.argwhere(left != right)
                if unequal.size:
                    position = tuple(int(item) for item in unequal[0])
                    first = f"parameter[{index}]{position}"
                    maximum = max(maximum, float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))))
                    break
        return RuntimeError(
            "candidate snapshot changed after submission (integrity failure): "
            f"current_node={current_node}, candidate={candidate}, round={round_number}, path={path}, "
            f"expected_hash={expected_hash}, observed_hash={observed_hash}, first_difference={first}, "
            f"maximum_absolute_difference={maximum}, expected={expected_meta}, observed={observed_meta}, "
            f"transport_occurred={transport_occurred}, lifecycle_state=submission_or_pre_aggregation"
        )

    def publish_vote(self, round_id: Any, candidate: str, validator: str, decision: bool) -> None:
        """
        Reject external invalid or duplicate publications descriptively.

        Normal votes are published atomically by :meth:`submit_and_decide`; this
        method is the strict extension point for future remote validator actors.
        """
        row = self._candidate(round_id, candidate, "publish vote")
        if validator not in self.policy.validators:
            row["invalid_votes"].append(validator)
            raise RuntimeError(f"vote from ineligible validator: {validator}")
        if validator in row["received_validators"]:
            row["duplicate_votes"].append(validator)
            raise RuntimeError(f"duplicate validator vote: {validator}")
        raise RuntimeError("late vote publication is not allowed after admission calculation")

    def evidence(self) -> list[dict[str, Any]]:
        """Return detached, JSON-safe evidence in round/candidate order."""
        result = []
        for key in sorted(self._candidates):
            row = {name: value for name, value in self._candidates[key].items() if not name.startswith("_")}
            result.append(json.loads(json.dumps(row, sort_keys=True)))
        return result

    def _candidate(self, round_id: Any, candidate: str, operation: str) -> dict[str, Any]:
        key = (self._round(round_id), candidate)
        row = self._candidates.get(key)
        if row is None:
            raise RuntimeError(f"cannot {operation} before candidate submission: round={key[0]}, candidate={candidate}")
        return row

    @staticmethod
    def _round(round_id: Any) -> int:
        try:
            value = int(round_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"round ID must be a canonical integer: {round_id!r}") from exc
        if isinstance(round_id, float) and round_id != value:
            raise ValueError(f"round ID must be a canonical integer: {round_id!r}")
        return value


_gate_policy: AdmissionPolicy | None = None
_gates: dict[str, ValidatorSubgroupGate] = {}
_registry_lock = threading.Lock()


def install_validator_gate(gate: ValidatorSubgroupGate) -> None:
    """Install a policy whose ledgers are isolated for every node."""
    global _gate_policy
    with _registry_lock:
        _gate_policy = gate.policy
        _gates.clear()


def get_validator_gate(node_id: str | None = None) -> ValidatorSubgroupGate | None:
    """Return the configured gate, if validation is enabled."""
    with _registry_lock:
        if _gate_policy is None:
            return None
        key = node_id or "legacy"
        if key not in _gates:
            _gates[key] = ValidatorSubgroupGate(_gate_policy)
        return _gates[key]


def validator_evidence() -> list[dict[str, Any]]:
    """Return detached evidence from all node-local validation ledgers."""
    with _registry_lock:
        return [dict(row, current_node=node) for node, gate in sorted(_gates.items()) for row in gate.evidence()]


def clear_validator_gate() -> None:
    """Remove experiment state during deterministic shutdown."""
    global _gate_policy
    with _registry_lock:
        _gate_policy = None
        _gates.clear()
