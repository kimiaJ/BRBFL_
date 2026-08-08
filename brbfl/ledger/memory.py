# ruff: noqa: D102, D107
"""Deterministic process-local implementation of the ledger semantic contract."""  # noqa: D102, D107

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from brbfl.canonical import canonical_hash
from brbfl.ledger.base import BlockchainLedger, EventType, LedgerEvent, LedgerReceipt
from brbfl.selection.roles import RoundRoleAssignment, validate_capabilities


@dataclass
class _Round:
    assignment: RoundRoleAssignment | None = None
    parent_model_hash: str | None = None
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    decisions: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    admission: dict[str, bool] | None = None
    aggregate: dict[str, Any] | None = None
    installations: dict[str, str] = field(default_factory=dict)
    finalized: bool = False


@dataclass
class _Experiment:
    required_installers: tuple[str, ...]
    participants: dict[str, frozenset[str]] = field(default_factory=dict)
    rounds: dict[int, _Round] = field(default_factory=dict)


class InMemoryLedger(BlockchainLedger):
    """Strict deterministic test/backend adapter; this is not a real blockchain."""

    def __init__(self, ledger_identifier: str = "memory") -> None:
        self.ledger_identifier = ledger_identifier
        self._experiments: dict[str, _Experiment] = {}
        self._events: list[LedgerEvent] = []
        self._closed = False

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    @property
    def final_event_chain_hash(self) -> str | None:
        return self._events[-1].event_hash if self._events else None

    def _append(
        self,
        event_type: EventType,
        experiment_id: str,
        payload: dict[str, Any],
        round_number: int | None = None,
        participant_id: str | None = None,
    ) -> LedgerReceipt:
        if self._closed:
            raise RuntimeError("ledger is closed")
        previous = self.final_event_chain_hash
        payload_hash = canonical_hash(f"LedgerPayload/{event_type.value}/v1", payload)
        sequence = len(self._events)
        envelope = {
            "experiment_id": experiment_id,
            "round_number": round_number,
            "participant_id": participant_id,
            "event_type": event_type.value,
            "payload_hash": payload_hash,
            "previous_event_hash": previous,
            "sequence_number": sequence,
        }
        event_hash = canonical_hash("LedgerEvent/v1", envelope)
        reference = f"{self.ledger_identifier}:{sequence}:{event_hash}"
        self._events.append(
            LedgerEvent(
                experiment_id,
                round_number,
                participant_id,
                event_type,
                deepcopy(payload),
                payload_hash,
                previous,
                sequence,
                reference,
                event_hash,
            )
        )
        return LedgerReceipt(self.ledger_identifier, sequence, event_hash)

    def _experiment(self, experiment_id: str) -> _Experiment:
        try:
            return self._experiments[experiment_id]
        except KeyError as exc:
            raise RuntimeError(f"experiment does not exist: {experiment_id}") from exc

    def _round(self, experiment_id: str, round_number: int, *, mutable: bool = True) -> _Round:
        experiment = self._experiment(experiment_id)
        try:
            record = experiment.rounds[round_number]
        except KeyError as exc:
            raise RuntimeError(f"round roles are not committed: experiment={experiment_id}, round={round_number}") from exc
        if mutable and record.finalized:
            raise RuntimeError(f"finalized round cannot be mutated: round={round_number}")
        return record

    def start_experiment(self, experiment_id: str, required_installers: tuple[str, ...]) -> LedgerReceipt:
        payload = {"required_installers": sorted(set(required_installers))}
        if experiment_id in self._experiments:
            return self._idempotent(EventType.EXPERIMENT_STARTED, experiment_id, payload)
        self._experiments[experiment_id] = _Experiment(tuple(payload["required_installers"]))
        return self._append(EventType.EXPERIMENT_STARTED, experiment_id, payload)

    def register_participant(self, experiment_id: str, participant_id: str, capabilities: frozenset[str]) -> LedgerReceipt:
        experiment = self._experiment(experiment_id)
        if not capabilities:
            raise ValueError("participant capabilities must not be empty")
        payload = {"participant_id": participant_id, "capabilities": sorted(capabilities)}
        existing = experiment.participants.get(participant_id)
        if existing is not None:
            if existing != capabilities:
                raise RuntimeError(f"conflicting participant registration: {participant_id}")
            return self._idempotent(EventType.PARTICIPANT_REGISTERED, experiment_id, payload, participant_id=participant_id)
        experiment.participants[participant_id] = capabilities
        return self._append(EventType.PARTICIPANT_REGISTERED, experiment_id, payload, participant_id=participant_id)

    def commit_round_roles(self, assignment: RoundRoleAssignment) -> LedgerReceipt:
        experiment = self._experiment(assignment.experiment_id)
        if set(assignment.network_participants) != set(experiment.participants):
            raise ValueError("round network participants must exactly match registered participants")
        validate_capabilities(assignment, experiment.participants)
        payload = {**assignment.canonical_payload(), "assignment_hash": assignment.assignment_hash}
        existing = experiment.rounds.get(assignment.round_number)
        if existing is not None:
            if existing.assignment != assignment:
                raise RuntimeError(f"round-role commitment is immutable: round={assignment.round_number}")
            return self._idempotent(EventType.ROUND_ROLES_COMMITTED, assignment.experiment_id, payload, assignment.round_number)
        if assignment.round_number > 0 and not experiment.rounds.get(assignment.round_number - 1, _Round()).finalized:
            raise RuntimeError("previous round must be finalized before committing next-round roles")
        experiment.rounds[assignment.round_number] = _Round(assignment=assignment)
        return self._append(EventType.ROUND_ROLES_COMMITTED, assignment.experiment_id, payload, assignment.round_number)

    def open_round(self, experiment_id: str, round_number: int, parent_model_hash: str) -> LedgerReceipt:
        record = self._round(experiment_id, round_number)
        payload = {"parent_model_hash": parent_model_hash, "assignment_hash": record.assignment.assignment_hash}
        if record.parent_model_hash is not None:
            if record.parent_model_hash != parent_model_hash:
                raise RuntimeError("conflicting round parent model hash")
            return self._idempotent(EventType.ROUND_OPENED, experiment_id, payload, round_number)
        record.parent_model_hash = parent_model_hash
        return self._append(EventType.ROUND_OPENED, experiment_id, payload, round_number)

    def commit_candidate(
        self, experiment_id: str, round_number: int, contributor_id: str, parent_model_hash: str, candidate_hash: str
    ) -> LedgerReceipt:
        record = self._round(experiment_id, round_number)
        if record.parent_model_hash is None:
            raise RuntimeError("round is not open")
        if contributor_id not in record.assignment.selected_contributors:
            raise RuntimeError(f"candidate submitter is not a selected contributor: {contributor_id}")
        if parent_model_hash != record.parent_model_hash:
            raise RuntimeError("candidate references stale or wrong-round parent model")
        payload = {"contributor_id": contributor_id, "parent_model_hash": parent_model_hash, "candidate_hash": candidate_hash}
        existing = record.candidates.get(contributor_id)
        if existing is not None:
            if existing != payload:
                raise RuntimeError(f"conflicting candidate commitment: {contributor_id}")
            return self._idempotent(EventType.CANDIDATE_COMMITTED, experiment_id, payload, round_number, contributor_id)
        record.candidates[contributor_id] = payload
        return self._append(EventType.CANDIDATE_COMMITTED, experiment_id, payload, round_number, contributor_id)

    def record_validator_decision(
        self, experiment_id: str, round_number: int, validator_id: str, contributor_id: str, candidate_hash: str, admitted: bool
    ) -> LedgerReceipt:
        record = self._round(experiment_id, round_number)
        if validator_id not in record.assignment.selected_validators:
            raise RuntimeError(f"decision submitter is not a selected validator: {validator_id}")
        candidate = record.candidates.get(contributor_id)
        if candidate is None or candidate["candidate_hash"] != candidate_hash:
            raise RuntimeError("validator decision does not reference the exact committed candidate hash")
        payload = {"validator_id": validator_id, "contributor_id": contributor_id, "candidate_hash": candidate_hash, "admitted": admitted}
        key = (validator_id, contributor_id)
        existing = record.decisions.get(key)
        if existing is not None:
            if existing != payload:
                raise RuntimeError(f"conflicting validator decision: {key}")
            return self._idempotent(EventType.VALIDATOR_DECISION_COMMITTED, experiment_id, payload, round_number, validator_id)
        record.decisions[key] = payload
        return self._append(EventType.VALIDATOR_DECISION_COMMITTED, experiment_id, payload, round_number, validator_id)

    def finalize_admission(self, experiment_id: str, round_number: int, decisions: dict[str, bool]) -> LedgerReceipt:
        record = self._round(experiment_id, round_number)
        expected = set(record.candidates)
        if set(decisions) != expected:
            raise RuntimeError("admission must decide every and only committed candidate")
        missing = [
            (validator, candidate)
            for validator in record.assignment.selected_validators
            for candidate in expected
            if (validator, candidate) not in record.decisions
        ]
        if missing:
            raise RuntimeError(f"admission cannot finalize before required validator decisions: {missing}")
        payload = {"decisions": {key: decisions[key] for key in sorted(decisions)}}
        if record.admission is not None:
            if record.admission != decisions:
                raise RuntimeError("finalized admission is immutable")
            return self._idempotent(EventType.ADMISSION_FINALIZED, experiment_id, payload, round_number)
        record.admission = dict(decisions)
        return self._append(EventType.ADMISSION_FINALIZED, experiment_id, payload, round_number)

    def commit_aggregate(
        self, experiment_id: str, round_number: int, contributor_hashes: dict[str, str], aggregate_hash: str
    ) -> LedgerReceipt:
        record = self._round(experiment_id, round_number)
        if record.admission is None:
            raise RuntimeError("admission is not finalized")
        expected = {node: record.candidates[node]["candidate_hash"] for node, admitted in record.admission.items() if admitted}
        if contributor_hashes != expected:
            raise RuntimeError("aggregate inputs must exactly match finalized admission and candidate hashes")
        payload = {
            "contributor_hashes": {key: contributor_hashes[key] for key in sorted(contributor_hashes)},
            "aggregate_hash": aggregate_hash,
        }
        if record.aggregate is not None:
            if record.aggregate != payload:
                raise RuntimeError("aggregate commitment is immutable")
            return self._idempotent(EventType.AGGREGATE_COMMITTED, experiment_id, payload, round_number)
        record.aggregate = payload
        return self._append(EventType.AGGREGATE_COMMITTED, experiment_id, payload, round_number)

    def confirm_model_installation(self, experiment_id: str, round_number: int, participant_id: str, aggregate_hash: str) -> LedgerReceipt:
        record = self._round(experiment_id, round_number)
        if participant_id not in self._experiment(experiment_id).participants:
            raise RuntimeError(f"installation confirmer is not registered: {participant_id}")
        if record.aggregate is None or record.aggregate["aggregate_hash"] != aggregate_hash:
            raise RuntimeError("installation does not reference the canonical aggregate")
        payload = {"participant_id": participant_id, "aggregate_hash": aggregate_hash}
        existing = record.installations.get(participant_id)
        if existing is not None:
            if existing != aggregate_hash:
                raise RuntimeError(f"conflicting installation confirmation: {participant_id}")
            return self._idempotent(EventType.MODEL_INSTALLATION_CONFIRMED, experiment_id, payload, round_number, participant_id)
        record.installations[participant_id] = aggregate_hash
        return self._append(EventType.MODEL_INSTALLATION_CONFIRMED, experiment_id, payload, round_number, participant_id)

    def finalize_round(self, experiment_id: str, round_number: int) -> LedgerReceipt:
        record = self._round(experiment_id, round_number, mutable=False)
        if record.finalized:
            payload = {
                "aggregate_hash": record.aggregate["aggregate_hash"],
                "confirmed_installers": sorted(record.installations),
            }
            return self._idempotent(EventType.ROUND_FINALIZED, experiment_id, payload, round_number)
        if record.aggregate is None:
            raise RuntimeError("aggregate is not committed")
        required = set(self._experiment(experiment_id).required_installers)
        missing = required - set(record.installations)
        if missing:
            raise RuntimeError(f"required installation confirmations are missing: {sorted(missing)}")
        payload = {"aggregate_hash": record.aggregate["aggregate_hash"], "confirmed_installers": sorted(record.installations)}
        record.finalized = True
        return self._append(EventType.ROUND_FINALIZED, experiment_id, payload, round_number)

    def _idempotent(
        self,
        event_type: EventType,
        experiment_id: str,
        payload: dict[str, Any],
        round_number: int | None = None,
        participant_id: str | None = None,
    ) -> LedgerReceipt:
        payload_hash = canonical_hash(f"LedgerPayload/{event_type.value}/v1", payload)
        for event in self._events:
            if (event.event_type, event.experiment_id, event.round_number, event.participant_id, event.payload_hash) == (
                event_type,
                experiment_id,
                round_number,
                participant_id,
                payload_hash,
            ):
                return LedgerReceipt(self.ledger_identifier, event.sequence_number, event.event_hash)
        raise RuntimeError(f"ledger state has no matching event for idempotent {event_type.value}")

    def get_round_record(self, experiment_id: str, round_number: int) -> dict[str, Any]:
        record = self._round(experiment_id, round_number, mutable=False)
        return deepcopy(
            {
                "assignment": record.assignment.canonical_payload(),
                "assignment_hash": record.assignment.assignment_hash,
                "parent_model_hash": record.parent_model_hash,
                "candidates": record.candidates,
                "decisions": record.decisions,
                "admission": record.admission,
                "aggregate": record.aggregate,
                "installations": record.installations,
                "finalized": record.finalized,
            }
        )

    def verify_round(self, experiment_id: str, round_number: int) -> bool:
        self._round(experiment_id, round_number, mutable=False)
        previous = None
        for sequence, event in enumerate(self._events):
            if event.sequence_number != sequence or event.previous_event_hash != previous:
                raise RuntimeError(f"broken previous-event linkage at sequence {sequence}")
            payload_hash = canonical_hash(f"LedgerPayload/{event.event_type.value}/v1", event.payload)
            if payload_hash != event.payload_hash:
                raise RuntimeError(f"payload tampering detected at sequence {sequence}")
            envelope = {
                "experiment_id": event.experiment_id,
                "round_number": event.round_number,
                "participant_id": event.participant_id,
                "event_type": event.event_type.value,
                "payload_hash": event.payload_hash,
                "previous_event_hash": event.previous_event_hash,
                "sequence_number": event.sequence_number,
            }
            if canonical_hash("LedgerEvent/v1", envelope) != event.event_hash:
                raise RuntimeError(f"event tampering detected at sequence {sequence}")
            previous = event.event_hash
        return True

    def validation_artifact(self, experiment_id: str) -> dict[str, Any]:
        experiment = self._experiment(experiment_id)
        rounds = sorted(experiment.rounds)
        return {
            "enabled": True,
            "backend": "memory",
            "ledger_identifier": self.ledger_identifier,
            "experiment_id": experiment_id,
            "participant_registrations": {node: sorted(caps) for node, caps in sorted(experiment.participants.items())},
            "per_round_role_assignment": {str(number): experiment.rounds[number].assignment.canonical_payload() for number in rounds},
            "per_round_role_assignment_hash": {str(number): experiment.rounds[number].assignment.assignment_hash for number in rounds},
            "per_round_events": {
                str(number): [event.event_type.value for event in self._events if event.round_number == number] for number in rounds
            },
            "event_receipts": [event.backend_reference for event in self._events],
            "final_event_chain_hash": self.final_event_chain_hash,
            "round_verification": {str(number): self.verify_round(experiment_id, number) for number in rounds},
            "ledger_round_consensus": all(experiment.rounds[number].finalized for number in rounds),
        }

    def close(self) -> None:
        self._closed = True
