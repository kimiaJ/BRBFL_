# ruff: noqa: D101, D102
"""Backend-independent lifecycle ledger contract and stable records."""  # noqa: D101, D102

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brbfl.selection.roles import RoundRoleAssignment


class EventType(str, Enum):
    EXPERIMENT_STARTED = "ExperimentStarted"
    PARTICIPANT_REGISTERED = "ParticipantRegistered"
    ROUND_ROLES_COMMITTED = "RoundRolesCommitted"
    ROUND_OPENED = "RoundOpened"
    CANDIDATE_COMMITTED = "CandidateCommitted"
    VALIDATOR_DECISION_COMMITTED = "ValidatorDecisionCommitted"
    ADMISSION_FINALIZED = "AdmissionFinalized"
    AGGREGATE_COMMITTED = "AggregateCommitted"
    MODEL_INSTALLATION_CONFIRMED = "ModelInstallationConfirmed"
    ROUND_FINALIZED = "RoundFinalized"


@dataclass(frozen=True)
class LedgerReceipt:
    """Deterministic backend reference returned for a committed event."""

    ledger_identifier: str
    sequence_number: int
    event_hash: str


@dataclass(frozen=True)
class LedgerEvent:
    """One immutable hash-linked lifecycle event."""

    experiment_id: str
    round_number: int | None
    participant_id: str | None
    event_type: EventType
    payload: dict[str, Any]
    payload_hash: str
    previous_event_hash: str | None
    sequence_number: int
    backend_reference: str
    event_hash: str


class BlockchainLedger(ABC):
    """Semantic contract for memory and future permissioned-chain backends."""

    @abstractmethod
    def start_experiment(self, experiment_id: str, required_installers: tuple[str, ...]) -> LedgerReceipt: ...

    @abstractmethod
    def register_participant(self, experiment_id: str, participant_id: str, capabilities: frozenset[str]) -> LedgerReceipt: ...

    @abstractmethod
    def commit_round_roles(self, assignment: RoundRoleAssignment) -> LedgerReceipt: ...

    @abstractmethod
    def open_round(self, experiment_id: str, round_number: int, parent_model_hash: str) -> LedgerReceipt: ...

    @abstractmethod
    def commit_candidate(
        self, experiment_id: str, round_number: int, contributor_id: str, parent_model_hash: str, candidate_hash: str
    ) -> LedgerReceipt: ...

    @abstractmethod
    def record_validator_decision(
        self,
        experiment_id: str,
        round_number: int,
        validator_id: str,
        contributor_id: str,
        candidate_hash: str,
        admitted: bool,
        evidence: dict[str, Any] | None = None,
    ) -> LedgerReceipt: ...

    @abstractmethod
    def finalize_admission(self, experiment_id: str, round_number: int, decisions: dict[str, bool]) -> LedgerReceipt: ...

    @abstractmethod
    def commit_aggregate(
        self, experiment_id: str, round_number: int, contributor_hashes: dict[str, str], aggregate_hash: str
    ) -> LedgerReceipt: ...

    @abstractmethod
    def confirm_model_installation(
        self, experiment_id: str, round_number: int, participant_id: str, aggregate_hash: str
    ) -> LedgerReceipt: ...

    @abstractmethod
    def finalize_round(self, experiment_id: str, round_number: int) -> LedgerReceipt: ...

    @abstractmethod
    def get_round_record(self, experiment_id: str, round_number: int) -> dict[str, Any]: ...

    @abstractmethod
    def verify_round(self, experiment_id: str, round_number: int) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...
