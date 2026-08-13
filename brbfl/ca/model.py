"""Deterministic, immutable participant cellular-automata transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType

from brbfl.canonical import canonical_hash


class ParticipantState(str, Enum):
    """A participant's CA state, ordered from trusted to excluded."""

    TRUSTED = "trusted"
    OBSERVATION = "observation"
    SUSPICIOUS = "suspicious"
    EXCLUDED = "excluded"


class EvidenceCategory(str, Enum):
    """Finalized local evidence for one round."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    SEVERE = "severe"


@dataclass(frozen=True, slots=True)
class ParticipantCAState:
    """Immutable history and state for one registered participant."""

    participant_id: str
    state: ParticipantState = ParticipantState.OBSERVATION
    consecutive_positive_rounds: int = 0
    consecutive_negative_rounds: int = 0
    rounds_in_state: int = 0
    last_transition_round: int | None = None

    def __post_init__(self) -> None:
        """Validate participant state invariants."""
        if not self.participant_id:
            raise ValueError("participant_id must not be empty")
        if not isinstance(self.state, ParticipantState):
            raise ValueError("state must be a ParticipantState")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.consecutive_positive_rounds, self.consecutive_negative_rounds, self.rounds_in_state)
        ):
            raise ValueError("CA history counters must be integers")
        if min(self.consecutive_positive_rounds, self.consecutive_negative_rounds, self.rounds_in_state) < 0:
            raise ValueError("CA history counters must be non-negative")
        if self.last_transition_round is not None and (
            isinstance(self.last_transition_round, bool)
            or not isinstance(self.last_transition_round, int)
            or self.last_transition_round < 0
        ):
            raise ValueError("last_transition_round must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class CATransitionPolicy:
    """Every threshold controlling CA evolution."""

    promotion_positive_rounds: int = 3
    recovery_positive_rounds: int = 3
    exclusion_negative_rounds: int = 2
    excluded_cooldown_rounds: int = 3
    trusted_neighbor_min_count: int = 1
    trusted_neighbor_min_fraction: float = 0.5
    promotion_min_trust: float = 0.70
    recovery_min_trust: float = 0.60
    recovery_enabled: bool = False
    require_undirected_topology: bool = True
    allow_self_neighbors: bool = False
    severe_trusted_target: ParticipantState = ParticipantState.SUSPICIOUS

    def __post_init__(self) -> None:
        """Validate policy invariants."""
        integer_thresholds = (
            self.promotion_positive_rounds,
            self.recovery_positive_rounds,
            self.exclusion_negative_rounds,
            self.excluded_cooldown_rounds,
            self.trusted_neighbor_min_count,
        )
        if any(value < 0 for value in integer_thresholds):
            raise ValueError("policy counters must be non-negative")
        for value in (self.trusted_neighbor_min_fraction, self.promotion_min_trust, self.recovery_min_trust):
            if not math.isfinite(value):
                raise ValueError("policy numeric thresholds must be finite")
        if not 0.0 <= self.trusted_neighbor_min_fraction <= 1.0:
            raise ValueError("trusted_neighbor_min_fraction must be between zero and one")
        if self.severe_trusted_target not in (ParticipantState.OBSERVATION, ParticipantState.SUSPICIOUS):
            raise ValueError("severe trusted transition may target only observation or suspicious")

    @property
    def policy_hash(self) -> str:
        """Return the domain-separated hash of this policy."""
        return canonical_hash("ca-transition-policy-v1", _dataclass_payload(self))


@dataclass(frozen=True, slots=True)
class NeighborStateSummary:
    """Canonical summary of direct neighbors from the previous snapshot."""

    total: int
    trusted: int
    observation: int
    suspicious: int
    excluded: int
    trusted_fraction: float


@dataclass(frozen=True, slots=True)
class CATransitionRecord:
    """Auditable explanation of one synchronous transition."""

    participant_id: str
    source_round: int
    previous_state: ParticipantState
    next_state: ParticipantState
    evidence_category: EvidenceCategory
    trust_score: float
    previous_positive_rounds: int
    previous_negative_rounds: int
    next_positive_rounds: int
    next_negative_rounds: int
    neighbor_summary: NeighborStateSummary
    reason_code: str
    policy_hash: str
    previous_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class CATransitionInput:
    """Frozen round evidence and topology supplied to the transition engine."""

    source_round: int
    trust_scores: Mapping[str, float]
    evidence: Mapping[str, EvidenceCategory]
    topology: Mapping[str, frozenset[str] | set[str] | tuple[str, ...] | list[str]]

    def __post_init__(self) -> None:
        """Defensively freeze all round inputs at construction time."""
        if isinstance(self.source_round, bool) or not isinstance(self.source_round, int):
            raise ValueError("source_round must be an integer")
        object.__setattr__(self, "trust_scores", MappingProxyType(dict(self.trust_scores)))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        frozen_topology = {participant_id: frozenset(neighbors) for participant_id, neighbors in self.topology.items()}
        object.__setattr__(self, "topology", MappingProxyType(frozen_topology))


@dataclass(frozen=True, slots=True)
class CAStateSnapshot:
    """Deeply immutable CA generation snapshot."""

    experiment_id: str
    generation: int
    participant_states: Mapping[str, ParticipantCAState]
    source_round: int | None
    previous_snapshot_hash: str | None
    policy_hash: str
    topology_hash: str
    snapshot_hash: str
    transition_records: tuple[CATransitionRecord, ...] = ()

    def __post_init__(self) -> None:
        """Validate and defensively freeze snapshot mappings."""
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        copied = dict(sorted(self.participant_states.items()))
        if any(not isinstance(value, ParticipantCAState) for value in copied.values()):
            raise ValueError("participant_states must contain ParticipantCAState values")
        if any(key != value.participant_id for key, value in copied.items()):
            raise ValueError("participant state keys must match participant IDs")
        object.__setattr__(self, "participant_states", MappingProxyType(copied))
        object.__setattr__(self, "transition_records", tuple(self.transition_records))


def _dataclass_payload(value: object) -> dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _state_payload(state: ParticipantCAState) -> dict[str, object]:
    return _dataclass_payload(state)


class CATransitionEngine:
    """Stateless deterministic CA initialization and synchronous evolution."""

    @staticmethod
    def initialize(
        experiment_id: str,
        participant_ids: set[str] | frozenset[str] | tuple[str, ...] | list[str],
        policy: CATransitionPolicy,
        topology: Mapping[str, frozenset[str] | set[str] | tuple[str, ...] | list[str]],
        bootstrap_states: Mapping[str, ParticipantCAState] | None = None,
    ) -> CAStateSnapshot:
        """Create generation zero; absent bootstrap state always means observation."""
        ids = _validate_id_set(participant_ids)
        canonical_topology = _validate_topology(ids, topology, policy)
        if bootstrap_states is None:
            states = {participant_id: ParticipantCAState(participant_id) for participant_id in ids}
        else:
            if set(bootstrap_states) != set(ids):
                raise ValueError("bootstrap snapshot participant set is incomplete")
            states = {participant_id: bootstrap_states[participant_id] for participant_id in ids}
            if any(
                not isinstance(state, ParticipantCAState) or state.participant_id != participant_id
                for participant_id, state in states.items()
            ):
                raise ValueError("bootstrap state identity mismatch")
        topology_hash = _topology_hash(canonical_topology)
        payload = _snapshot_payload(experiment_id, 0, states, None, None, policy.policy_hash, topology_hash, ())
        return CAStateSnapshot(
            experiment_id, 0, states, None, None, policy.policy_hash, topology_hash, canonical_hash("ca-state-snapshot-v1", payload)
        )

    @staticmethod
    def transition(previous: CAStateSnapshot, transition_input: CATransitionInput, policy: CATransitionPolicy) -> CAStateSnapshot:
        """Evolve all participants using only the same immutable previous generation."""
        ids = tuple(previous.participant_states)
        expected_previous_hash = canonical_hash(
            "ca-state-snapshot-v1",
            _snapshot_payload(
                previous.experiment_id,
                previous.generation,
                previous.participant_states,
                previous.source_round,
                previous.previous_snapshot_hash,
                previous.policy_hash,
                previous.topology_hash,
                previous.transition_records,
            ),
        )
        if previous.snapshot_hash != expected_previous_hash:
            raise ValueError("previous snapshot hash verification failed")
        if previous.policy_hash != policy.policy_hash:
            raise ValueError("transition policy does not match previous snapshot")
        if transition_input.source_round < 0 or (
            previous.source_round is not None and transition_input.source_round <= previous.source_round
        ):
            raise ValueError("source round must advance")
        _validate_complete_mapping(ids, transition_input.trust_scores, "trust")
        _validate_complete_mapping(ids, transition_input.evidence, "evidence")
        topology = _validate_topology(ids, transition_input.topology, policy)
        topology_hash = _topology_hash(topology)
        records: list[CATransitionRecord] = []
        states: dict[str, ParticipantCAState] = {}
        for participant_id in ids:
            trust = transition_input.trust_scores[participant_id]
            if isinstance(trust, bool) or not isinstance(trust, int | float) or not math.isfinite(trust):
                raise ValueError(f"trust score for {participant_id!r} must be finite")
            evidence = transition_input.evidence[participant_id]
            if not isinstance(evidence, EvidenceCategory):
                raise ValueError(f"invalid evidence category for {participant_id!r}")
            old = previous.participant_states[participant_id]
            summary = _neighbor_summary(topology[participant_id], previous.participant_states)
            positive = old.consecutive_positive_rounds + 1 if evidence is EvidenceCategory.POSITIVE else 0
            negative = old.consecutive_negative_rounds + 1 if evidence in (EvidenceCategory.NEGATIVE, EvidenceCategory.SEVERE) else 0
            next_state, reason = _next_state(old, evidence, float(trust), positive, negative, summary, policy)
            changed = next_state is not old.state
            state = ParticipantCAState(
                participant_id,
                next_state,
                positive,
                negative,
                0 if changed else old.rounds_in_state + 1,
                transition_input.source_round if changed else old.last_transition_round,
            )
            states[participant_id] = state
            records.append(
                CATransitionRecord(
                    participant_id,
                    transition_input.source_round,
                    old.state,
                    next_state,
                    evidence,
                    float(trust),
                    old.consecutive_positive_rounds,
                    old.consecutive_negative_rounds,
                    positive,
                    negative,
                    summary,
                    reason,
                    policy.policy_hash,
                    previous.snapshot_hash,
                )
            )
        record_tuple = tuple(records)
        payload = _snapshot_payload(
            previous.experiment_id,
            previous.generation + 1,
            states,
            transition_input.source_round,
            previous.snapshot_hash,
            policy.policy_hash,
            topology_hash,
            record_tuple,
        )
        return CAStateSnapshot(
            previous.experiment_id,
            previous.generation + 1,
            states,
            transition_input.source_round,
            previous.snapshot_hash,
            policy.policy_hash,
            topology_hash,
            canonical_hash("ca-state-snapshot-v1", payload),
            record_tuple,
        )


def _validate_id_set(participant_ids: object) -> tuple[str, ...]:
    ids = tuple(sorted(participant_ids))  # type: ignore[arg-type]
    if not ids or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("participant IDs must be a non-empty unique collection of non-empty strings")
    return ids


def _validate_complete_mapping(ids: tuple[str, ...], mapping: Mapping[str, object], label: str) -> None:
    if set(mapping) != set(ids):
        raise ValueError(f"{label} participant set is incomplete or contains unknown identities")


def _validate_topology(ids: tuple[str, ...], topology: Mapping[str, object], policy: CATransitionPolicy) -> dict[str, tuple[str, ...]]:
    _validate_complete_mapping(ids, topology, "topology")
    known = set(ids)
    result: dict[str, tuple[str, ...]] = {}
    for participant_id in ids:
        raw = topology[participant_id]
        if isinstance(raw, str):
            raise ValueError("neighbor collections must not be strings")
        neighbors = tuple(sorted(raw))  # type: ignore[arg-type]
        if len(set(neighbors)) != len(neighbors):
            raise ValueError("duplicate neighbor identity")
        if not set(neighbors) <= known:
            raise ValueError("topology contains unknown participant")
        if participant_id in neighbors and not policy.allow_self_neighbors:
            raise ValueError("self-neighbor edge is not allowed")
        result[participant_id] = neighbors
    if policy.require_undirected_topology:
        for participant_id, neighbors in result.items():
            if any(participant_id not in result[neighbor] for neighbor in neighbors):
                raise ValueError("topology must be symmetric")
    return result


def _topology_hash(topology: Mapping[str, tuple[str, ...]]) -> str:
    return canonical_hash("ca-topology-v1", {participant_id: list(topology[participant_id]) for participant_id in sorted(topology)})


def _neighbor_summary(neighbors: tuple[str, ...], states: Mapping[str, ParticipantCAState]) -> NeighborStateSummary:
    counts = {state: 0 for state in ParticipantState}
    for neighbor in neighbors:
        counts[states[neighbor].state] += 1
    total = len(neighbors)
    return NeighborStateSummary(
        total,
        counts[ParticipantState.TRUSTED],
        counts[ParticipantState.OBSERVATION],
        counts[ParticipantState.SUSPICIOUS],
        counts[ParticipantState.EXCLUDED],
        counts[ParticipantState.TRUSTED] / total if total else 0.0,
    )


def _supported(summary: NeighborStateSummary, policy: CATransitionPolicy) -> bool:
    return summary.trusted >= policy.trusted_neighbor_min_count and summary.trusted_fraction >= policy.trusted_neighbor_min_fraction


def _next_state(
    old: ParticipantCAState,
    evidence: EvidenceCategory,
    trust: float,
    positive: int,
    negative: int,
    neighbors: NeighborStateSummary,
    policy: CATransitionPolicy,
) -> tuple[ParticipantState, str]:
    state = old.state
    if state is ParticipantState.OBSERVATION:
        if evidence is EvidenceCategory.SEVERE:
            return ParticipantState.SUSPICIOUS, "observation_severe"
        if evidence is EvidenceCategory.NEGATIVE:
            return ParticipantState.SUSPICIOUS, "observation_negative"
        if (
            evidence is EvidenceCategory.POSITIVE
            and positive >= policy.promotion_positive_rounds
            and trust >= policy.promotion_min_trust
            and _supported(neighbors, policy)
        ):
            return ParticipantState.TRUSTED, "observation_promoted"
    elif state is ParticipantState.TRUSTED:
        if evidence is EvidenceCategory.SEVERE:
            return policy.severe_trusted_target, "trusted_severe"
        if evidence is EvidenceCategory.NEGATIVE:
            return ParticipantState.OBSERVATION, "trusted_negative"
    elif state is ParticipantState.SUSPICIOUS:
        if evidence in (EvidenceCategory.NEGATIVE, EvidenceCategory.SEVERE) and negative >= policy.exclusion_negative_rounds:
            return ParticipantState.EXCLUDED, "suspicious_repeated_negative"
        if (
            evidence is EvidenceCategory.POSITIVE
            and positive >= policy.recovery_positive_rounds
            and trust >= policy.recovery_min_trust
            and _supported(neighbors, policy)
        ):
            return ParticipantState.OBSERVATION, "suspicious_recovered"
    elif (
        policy.recovery_enabled
        and old.rounds_in_state >= policy.excluded_cooldown_rounds
        and evidence is EvidenceCategory.POSITIVE
        and positive >= policy.recovery_positive_rounds
        and trust >= policy.recovery_min_trust
        and _supported(neighbors, policy)
    ):
        return ParticipantState.SUSPICIOUS, "excluded_cooldown_recovery"
    return state, "state_held"


def _record_payload(record: CATransitionRecord) -> dict[str, object]:
    payload = _dataclass_payload(record)
    payload["neighbor_summary"] = _dataclass_payload(record.neighbor_summary)
    return payload


def _snapshot_payload(
    experiment_id: str,
    generation: int,
    states: Mapping[str, ParticipantCAState],
    source_round: int | None,
    previous_hash: str | None,
    policy_hash: str,
    topology_hash: str,
    records: tuple[CATransitionRecord, ...],
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "generation": generation,
        "participant_states": {key: _state_payload(states[key]) for key in sorted(states)},
        "source_round": source_round,
        "previous_snapshot_hash": previous_hash,
        "policy_hash": policy_hash,
        "topology_hash": topology_hash,
        "transition_records": [_record_payload(record) for record in records],
    }
