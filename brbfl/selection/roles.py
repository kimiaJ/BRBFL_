# ruff: noqa: D102, D105, D107
"""Round-scoped participant roles and selection abstractions."""  # noqa: D102, D105, D107

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from brbfl.canonical import canonical_hash

CONTRIBUTOR = "contributor"
VALIDATOR = "validator"
AGGREGATOR = "aggregator"
DETECTOR = "detector"


@dataclass(frozen=True, order=True)
class Role:
    """A responsibility selected for exactly one node and round."""

    node_id: str
    round_number: int
    name: str


@dataclass(frozen=True)
class SelectionContext:
    """Immutable input supplied to a round-role selector."""

    experiment_id: str
    round_number: int
    participant_capabilities: Mapping[str, frozenset[str]]
    previous_state_hash: str | None = None
    trust_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "participant_capabilities", MappingProxyType(dict(self.participant_capabilities)))
        object.__setattr__(self, "trust_scores", MappingProxyType(dict(sorted(self.trust_scores.items()))))


@dataclass(frozen=True)
class RoundRoleAssignment:
    """Canonical, immutable selection of responsibilities for a single round."""

    experiment_id: str
    round_number: int
    network_participants: tuple[str, ...]
    selected_contributors: tuple[str, ...]
    selected_validators: tuple[str, ...]
    detector_subgroups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    aggregation_eligible_nodes: tuple[str, ...] = ()
    selection_source: str = "static"
    previous_state_hash: str | None = None
    assignment_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("network_participants", "selected_contributors", "selected_validators", "aggregation_eligible_nodes"):
            object.__setattr__(self, name, tuple(sorted(set(getattr(self, name)))))
        groups = {name: tuple(sorted(set(nodes))) for name, nodes in sorted(self.detector_subgroups.items())}
        object.__setattr__(self, "detector_subgroups", MappingProxyType(groups))
        if self.round_number < 0:
            raise ValueError("round_number must be non-negative")
        participants = set(self.network_participants)
        referenced = set(self.selected_contributors) | set(self.selected_validators) | set(self.aggregation_eligible_nodes)
        referenced.update(node for nodes in groups.values() for node in nodes)
        unknown = referenced - participants
        if unknown:
            raise ValueError(f"round roles reference unknown participants: {sorted(unknown)}")
        object.__setattr__(self, "assignment_hash", canonical_hash("RoundRoleAssignment/v1", self.canonical_payload()))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "round_number": self.round_number,
            "network_participants": self.network_participants,
            "selected_contributors": self.selected_contributors,
            "selected_validators": self.selected_validators,
            "detector_subgroups": dict(self.detector_subgroups),
            "aggregation_eligible_nodes": self.aggregation_eligible_nodes,
            "selection_source": self.selection_source,
            "previous_state_hash": self.previous_state_hash,
        }

    def roles(self) -> tuple[Role, ...]:
        roles = [Role(node, self.round_number, CONTRIBUTOR) for node in self.selected_contributors]
        roles.extend(Role(node, self.round_number, VALIDATOR) for node in self.selected_validators)
        roles.extend(Role(node, self.round_number, AGGREGATOR) for node in self.aggregation_eligible_nodes)
        return tuple(sorted(roles))

    def verify_hash(self) -> bool:
        return self.assignment_hash == canonical_hash("RoundRoleAssignment/v1", self.canonical_payload())


class RoundRoleSelector(ABC):
    """Extension point for static bootstrap and future CA-backed selection."""

    @abstractmethod
    def select_roles(self, context: SelectionContext) -> RoundRoleAssignment:
        """Return the finalized assignment for ``context.round_number``."""


class StaticRoundRoleSelector(RoundRoleSelector):
    """Repeat configured roles, or use an explicit per-round static fixture."""

    def __init__(
        self,
        contributors: tuple[str, ...],
        validators: tuple[str, ...],
        aggregation_eligible_nodes: tuple[str, ...] = (),
        detector_subgroups: Mapping[str, tuple[str, ...]] | None = None,
        per_round: Mapping[int, RoundRoleAssignment] | None = None,
    ) -> None:
        self._contributors = contributors
        self._validators = validators
        self._aggregators = aggregation_eligible_nodes
        self._detectors = detector_subgroups or {}
        self._per_round = dict(per_round or {})

    def select_roles(self, context: SelectionContext) -> RoundRoleAssignment:
        configured = self._per_round.get(context.round_number)
        if configured is not None:
            assignment = replace(
                configured,
                experiment_id=context.experiment_id,
                round_number=context.round_number,
                previous_state_hash=context.previous_state_hash,
            )
        else:
            assignment = RoundRoleAssignment(
                experiment_id=context.experiment_id,
                round_number=context.round_number,
                network_participants=tuple(context.participant_capabilities),
                selected_contributors=self._contributors,
                selected_validators=self._validators,
                detector_subgroups=self._detectors,
                aggregation_eligible_nodes=self._aggregators,
                previous_state_hash=context.previous_state_hash,
            )
        validate_capabilities(assignment, context.participant_capabilities)
        return assignment


class TrustRankedValidatorSelector(RoundRoleSelector):
    """Keep static roles during bootstrap, then rank validators by prior finalized trust."""

    def __init__(
        self,
        bootstrap: StaticRoundRoleSelector,
        eligible_participants: tuple[str, ...],
        target_count: int,
        minimum_trust: float = 0.5,
        bootstrap_rounds: int = 1,
    ) -> None:
        self._bootstrap = bootstrap
        self.eligible = tuple(eligible_participants)
        self.target_count = target_count
        self.minimum_trust = minimum_trust
        self.bootstrap_rounds = bootstrap_rounds
        self._assignments: dict[int, RoundRoleAssignment] = {}
        self._evidence: dict[int, dict[str, object]] = {}
        if len(set(self.eligible)) != len(self.eligible):
            raise ValueError("duplicate eligible participant IDs")
        if target_count <= 0 or target_count > len(self.eligible):
            raise ValueError("invalid validator target_count")
        if bootstrap_rounds < 1 or not math.isfinite(minimum_trust) or not 0 <= minimum_trust <= 1:
            raise ValueError("invalid trust-ranked selection policy")

    def select_roles(self, context: SelectionContext) -> RoundRoleAssignment:
        if context.round_number in self._assignments:
            raise RuntimeError("round role assignment is already frozen")
        base = self._bootstrap.select_roles(context)
        missing = set(self.eligible) - set(context.participant_capabilities)
        if missing:
            raise ValueError(f"unknown eligible participants: {sorted(missing)}")
        scores = dict(context.trust_scores)
        if context.round_number < self.bootstrap_rounds:
            if len(base.selected_validators) != self.target_count or not set(base.selected_validators) <= set(self.eligible):
                raise RuntimeError("invalid bootstrap validator assignment")
            selected = base.selected_validators
            mode, source, reason, ranking = "bootstrap", None, "bootstrap_static_assignment", ()
        else:
            if context.previous_state_hash is None:
                raise RuntimeError("finalized prior-round state is unavailable")
            if set(scores) != set(self.eligible):
                raise RuntimeError("complete eligible-pool trust snapshot is required")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
                raise RuntimeError("invalid trust score")
            ranking = tuple(
                sorted((node for node in self.eligible if scores[node] >= self.minimum_trust), key=lambda node: (-scores[node], node))
            )
            if len(ranking) < self.target_count:
                raise RuntimeError("insufficient participants meet minimum_trust")
            selected = ranking[: self.target_count]
            mode, source, reason = "trust_ranked", context.round_number - 1, "highest_eligible_finalized_trust"
        assignment = replace(base, selected_validators=tuple(selected), selection_source=mode)
        validate_capabilities(assignment, context.participant_capabilities)
        evidence = {
            "round": context.round_number,
            "mode": mode,
            "trust_source_round": source,
            "eligible_participants": sorted(self.eligible),
            "pre_selection_scores": {node: scores[node] for node in sorted(scores)},
            "filtered_participants": list(ranking),
            "ranking": list(ranking),
            "selected_validators": list(assignment.selected_validators),
            "excluded_participants": sorted(set(self.eligible) - set(selected)),
            "assignment_hash": assignment.assignment_hash,
            "reason": reason,
        }
        evidence["selection_sha256"] = canonical_hash("TrustRankedSelection/v1", evidence)
        self._assignments[context.round_number] = assignment
        self._evidence[context.round_number] = evidence
        return assignment

    def artifact(self) -> dict[str, object]:
        return {
            "strategy": "trust_ranked",
            "dynamic_validator_selection": True,
            "bootstrap_rounds": self.bootstrap_rounds,
            "target_count": self.target_count,
            "minimum_trust": self.minimum_trust,
            "tie_breaker": "node_id",
            "eligible_participants": sorted(self.eligible),
            "rounds": {str(k): v for k, v in sorted(self._evidence.items())},
            "verification_result": True,
            "verification_reason": "verified",
        }


def validate_capabilities(assignment: RoundRoleAssignment, capabilities: Mapping[str, frozenset[str]]) -> None:
    """Fail descriptively unless every selected responsibility is registered."""
    unknown = set(assignment.network_participants) - set(capabilities)
    if unknown:
        raise ValueError(f"unknown registered participants: {sorted(unknown)}")
    required: dict[str, set[str]] = {
        CONTRIBUTOR: set(assignment.selected_contributors),
        VALIDATOR: set(assignment.selected_validators),
        AGGREGATOR: set(assignment.aggregation_eligible_nodes),
    }
    for nodes in assignment.detector_subgroups.values():
        required.setdefault(DETECTOR, set()).update(nodes)
    invalid = [(node, role) for role, nodes in required.items() for node in sorted(nodes) if role not in capabilities[node]]
    if invalid:
        raise ValueError(f"selected roles exceed registered capabilities: {invalid}")
