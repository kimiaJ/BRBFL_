# ruff: noqa: D102, D105, D107
"""Round-scoped participant roles and selection abstractions."""  # noqa: D102, D105, D107

from __future__ import annotations

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
