"""Focused tests for the pure participant cellular automaton."""

# ruff: noqa: D103

from __future__ import annotations

from dataclasses import replace

import pytest

from brbfl.ca import (
    CATransitionEngine,
    CATransitionInput,
    CATransitionPolicy,
    EvidenceCategory,
    ParticipantCAState,
    ParticipantState,
)


def topology(*ids: str) -> dict[str, set[str]]:
    """Return a complete undirected topology."""
    return {participant_id: set(ids) - {participant_id} for participant_id in ids}


def initialized(policy: CATransitionPolicy | None = None, states: dict[str, ParticipantCAState] | None = None, experiment: str = "exp"):
    """Build a three-cell snapshot."""
    selected = policy or CATransitionPolicy()
    return CATransitionEngine.initialize(experiment, ["a", "b", "c"], selected, topology("a", "b", "c"), states)


def evolve(snapshot, policy, evidence, trusts=None, graph=None, round_number=0):
    """Apply one complete transition."""
    return CATransitionEngine.transition(
        snapshot,
        CATransitionInput(
            round_number,
            trusts or dict.fromkeys(snapshot.participant_states, 0.8),
            evidence,
            graph or topology(*snapshot.participant_states),
        ),
        policy,
    )


def test_new_participants_are_observation_and_snapshot_is_immutable() -> None:
    policy = CATransitionPolicy()
    snapshot = initialized(policy)
    assert {value.state for value in snapshot.participant_states.values()} == {ParticipantState.OBSERVATION}
    with pytest.raises(TypeError):
        snapshot.participant_states["a"] = ParticipantCAState("a")  # type: ignore[index]


def test_promotion_requires_history_trust_and_neighborhood_support() -> None:
    policy = CATransitionPolicy(promotion_positive_rounds=2, trusted_neighbor_min_count=1)
    states = {
        "a": ParticipantCAState("a"),
        "b": ParticipantCAState("b", ParticipantState.TRUSTED),
        "c": ParticipantCAState("c"),
    }
    snapshot = initialized(policy, states)
    positive = dict.fromkeys(states, EvidenceCategory.POSITIVE)
    snapshot = evolve(snapshot, policy, positive)
    assert snapshot.participant_states["a"].state is ParticipantState.OBSERVATION
    snapshot = evolve(snapshot, policy, positive, round_number=1)
    assert snapshot.participant_states["a"].state is ParticipantState.TRUSTED

    unsupported = initialized(policy)
    unsupported = evolve(unsupported, policy, positive)
    unsupported = evolve(unsupported, policy, positive, round_number=1)
    assert unsupported.participant_states["a"].state is ParticipantState.OBSERVATION


def test_negative_degradation_severe_and_exclusion() -> None:
    policy = CATransitionPolicy(exclusion_negative_rounds=2)
    states = {
        "a": ParticipantCAState("a"),
        "b": ParticipantCAState("b", ParticipantState.TRUSTED),
        "c": ParticipantCAState("c", ParticipantState.TRUSTED),
    }
    snapshot = initialized(policy, states)
    evidence = {"a": EvidenceCategory.NEGATIVE, "b": EvidenceCategory.NEGATIVE, "c": EvidenceCategory.SEVERE}
    snapshot = evolve(snapshot, policy, evidence)
    assert snapshot.participant_states["a"].state is ParticipantState.SUSPICIOUS
    assert snapshot.participant_states["b"].state is ParticipantState.OBSERVATION
    assert snapshot.participant_states["c"].state is ParticipantState.SUSPICIOUS
    snapshot = evolve(snapshot, policy, evidence, round_number=1)
    assert snapshot.participant_states["a"].state is ParticipantState.EXCLUDED
    assert snapshot.participant_states["c"].state is ParticipantState.EXCLUDED


def test_severe_probation_retains_provenance_across_neutral_quarantine() -> None:
    """A validator cannot evade its severe finding by becoming unevaluable."""
    policy = CATransitionPolicy(severe_suspicious_probation_rounds=1)
    snapshot = initialized(policy)
    severe = dict.fromkeys(snapshot.participant_states, EvidenceCategory.NEUTRAL)
    severe["a"] = EvidenceCategory.SEVERE
    suspicious = evolve(snapshot, policy, severe)
    state = suspicious.participant_states["a"]
    assert state.state is ParticipantState.SUSPICIOUS
    assert state.unresolved_severe_since_round == 0
    assert snapshot.participant_states["a"].unresolved_severe_since_round is None

    neutral = dict.fromkeys(snapshot.participant_states, EvidenceCategory.NEUTRAL)
    excluded = evolve(suspicious, policy, neutral, round_number=1)
    assert excluded.participant_states["a"].state is ParticipantState.EXCLUDED
    assert excluded.participant_states["a"].unresolved_severe_since_round == 0
    assert excluded.participant_states["a"].consecutive_negative_rounds == 0
    assert excluded.transition_records[0].reason_code == "suspicious_severe_probation_expired"


def test_nonsevere_suspicion_uses_negative_threshold_not_severe_probation() -> None:
    policy = CATransitionPolicy(exclusion_negative_rounds=3, severe_suspicious_probation_rounds=1)
    snapshot = initialized(policy)
    negative = dict.fromkeys(snapshot.participant_states, EvidenceCategory.NEUTRAL)
    negative["a"] = EvidenceCategory.NEGATIVE
    suspicious = evolve(snapshot, policy, negative)
    assert suspicious.participant_states["a"].unresolved_severe_since_round is None
    held = evolve(suspicious, policy, dict.fromkeys(negative, EvidenceCategory.NEUTRAL), round_number=1)
    assert held.participant_states["a"].state is ParticipantState.SUSPICIOUS


def test_positive_recovery_can_resolve_severe_provenance_only_when_enabled() -> None:
    policy = CATransitionPolicy(
        recovery_enabled=True,
        recovery_positive_rounds=1,
        recovery_min_trust=0,
        trusted_neighbor_min_count=0,
        trusted_neighbor_min_fraction=0,
        severe_suspicious_probation_rounds=1,
    )
    severe = {"a": EvidenceCategory.SEVERE, "b": EvidenceCategory.NEUTRAL, "c": EvidenceCategory.NEUTRAL}
    suspicious = evolve(initialized(policy), policy, severe)
    recovered = evolve(
        suspicious,
        policy,
        {"a": EvidenceCategory.POSITIVE, "b": EvidenceCategory.NEUTRAL, "c": EvidenceCategory.NEUTRAL},
        round_number=1,
    )
    assert recovered.participant_states["a"].state is ParticipantState.OBSERVATION
    assert recovered.participant_states["a"].unresolved_severe_since_round is None


def test_forged_severe_provenance_is_rejected_and_changes_hash() -> None:
    with pytest.raises(ValueError, match="requires suspicious or excluded"):
        ParticipantCAState("a", unresolved_severe_since_round=0)
    with pytest.raises(ValueError, match="backed by transition history"):
        ParticipantCAState("a", ParticipantState.SUSPICIOUS, unresolved_severe_since_round=0)
    policy = CATransitionPolicy()
    ordinary = initialized(
        policy,
        {
            "a": ParticipantCAState("a", ParticipantState.SUSPICIOUS),
            "b": ParticipantCAState("b"),
            "c": ParticipantCAState("c"),
        },
    )
    severe = initialized(
        policy,
        {
            "a": ParticipantCAState("a", ParticipantState.SUSPICIOUS, last_transition_round=0, unresolved_severe_since_round=0),
            "b": ParticipantCAState("b"),
            "c": ParticipantCAState("c"),
        },
    )
    assert ordinary.snapshot_hash != severe.snapshot_hash


def test_severe_is_not_cancelled_by_trusted_neighbors() -> None:
    policy = CATransitionPolicy()
    states = {key: ParticipantCAState(key, ParticipantState.TRUSTED) for key in ("a", "b", "c")}
    snapshot = initialized(policy, states)
    result = evolve(
        snapshot,
        policy,
        {"a": EvidenceCategory.SEVERE, "b": EvidenceCategory.POSITIVE, "c": EvidenceCategory.POSITIVE},
    )
    assert result.participant_states["a"].state is ParticipantState.SUSPICIOUS


def test_neutral_never_increments_positive_history() -> None:
    policy = CATransitionPolicy(promotion_positive_rounds=1)
    states = {
        "a": ParticipantCAState("a", consecutive_positive_rounds=8),
        "b": ParticipantCAState("b", ParticipantState.TRUSTED),
        "c": ParticipantCAState("c"),
    }
    result = evolve(initialized(policy, states), policy, dict.fromkeys(states, EvidenceCategory.NEUTRAL))
    assert result.participant_states["a"].consecutive_positive_rounds == 0
    assert result.participant_states["a"].state is ParticipantState.OBSERVATION


def test_excluded_recovery_is_stepwise_and_after_cooldown() -> None:
    policy = CATransitionPolicy(recovery_enabled=True, recovery_positive_rounds=2, excluded_cooldown_rounds=2)
    states = {
        "a": ParticipantCAState("a", ParticipantState.EXCLUDED),
        "b": ParticipantCAState("b", ParticipantState.TRUSTED),
        "c": ParticipantCAState("c", ParticipantState.TRUSTED),
    }
    snapshot = initialized(policy, states)
    positive = dict.fromkeys(states, EvidenceCategory.POSITIVE)
    for round_number in range(2):
        snapshot = evolve(snapshot, policy, positive, round_number=round_number)
        assert snapshot.participant_states["a"].state is ParticipantState.EXCLUDED
    snapshot = evolve(snapshot, policy, positive, round_number=2)
    assert snapshot.participant_states["a"].state is ParticipantState.SUSPICIOUS
    assert snapshot.participant_states["a"].state not in (ParticipantState.TRUSTED, ParticipantState.OBSERVATION)


def test_transitions_are_synchronous() -> None:
    policy = CATransitionPolicy(promotion_positive_rounds=1, trusted_neighbor_min_count=1)
    states = {
        "a": ParticipantCAState("a", ParticipantState.TRUSTED),
        "b": ParticipantCAState("b"),
        "c": ParticipantCAState("c"),
    }
    graph = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    result = evolve(initialized(policy, states), policy, dict.fromkeys(states, EvidenceCategory.POSITIVE), graph=graph)
    assert result.participant_states["b"].state is ParticipantState.TRUSTED
    assert result.participant_states["c"].state is ParticipantState.OBSERVATION


def test_ordering_does_not_affect_records_states_or_hashes() -> None:
    policy = CATransitionPolicy()
    first = CATransitionEngine.initialize("exp", ["a", "b", "c"], policy, topology("a", "b", "c"))
    reversed_graph = {key: list(reversed(tuple(value))) for key, value in reversed(tuple(topology("a", "b", "c").items()))}
    second = CATransitionEngine.initialize("exp", ["c", "b", "a"], policy, reversed_graph)
    first_input = CATransitionInput(
        0,
        {"a": 0.1, "b": 0.8, "c": 0.5},
        {"a": EvidenceCategory.NEGATIVE, "b": EvidenceCategory.POSITIVE, "c": EvidenceCategory.NEUTRAL},
        topology("a", "b", "c"),
    )
    second_input = CATransitionInput(
        0, dict(reversed(tuple(first_input.trust_scores.items()))), dict(reversed(tuple(first_input.evidence.items()))), reversed_graph
    )
    one = CATransitionEngine.transition(first, first_input, policy)
    two = CATransitionEngine.transition(second, second_input, policy)
    assert one.participant_states == two.participant_states
    assert one.transition_records == two.transition_records
    assert one.snapshot_hash == two.snapshot_hash


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_trust_fails_closed(bad: float) -> None:
    snapshot = initialized()
    with pytest.raises(ValueError, match="finite"):
        evolve(
            snapshot,
            CATransitionPolicy(),
            dict.fromkeys(snapshot.participant_states, EvidenceCategory.NEUTRAL),
            {"a": bad, "b": 0.5, "c": 0.5},
        )


def test_missing_unknown_and_asymmetric_topology_fail_closed() -> None:
    policy = CATransitionPolicy()
    snapshot = initialized(policy)
    evidence = dict.fromkeys(snapshot.participant_states, EvidenceCategory.NEUTRAL)
    with pytest.raises(ValueError, match="incomplete"):
        evolve(snapshot, policy, evidence, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="unknown"):
        evolve(snapshot, policy, evidence, graph={"a": {"b", "x"}, "b": {"a"}, "c": set()})
    with pytest.raises(ValueError, match="symmetric"):
        evolve(snapshot, policy, evidence, graph={"a": {"b"}, "b": set(), "c": set()})


def test_inputs_are_not_mutated_and_experiments_do_not_share_state() -> None:
    policy = CATransitionPolicy()
    graph = topology("a", "b", "c")
    trusts = dict.fromkeys(graph, 0.5)
    evidence = dict.fromkeys(graph, EvidenceCategory.NEUTRAL)
    before = ({key: set(value) for key, value in graph.items()}, dict(trusts), dict(evidence))
    one = CATransitionEngine.initialize("one", list(graph), policy, graph)
    CATransitionEngine.transition(one, CATransitionInput(0, trusts, evidence, graph), policy)
    assert (graph, trusts, evidence) == before
    two = CATransitionEngine.initialize("two", list(graph), policy, graph)
    assert two.experiment_id == "two" and two.snapshot_hash != one.snapshot_hash
    assert all(state.rounds_in_state == 0 for state in two.participant_states.values())


def test_hashes_cover_policy_topology_state_and_evidence_and_are_repeatable() -> None:
    policy = CATransitionPolicy()
    base = initialized(policy)
    assert base.snapshot_hash == initialized(policy).snapshot_hash
    assert base.snapshot_hash != initialized(replace(policy, promotion_positive_rounds=4)).snapshot_hash
    sparse = {"a": {"b"}, "b": {"a"}, "c": set()}
    assert base.snapshot_hash != CATransitionEngine.initialize("exp", ["a", "b", "c"], policy, sparse).snapshot_hash
    changed_states = dict(base.participant_states)
    changed_states["a"] = ParticipantCAState("a", ParticipantState.SUSPICIOUS)
    assert base.snapshot_hash != initialized(policy, changed_states).snapshot_hash
    neutral = dict.fromkeys(base.participant_states, EvidenceCategory.NEUTRAL)
    positive = dict(neutral, a=EvidenceCategory.POSITIVE)
    assert evolve(base, policy, neutral).snapshot_hash != evolve(base, policy, positive).snapshot_hash
