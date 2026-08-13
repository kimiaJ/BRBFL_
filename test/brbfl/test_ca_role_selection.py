# ruff: noqa: D103
"""CA-aware, round-scoped role selection tests."""

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
from brbfl.selection import CARoleSelectionPolicy, CAStateRoleSelector, SelectionContext, StaticRoundRoleSelector

CAPS = {node: frozenset({"contributor", "validator", "aggregator"}) for node in "abcde"}
BOOTSTRAP = StaticRoundRoleSelector(("a", "b", "c"), ("a", "b"), ())


def snapshot(states, experiment="exp"):
    ids = tuple(states)
    topology = {node: tuple(other for other in ids if other != node) for node in ids}
    initial = CATransitionEngine.initialize(
        experiment,
        ids,
        CATransitionPolicy(),
        topology,
        {node: ParticipantCAState(node, state) for node, state in states.items()},
    )
    return CATransitionEngine.transition(
        initial,
        CATransitionInput(
            0,
            dict.fromkeys(ids, 0.5),
            dict.fromkeys(ids, EvidenceCategory.NEUTRAL),
            topology,
        ),
        CATransitionPolicy(),
    )


def context(value, scores=None, capabilities=CAPS):
    return SelectionContext(
        "exp", 1, capabilities, value.snapshot_hash,
        scores or {node: 0.5 for node in capabilities}, value, 0, "trust-0",
    )


def selector(**kwargs):
    return CAStateRoleSelector(BOOTSTRAP, CARoleSelectionPolicy(3, 2, **kwargs))


def test_bootstrap_is_unchanged_and_next_round_uses_ca_provenance():
    value = selector()
    bootstrap = value.select_roles(SelectionContext("exp", 0, CAPS))
    assert bootstrap == BOOTSTRAP.select_roles(SelectionContext("exp", 0, CAPS))
    ca = snapshot(dict.fromkeys("abcde", ParticipantState.TRUSTED))
    selected = value.select_roles(context(ca))
    assert selected.selection_source == "ca_state"
    assert (selected.source_ca_generation, selected.source_ca_snapshot_hash) == (1, ca.snapshot_hash)
    assert (selected.source_trust_round, selected.source_trust_hash) == (0, "trust-0")
    assert selected.policy_hash == value.policy.policy_hash


def test_state_eligibility_priority_trust_and_id_tiebreak():
    ca = snapshot({"a": ParticipantState.OBSERVATION, "b": ParticipantState.TRUSTED,
                   "c": ParticipantState.OBSERVATION, "d": ParticipantState.SUSPICIOUS,
                   "e": ParticipantState.EXCLUDED})
    selected = selector().select_roles(context(ca, {"a": .99, "b": .1, "c": .5, "d": 1.0, "e": 1.0}))
    assert selected.selected_validators == ("a", "b")  # assignment storage is canonical, not ranking order
    assert set(selected.selected_contributors) == {"a", "b", "c"}
    assert "d" not in selected.selected_validators
    assert "e" not in selected.selected_contributors + selected.selected_validators
    tie = selector().select_roles(context(snapshot(dict.fromkeys("abcde", ParticipantState.OBSERVATION))))
    assert tie.selected_validators == ("a", "b")


def test_capabilities_shortage_and_policy_fail_closed():
    ca = snapshot(dict.fromkeys("abcde", ParticipantState.TRUSTED))
    limited = dict(CAPS)
    limited["a"] = frozenset({"contributor"})
    selected = selector().select_roles(context(ca, capabilities=limited))
    assert selected.selected_validators == ("b", "c")
    bad = snapshot({"a": ParticipantState.TRUSTED, "b": ParticipantState.EXCLUDED,
                    "c": ParticipantState.EXCLUDED, "d": ParticipantState.EXCLUDED,
                    "e": ParticipantState.EXCLUDED})
    with pytest.raises(RuntimeError, match="insufficient"):
        selector().select_roles(context(bad))
    with pytest.raises(ValueError, match="quorum"):
        CARoleSelectionPolicy(3, 1, validator_quorum=2)


def test_state_and_input_order_affect_hash_only_when_semantics_change():
    states = {"a": ParticipantState.TRUSTED, "b": ParticipantState.OBSERVATION,
              "c": ParticipantState.OBSERVATION, "d": ParticipantState.SUSPICIOUS,
              "e": ParticipantState.EXCLUDED}
    one = selector().select_roles(context(snapshot(states)))
    reordered = dict(reversed(tuple(states.items())))
    reversed_caps = dict(reversed(tuple(CAPS.items())))
    two = selector().select_roles(context(snapshot(reordered), capabilities=reversed_caps))
    assert one.assignment_hash == two.assignment_hash
    changed = dict(states, a=ParticipantState.EXCLUDED, e=ParticipantState.TRUSTED)
    three = selector().select_roles(context(snapshot(changed)))
    assert one.assignment_hash != three.assignment_hash


def test_suspicious_contributor_setting_and_stale_inputs():
    ca = snapshot({"a": ParticipantState.TRUSTED, "b": ParticipantState.OBSERVATION,
                   "c": ParticipantState.SUSPICIOUS, "d": ParticipantState.SUSPICIOUS,
                   "e": ParticipantState.EXCLUDED})
    assert "c" in selector().select_roles(context(ca)).selected_contributors
    with pytest.raises(RuntimeError, match="insufficient"):
        selector(suspicious_contributors=False).select_roles(context(ca))
    with pytest.raises(RuntimeError, match="prior finalized"):
        selector().select_roles(replace(context(ca), ca_snapshot=replace(ca, source_round=None)))
