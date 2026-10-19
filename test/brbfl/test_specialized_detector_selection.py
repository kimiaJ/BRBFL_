"""CA eligibility and provenance of specialized detector committees."""

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
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig
from brbfl.selection import CARoleSelectionPolicy, CAStateRoleSelector, SelectionContext, StaticRoundRoleSelector

CAPABILITY = "model_update_detector"
CAPS = {node: frozenset({"contributor", "validator", "detector", CAPABILITY}) for node in "abcde"}


def snapshot(states):
    """Build a finalized CA snapshot without importing another test module."""
    ids = tuple(states)
    topology = {node: tuple(other for other in ids if other != node) for node in ids}
    policy = CATransitionPolicy()
    initial = CATransitionEngine.initialize(
        "exp",
        ids,
        policy,
        topology,
        {node: ParticipantCAState(node, state) for node, state in states.items()},
    )
    return CATransitionEngine.transition(
        initial,
        CATransitionInput(0, dict.fromkeys(ids, 0.5), dict.fromkeys(ids, EvidenceCategory.NEUTRAL), topology),
        policy,
    )


def context(value, scores=None, capabilities=CAPS):
    """Bind selection to the verified prior-round state and trust provenance."""
    return SelectionContext(
        "exp",
        1,
        capabilities,
        value.snapshot_hash,
        scores if scores is not None else dict.fromkeys(capabilities, 0.5),
        value,
        0,
        "trust-0",
    )


def selector():
    """Create a selector with a two-member specialist committee."""
    return CAStateRoleSelector(
        StaticRoundRoleSelector(("a", "b", "c"), ("a", "b")),
        CARoleSelectionPolicy(3, 2, detector_target_counts={CAPABILITY: 2}),
    )


def test_bootstrap_and_ca_rotation_use_specialized_capabilities_and_prior_state():
    """A high-trust suspicious member loses its sensitive committee role."""
    policy = selector()
    initial = policy.select_roles(SelectionContext("exp", 0, CAPS))
    assert initial.detector_subgroups[CAPABILITY] == ("a", "b")
    ca = snapshot(
        {
            "a": ParticipantState.SUSPICIOUS,
            "b": ParticipantState.OBSERVATION,
            "c": ParticipantState.TRUSTED,
            "d": ParticipantState.OBSERVATION,
            "e": ParticipantState.EXCLUDED,
        }
    )
    ctx = context(ca, {"a": 1.0, "b": 0.6, "c": 0.1, "d": 0.9, "e": 1.0}, CAPS)
    selected = policy.select_roles(ctx)
    assert selected.detector_subgroups[CAPABILITY] == ("c", "d")
    assert selected.source_ca_snapshot_hash == ca.snapshot_hash
    assert selected.verify_hash()
    reordered = replace(ctx, participant_capabilities=dict(reversed(tuple(CAPS.items()))))
    assert policy.select_roles(reordered).assignment_hash == selected.assignment_hash


def test_general_detector_capability_is_insufficient_and_shortage_fails_closed():
    """Only registered specialists may fill the committee, including bootstrap."""
    limited = {node: caps - {CAPABILITY} for node, caps in CAPS.items()}
    with pytest.raises(RuntimeError, match="insufficient"):
        selector().select_roles(SelectionContext("exp", 0, limited))
    ca = snapshot(dict.fromkeys(CAPS, ParticipantState.OBSERVATION))
    with pytest.raises(RuntimeError, match="insufficient"):
        selector().select_roles(context(ca, capabilities=limited))


def test_committee_policy_is_immutable_and_changes_assignment_identity():
    """Committee cardinality is part of the canonical selection policy."""
    targets = {CAPABILITY: 2}
    policy = CARoleSelectionPolicy(3, 2, detector_target_counts=targets)
    original = policy.policy_hash
    targets[CAPABILITY] = 3
    assert policy.policy_hash == original
    assert replace(policy, detector_target_counts=targets).policy_hash != original
    for count in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            CARoleSelectionPolicy(3, 2, detector_target_counts={CAPABILITY: count})


def test_runtime_commits_specialized_committee_in_authoritative_round_assignment():
    """Runtime registration feeds the same selector used by the ledger workflow."""
    runtime = RuntimeLedgerAdapter(
        RuntimeLedgerConfig(
            enabled=True,
            trust_enabled=True,
            ca_enabled=True,
            selection_strategy="ca_state",
            validator_eligible_participants=tuple("abcde"),
            detector_target_counts={CAPABILITY: 2},
            detector_capabilities={node: (CAPABILITY,) for node in "bcd"},
        ),
        "exp",
        tuple("abcde"),
        tuple("abc"),
        tuple("ab"),
    )
    runtime.open_round(0, "a" * 64)
    assignment = runtime.role_assignment(0)
    assert assignment.detector_subgroups[CAPABILITY] == ("b", "c")
    assert assignment.verify_hash()
