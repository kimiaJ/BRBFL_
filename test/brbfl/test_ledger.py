# ruff: noqa: D103
"""Deterministic ledger semantic and workflow tests."""  # noqa: D103

from dataclasses import replace

import pytest

from brbfl.ledger import InMemoryLedger, canonical_bytes, canonical_hash, create_ledger, disabled_ledger_artifact
from brbfl.selection.roles import RoundRoleAssignment


def assignment() -> RoundRoleAssignment:
    return RoundRoleAssignment(
        "exp",
        0,
        tuple(f"node-{i}" for i in range(5)),
        ("node-0", "node-1", "node-2"),
        ("node-0", "node-3", "node-4"),
        aggregation_eligible_nodes=("node-0",),
    )


def prepared() -> InMemoryLedger:
    ledger = InMemoryLedger()
    nodes = assignment().network_participants
    ledger.start_experiment("exp", nodes)
    for node in nodes:
        capabilities = {"contributor", "validator"}
        if node == "node-0":
            capabilities.add("aggregator")
        ledger.register_participant("exp", node, frozenset(capabilities))
    ledger.commit_round_roles(assignment())
    ledger.open_round("exp", 0, "parent")
    return ledger


def test_canonical_encoding_and_domain_hashes():
    assert canonical_bytes({"z": None, "a": [True, 2]}) == b'{"a":[true,2],"z":null}'
    assert canonical_hash("a", {"x": 1}) == canonical_hash("a", {"x": 1})
    assert canonical_hash("a", {"x": 1}) != canonical_hash("b", {"x": 1})


def test_registration_and_conflicting_duplicates():
    ledger = InMemoryLedger()
    first = ledger.start_experiment("exp", ())
    assert ledger.start_experiment("exp", ()) == first
    receipt = ledger.register_participant("exp", "a", frozenset({"validator"}))
    assert ledger.register_participant("exp", "a", frozenset({"validator"})) == receipt
    with pytest.raises(RuntimeError, match="conflicting"):
        ledger.register_participant("exp", "a", frozenset({"contributor"}))


def test_permissions_binding_and_premature_admission():
    ledger = prepared()
    candidate = ledger.commit_candidate("exp", 0, "node-1", "parent", "candidate-1")
    assert ledger.commit_candidate("exp", 0, "node-1", "parent", "candidate-1") == candidate
    with pytest.raises(RuntimeError, match="selected contributor"):
        ledger.commit_candidate("exp", 0, "node-3", "parent", "x")
    with pytest.raises(RuntimeError, match="exact committed"):
        ledger.record_validator_decision("exp", 0, "node-3", "node-1", "wrong", True)
    with pytest.raises(RuntimeError, match="selected validator"):
        ledger.record_validator_decision("exp", 0, "node-2", "node-1", "candidate-1", True)
    ledger.record_validator_decision("exp", 0, "node-0", "node-1", "candidate-1", True)
    with pytest.raises(RuntimeError, match="required validator"):
        ledger.finalize_admission("exp", 0, {"node-1": True})


def complete() -> InMemoryLedger:
    ledger = prepared()
    for contributor in assignment().selected_contributors:
        ledger.commit_candidate("exp", 0, contributor, "parent", f"hash-{contributor}")
    for validator in assignment().selected_validators:
        for contributor in assignment().selected_contributors:
            ledger.record_validator_decision("exp", 0, validator, contributor, f"hash-{contributor}", contributor != "node-2")
    ledger.finalize_admission("exp", 0, {"node-0": True, "node-1": True, "node-2": False})
    ledger.commit_aggregate("exp", 0, {"node-0": "hash-node-0", "node-1": "hash-node-1"}, "aggregate")
    for node in assignment().network_participants:
        ledger.confirm_model_installation("exp", 0, node, "aggregate")
    ledger.finalize_round("exp", 0)
    assert ledger.verify_round("exp", 0)
    return ledger


def test_five_node_workflow_is_reproducible_and_finalized_is_immutable():
    left, right = complete(), complete()
    assert left.final_event_chain_hash == right.final_event_chain_hash
    assert left.validation_artifact("exp")["ledger_round_consensus"] is True
    assert left.finalize_round("exp", 0) == left.finalize_round("exp", 0)
    with pytest.raises(RuntimeError, match="finalized"):
        left.commit_candidate("exp", 0, "node-0", "parent", "new")


def test_aggregate_and_installation_fail_closed():
    ledger = prepared()
    ledger.commit_candidate("exp", 0, "node-0", "parent", "h")
    for validator in assignment().selected_validators:
        ledger.record_validator_decision("exp", 0, validator, "node-0", "h", False)
    ledger.finalize_admission("exp", 0, {"node-0": False})
    with pytest.raises(RuntimeError, match="exactly match"):
        ledger.commit_aggregate("exp", 0, {"node-0": "h"}, "aggregate")
    ledger.commit_aggregate("exp", 0, {}, "aggregate")
    with pytest.raises(RuntimeError, match="canonical aggregate"):
        ledger.confirm_model_installation("exp", 0, "node-0", "wrong")
    with pytest.raises(RuntimeError, match="missing"):
        ledger.finalize_round("exp", 0)


def test_tampering_and_broken_linkage_are_detected():
    ledger = complete()
    object.__setattr__(ledger._events[2], "payload", {"tampered": True})
    with pytest.raises(RuntimeError, match="tampering"):
        ledger.verify_round("exp", 0)
    ledger = complete()
    object.__setattr__(ledger._events[2], "previous_event_hash", "broken")
    with pytest.raises(RuntimeError, match="linkage"):
        ledger.verify_round("exp", 0)


def test_role_commitment_and_admission_are_immutable():
    ledger = prepared()
    with pytest.raises(RuntimeError, match="immutable"):
        ledger.commit_round_roles(replace(assignment(), selected_contributors=("node-0",)))


def test_disabled_artifact_has_no_fabricated_chain_data():
    assert disabled_ledger_artifact() == {
        "enabled": False,
        "backend": None,
        "ledger_identifier": None,
        "event_receipts": [],
        "final_event_chain_hash": None,
    }
    assert create_ledger(enabled=False, backend="anything") is None
    with pytest.raises(ValueError, match="unsupported"):
        create_ledger(enabled=True, backend="ethereum")
