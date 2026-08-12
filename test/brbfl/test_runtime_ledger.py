"""Focused integration tests for workflow-to-memory-ledger adaptation."""

# ruff: noqa: D103

import json
import threading

import pytest

from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig

PARTICIPANTS = tuple(f"node-{index}" for index in range(5))
CONTRIBUTORS = ("node-0", "node-1", "node-2")
VALIDATORS = ("node-0", "node-3", "node-4")


def adapter(*, enabled: bool = True, fail_closed: bool = True) -> RuntimeLedgerAdapter:
    return RuntimeLedgerAdapter(
        RuntimeLedgerConfig(enabled=enabled, fail_closed=fail_closed),
        "five-node-clean",
        PARTICIPANTS,
        CONTRIBUTORS,
        VALIDATORS,
    )


def votes(candidate: str, decision: bool = True) -> list[dict[str, object]]:
    return [
        {
            "validator_node_id": validator,
            "candidate_node_id": candidate,
            "reported_decision": decision,
            "vote_sha256": f"vote-{validator}-{candidate}",
            "reference_decision": decision,
            "byzantine": False,
            "strategy": "honest_reference",
            "attack_group_id": None,
            "order_index": 1,
        }
        for validator in VALIDATORS
    ]


def admit_round(runtime: RuntimeLedgerAdapter, round_number: int = 0, parent: str = "parent-0") -> None:
    for contributor in CONTRIBUTORS:
        runtime.record_candidate(round_number, contributor, parent, f"candidate-{round_number}-{contributor}", votes(contributor))
    runtime.finalize_admission(round_number, dict.fromkeys(CONTRIBUTORS, True))


def complete_round(runtime: RuntimeLedgerAdapter, round_number: int = 0, parent: str = "parent-0") -> None:
    admit_round(runtime, round_number, parent)
    for participant in PARTICIPANTS:
        runtime.confirm_installation(round_number, participant, f"aggregate-{round_number}")
    runtime.finalize_round(round_number)


def test_disabled_path_preserves_behavior_and_emits_disabled_artifact():
    runtime = adapter(enabled=False)
    runtime.record_candidate(0, "not-selected", "anything", "anything", [])
    runtime.finalize_admission(0, {})
    runtime.confirm_installation(0, "unknown", "anything")
    runtime.finalize_round(0)
    assert runtime.validation_artifact() == {
        "enabled": False,
        "backend": None,
        "ledger_identifier": None,
        "event_receipts": [],
        "final_event_chain_hash": None,
        "fail_closed": True,
    }


def test_clean_five_node_trace_is_complete_causal_and_json_serializable():
    runtime = adapter()
    complete_round(runtime)
    artifact = runtime.validation_artifact()
    event_types = [event["event_type"] for event in artifact["ordered_events"]]
    assert event_types[:6] == ["ExperimentStarted", *("ParticipantRegistered",) * 5]
    assert event_types[6:] == [
        "RoundRolesCommitted",
        "RoundOpened",
        *("CandidateCommitted", "ValidatorDecisionCommitted", "ValidatorDecisionCommitted", "ValidatorDecisionCommitted") * 3,
        "AdmissionFinalized",
        "AggregateCommitted",
        *("ModelInstallationConfirmed",) * 5,
        "RoundFinalized",
    ]
    assert artifact["round_verification"] == {"0": True}
    assert artifact["verification_reason"] == {"0": "verified"}
    assert artifact["ledger_round_consensus"] is True
    json.dumps(artifact, sort_keys=True)


def test_identical_callbacks_are_idempotent_and_conflicts_fail_closed():
    runtime = adapter()
    runtime.record_candidate(0, "node-0", "parent", "candidate", votes("node-0"))
    count = len(runtime.ledger.events)
    runtime.record_candidate(0, "node-0", "parent", "candidate", votes("node-0"))
    assert len(runtime.ledger.events) == count
    with pytest.raises(RuntimeError, match="conflicting runtime round parent"):
        runtime.record_candidate(0, "node-1", "stale", "candidate-1", votes("node-1"))


def test_missing_votes_and_early_aggregate_fail_closed():
    runtime = adapter()
    runtime.open_round(0, "parent")
    runtime.ledger.commit_candidate("five-node-clean", 0, "node-0", "parent", "candidate")
    with pytest.raises(RuntimeError, match="required validator decisions"):
        runtime.ledger.finalize_admission("five-node-clean", 0, {"node-0": True})
    with pytest.raises(RuntimeError, match="before admission"):
        runtime.confirm_installation(0, "node-0", "aggregate")


def test_installation_mismatch_missing_confirmation_and_round_isolation():
    runtime = adapter()
    admit_round(runtime)
    runtime.confirm_installation(0, "node-0", "aggregate-0")
    with pytest.raises(RuntimeError, match="conflicting runtime aggregate"):
        runtime.confirm_installation(0, "node-1", "wrong")
    with pytest.raises(RuntimeError, match="confirmations are missing"):
        runtime.finalize_round(0)
    for participant in PARTICIPANTS[1:]:
        runtime.confirm_installation(0, participant, "aggregate-0")
    runtime.finalize_round(0)
    runtime.record_candidate(1, "node-0", "aggregate-0", "candidate-1", votes("node-0"))
    with pytest.raises(RuntimeError, match="every and only committed"):
        runtime.finalize_admission(1, dict.fromkeys(CONTRIBUTORS, True))


def test_deterministic_equivalent_executions_and_ledger_does_not_choose_admission():
    left = adapter()
    right = adapter()
    complete_round(left)
    complete_round(right)
    assert left.validation_artifact() == right.validation_artifact()

    rejected = adapter()
    for contributor in CONTRIBUTORS:
        rejected.record_candidate(0, contributor, "parent-0", f"candidate-0-{contributor}", votes(contributor, False))
    workflow_decisions = {"node-0": False, "node-1": True, "node-2": False}
    rejected.finalize_admission(0, workflow_decisions)
    assert rejected.validation_artifact()["rounds"]["0"]["admission"] == workflow_decisions


def test_participant_and_role_commitments_are_deterministic():
    left = adapter()
    right = adapter()
    left.open_round(0, "parent")
    right.open_round(0, "parent")
    assert left.validation_artifact()["participant_registrations"] == right.validation_artifact()["participant_registrations"]
    assert left.validation_artifact()["per_round_role_assignment_hash"] == right.validation_artifact()[
        "per_round_role_assignment_hash"
    ]


def test_five_node_callback_topology_has_one_authoritative_publisher_per_candidate():
    runtime = adapter()
    barrier = threading.Barrier(len(PARTICIPANTS))
    errors: list[BaseException] = []

    def callback(node_id: str) -> None:
        try:
            barrier.wait()
            for offset, contributor in enumerate(reversed(CONTRIBUTORS)):
                observed_votes = votes(contributor)
                for vote in observed_votes:
                    # Reproduce node-local gate ordering differences.  These
                    # rows are valid observations, not authoritative publishes.
                    vote["order_index"] = offset + int(node_id[-1]) + 1
                    vote["vote_sha256"] = f"{node_id}-{offset}-{vote['validator_node_id']}"
                runtime.record_candidate(
                    0,
                    contributor,
                    "parent",
                    f"candidate-{contributor}",
                    observed_votes,
                    publisher_id=node_id,
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=callback, args=(node,)) for node in PARTICIPANTS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    decisions = [event for event in runtime.ledger.events if event.event_type.value == "ValidatorDecisionCommitted"]
    assert len(decisions) == len(CONTRIBUTORS) * len(VALIDATORS)
    assert {event.payload["contributor_id"] for event in decisions} == set(CONTRIBUTORS)
    runtime.finalize_admission(0, dict.fromkeys(CONTRIBUTORS, True))


def test_authoritative_repeats_are_idempotent_but_meaningful_conflicts_fail_closed():
    runtime = adapter()
    original = votes("node-0")
    runtime.record_candidate(0, "node-0", "parent", "candidate", original, publisher_id="node-0")
    event_count = len(runtime.ledger.events)
    runtime.record_candidate(0, "node-0", "parent", "candidate", original, publisher_id="node-0")
    assert len(runtime.ledger.events) == event_count

    changed = votes("node-0")
    changed[0]["order_index"] = 99
    changed[0]["vote_sha256"] = "changed"
    with pytest.raises(RuntimeError, match=r"first_differing_field=order_index"):
        runtime.record_candidate(0, "node-0", "parent", "candidate", changed, publisher_id="node-0")

    with pytest.raises(RuntimeError, match="conflicting candidate commitment"):
        runtime.record_candidate(0, "node-0", "parent", "different-candidate", original, publisher_id="node-0")
    with pytest.raises(RuntimeError, match="conflicting runtime round parent"):
        runtime.record_candidate(0, "node-0", "different-parent", "candidate", original, publisher_id="node-0")


def test_decision_conflict_prevents_admission_aggregation_and_installation():
    runtime = adapter()
    runtime.record_candidate(0, "node-0", "parent", "candidate", votes("node-0"))
    with pytest.raises(RuntimeError, match="conflicting validator decision"):
        runtime.ledger.record_validator_decision(
            "five-node-clean", 0, "node-0", "node-0", "candidate", False, {"strategy": "contradiction"}
        )
    with pytest.raises(RuntimeError, match="every and only committed"):
        runtime.finalize_admission(0, dict.fromkeys(CONTRIBUTORS, True))
    with pytest.raises(RuntimeError, match="before admission"):
        runtime.confirm_installation(0, "node-0", "aggregate")
