# ruff: noqa: D103
"""Trust integration at the authoritative finalized memory-ledger boundary."""

from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig


def test_finalized_canonical_votes_update_trust_without_changing_ledger():
    runtime = RuntimeLedgerAdapter(
        RuntimeLedgerConfig(enabled=True, trust_enabled=True), "trust-fixture",
        ("c", "honest", "bad"), ("c",), ("honest", "bad"),
    )
    runtime.record_candidate(0, "c", "parent", "candidate", [
        {"validator_node_id": "honest", "reported_decision": True, "reference_decision": True},
        {"validator_node_id": "bad", "reported_decision": False, "reference_decision": True},
    ])
    runtime.finalize_admission(0, {"c": True})
    runtime.confirm_installation(0, "c", "aggregate")
    runtime.confirm_installation(0, "honest", "aggregate")
    runtime.confirm_installation(0, "bad", "aggregate")
    runtime.finalize_round(0)
    artifact = runtime.trust_artifact()
    assert artifact["final_states"]["honest"]["score"] == 2 / 3
    assert artifact["final_states"]["bad"]["score"] == 1 / 3
    assert runtime.validation_artifact()["ledger_round_consensus"] is True
    runtime.finalize_round(0)  # repeated barrier observation does not double count
    assert runtime.trust_artifact() == artifact


def test_disabled_runtime_has_no_trust_artifact():
    runtime = RuntimeLedgerAdapter(RuntimeLedgerConfig(), "disabled", ("c",), ("c",), ("c",))
    assert runtime.trust_artifact() is None
