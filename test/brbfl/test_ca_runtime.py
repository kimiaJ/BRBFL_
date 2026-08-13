"""CA lifecycle integration at the finalized ledger/trust boundary."""

# ruff: noqa: D103

import pytest

from brbfl.ca import EvidenceCategory, FinalizedTrustEvidenceMapper, ParticipantState
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig

PARTICIPANTS = ("a", "b")


def runtime(experiment="ca-lifecycle"):
    return RuntimeLedgerAdapter(
        RuntimeLedgerConfig(enabled=True, trust_enabled=True, ca_enabled=True),
        experiment,
        PARTICIPANTS,
        PARTICIPANTS,
        PARTICIPANTS,
    )


def complete(value, round_number=0, *, disagree=()):
    parent = "parent" if round_number == 0 else f"aggregate-{round_number - 1}"
    for candidate in PARTICIPANTS:
        votes = [
            {
                "validator_node_id": validator,
                "reported_decision": validator not in disagree,
                "reference_decision": True,
            }
            for validator in PARTICIPANTS
        ]
        value.record_candidate(round_number, candidate, parent, f"candidate-{round_number}-{candidate}", votes)
    value.finalize_admission(round_number, dict.fromkeys(PARTICIPANTS, True))
    for participant in PARTICIPANTS:
        value.confirm_installation(round_number, participant, f"aggregate-{round_number}")
    value.finalize_round(round_number)


def test_access_and_next_round_selection_fail_closed_before_finalization():
    value = runtime()
    with pytest.raises(RuntimeError, match="unavailable"):
        value.ca_snapshot(1)
    with pytest.raises(RuntimeError, match="unavailable"):
        value.open_round(1, "aggregate-0")


def test_finalized_round_transitions_once_and_retains_provenance():
    value = runtime()
    complete(value)
    snapshot = value.ca_snapshot_for_round(1)
    provenance = value.ca_provenance(0)
    assert snapshot.source_round == 0
    assert provenance.resulting_ca_snapshot_hash == snapshot.snapshot_hash
    assert provenance.previous_ca_snapshot_hash == value.ca_snapshot(0).snapshot_hash
    assert provenance.source_trust_snapshot_hash == value._trust.snapshots[0].snapshot_sha256
    assert provenance.source_ledger_hash == value.ledger.events[-2].event_hash
    event_count = len(value.ledger.events)
    value.finalize_round(0)
    assert len(value.ledger.events) == event_count
    assert value.ca_snapshot(1) is snapshot
    value.open_round(1, "aggregate-0")
    assert value.role_assignment(1).previous_state_hash == snapshot.snapshot_hash


def test_authoritative_votes_not_attack_metadata_drive_category_and_nodes_are_deterministic():
    left, right = runtime("same"), runtime("same")
    complete(left, disagree=("b",))
    complete(right, disagree=("b",))
    assert left.ca_snapshot(1) == right.ca_snapshot(1)
    records = {record.participant_id: record for record in left.ca_snapshot(1).transition_records}
    assert records["a"].evidence_category is EvidenceCategory.POSITIVE
    assert records["b"].evidence_category is EvidenceCategory.SEVERE
    assert left.ca_snapshot(1).participant_states["b"].state is ParticipantState.SUSPICIOUS


def test_mapper_makes_unevaluated_participants_neutral_and_experiments_do_not_leak():
    value = runtime("first")
    complete(value)
    snapshot = value._trust.snapshots[0]
    assert FinalizedTrustEvidenceMapper().categories(("a", "b", "other"), snapshot)["other"] is EvidenceCategory.NEUTRAL
    other = runtime("second")
    assert other.ca_snapshot(0).experiment_id == "second"
    assert other.ca_snapshot(0).snapshot_hash != value.ca_snapshot(0).snapshot_hash
    with pytest.raises(RuntimeError, match="unavailable"):
        other.ca_snapshot(1)


def test_disabled_ca_preserves_artifacts_and_does_not_gate_rounds():
    value = RuntimeLedgerAdapter(
        RuntimeLedgerConfig(enabled=True, trust_enabled=True), "disabled-ca", PARTICIPANTS, PARTICIPANTS, PARTICIPANTS
    )
    assert value.ca_artifact() is None
    value.open_round(0, "parent")
