"""Commit real detector measurements through runtime into hash-linked ledger events."""

from dataclasses import replace

import numpy as np
import pytest

from brbfl.detectors import SignFlippingDetector
from brbfl.experiments.sign_flipping_evidence import canonical_parameter_hash
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig

CAPABILITY = "model_update_detector"


def fixture():
    """Open a runtime round with a selected committee and committed reversed update."""
    runtime = RuntimeLedgerAdapter(
        RuntimeLedgerConfig(
            enabled=True,
            trust_enabled=True,
            ca_enabled=True,
            selection_strategy="ca_state",
            validator_eligible_participants=("a", "b", "c"),
            detector_target_counts={CAPABILITY: 2},
            detector_capabilities={node: (CAPABILITY,) for node in ("b", "c")},
        ),
        "exp",
        ("a", "b", "c"),
        ("a",),
        ("b", "c"),
    )
    runtime.open_round(0, "parent")
    candidate = {"weight": np.array([-1.0, -2.0])}
    reference = {"weight": np.array([1.0, 2.0])}
    source_hash = canonical_parameter_hash(candidate)
    runtime.record_candidate(0, "a", "parent", source_hash, [])
    assignment = runtime.role_assignment(0)
    evidence = tuple(
        SignFlippingDetector(node).evaluate(
            experiment_id="exp",
            round_number=0,
            detector_assignment_hash=assignment.assignment_hash,
            evaluated_participant_id="a",
            candidate_update=candidate,
            reference_update=reference,
        )
        for node in assignment.detector_subgroups[CAPABILITY]
    )
    return runtime, source_hash, evidence


def test_measured_committee_result_is_hash_linked_exported_and_idempotent():
    """Numerical detector output reaches the ledger with complete evidence provenance."""
    runtime, source, evidence = fixture()
    runtime.record_detector_committee(0, CAPABILITY, "a", source, evidence)
    count = len(runtime.ledger.events)
    runtime.record_detector_committee(0, CAPABILITY, "a", source, tuple(reversed(evidence)))
    assert len(runtime.ledger.events) == count
    row = runtime.ledger.get_round_record("exp", 0)["detector_committees"][0]
    assert row["decision"]["category"] == "severe"
    assert row["decision"]["source_artifact_hash"] == source
    assert len(row["evidence"]) == 2
    assert runtime.ledger.verify_round("exp", 0)
    exported = runtime.validation_artifact()["rounds"]["0"]["detector_committees"]
    assert exported == [row]
    row["decision"]["category"] = "positive"
    assert runtime.ledger.get_round_record("exp", 0)["detector_committees"][0]["decision"]["category"] == "severe"


def test_invalid_committee_cannot_append_an_event():
    """Wrong-update evidence and missing quorum leave ledger state unchanged."""
    runtime, source, evidence = fixture()
    count = len(runtime.ledger.events)
    with pytest.raises(ValueError, match="candidate"):
        runtime.record_detector_committee(0, CAPABILITY, "a", "c" * 64, evidence)
    with pytest.raises(RuntimeError, match="quorum"):
        runtime.record_detector_committee(0, CAPABILITY, "a", source, evidence[:1])
    with pytest.raises(ValueError, match="context"):
        runtime.record_detector_committee(0, CAPABILITY, "a", source, (replace(evidence[0], round_number=1), evidence[1]))
    assert len(runtime.ledger.events) == count
