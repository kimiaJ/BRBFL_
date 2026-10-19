"""Tests for the detector-independent evidence boundary."""

from dataclasses import FrozenInstanceError
from dataclasses import replace as dataclass_replace

import pytest

from brbfl.ca import EvidenceCategory
from brbfl.detectors import DetectorEvidence

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def evidence(**changes):
    """Build a valid model-update detector record with optional changes."""
    values = {
        "experiment_id": "mnist:666:detectors",
        "round_number": 2,
        "detector_id": "node-3",
        "detector_capability": "model_update_detector",
        "detector_assignment_hash": SHA_A,
        "evaluated_participant_id": "node-1",
        "evidence_category": EvidenceCategory.SEVERE,
        "measurement_hash": SHA_B,
        "verification_result": True,
        "source_artifact_hash": SHA_C,
    }
    values.update(changes)
    return DetectorEvidence(**values)


def test_evidence_is_immutable_canonical_and_json_safe():
    """Equivalent records share an immutable JSON-safe canonical identity."""
    first = evidence()
    second = evidence()

    assert first == second
    assert first.verify_hash()
    assert first.artifact()["round"] == 2
    assert first.artifact()["evidence_category"] == "severe"
    assert first.artifact()["evidence_hash"] == first.evidence_hash
    with pytest.raises(FrozenInstanceError):
        first.detector_id = "changed"


def test_consensus_fields_change_evidence_identity():
    """Every decision-bearing field participates in the evidence hash."""
    original = evidence()

    assert dataclass_replace(original, detector_id="node-4").evidence_hash != original.evidence_hash
    assert dataclass_replace(original, evidence_category=EvidenceCategory.NEGATIVE).evidence_hash != original.evidence_hash
    assert dataclass_replace(original, verification_result=False).evidence_hash != original.evidence_hash


@pytest.mark.parametrize("field", ["detector_assignment_hash", "measurement_hash", "source_artifact_hash"])
def test_hash_fields_require_lowercase_sha256(field):
    """Reject ambiguous or malformed hash references."""
    with pytest.raises(ValueError, match=field):
        evidence(**{field: "not-a-digest"})


def test_round_category_and_verification_are_strictly_typed():
    """Reject Python values which serialize ambiguously at consensus boundaries."""
    with pytest.raises(ValueError, match="round_number"):
        evidence(round_number=True)
    with pytest.raises(TypeError, match="EvidenceCategory"):
        evidence(evidence_category="severe")
    with pytest.raises(TypeError, match="canonical boolean"):
        evidence(verification_result=1)
