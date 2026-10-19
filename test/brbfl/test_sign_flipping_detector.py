"""Tests for detector-generated sign-flipping evidence."""

import numpy as np

from brbfl.ca import EvidenceCategory
from brbfl.detectors import SignFlippingDetector

ASSIGNMENT_HASH = "a" * 64


def evaluate(candidate, reference):
    """Evaluate one synthetic update with a fixed round assignment."""
    return SignFlippingDetector("node-3").evaluate(
        experiment_id="mnist:666:detectors",
        round_number=1,
        detector_assignment_hash=ASSIGNMENT_HASH,
        evaluated_participant_id="node-1",
        candidate_update=[np.asarray(candidate, dtype=np.float32)],
        reference_update=[np.asarray(reference, dtype=np.float32)],
        parameter_names=["weight"],
    )


def test_reversed_update_produces_verified_severe_evidence():
    """An exactly reversed direction is severe without consulting attack metadata."""
    result = evaluate([-2.0, 4.0], [1.0, -2.0])

    assert result.evidence_category is EvidenceCategory.SEVERE
    assert result.verification_result is True
    assert result.detector_capability == "model_update_detector"
    assert result.detector_assignment_hash == ASSIGNMENT_HASH
    assert result.verify_hash()


def test_aligned_update_produces_positive_evidence():
    """A directionally aligned update is classified as positive."""
    result = evaluate([2.0, -4.0], [1.0, -2.0])

    assert result.evidence_category is EvidenceCategory.POSITIVE
    assert result.verification_result is True


def test_orthogonal_or_zero_update_cannot_be_severe():
    """Ambiguous and unverifiable measurements do not fabricate severe evidence."""
    orthogonal = evaluate([0.0, 1.0], [1.0, 0.0])
    zero = evaluate([0.0, 0.0], [1.0, 0.0])

    assert orthogonal.evidence_category is EvidenceCategory.NEGATIVE
    assert orthogonal.verification_result is True
    assert zero.evidence_category is EvidenceCategory.NEUTRAL
    assert zero.verification_result is False


def test_measurement_and_source_hashes_are_deterministic_and_content_sensitive():
    """Equivalent inputs reproduce evidence while changed content changes its identity."""
    first = evaluate([-2.0, 4.0], [1.0, -2.0])
    second = evaluate([-2.0, 4.0], [1.0, -2.0])
    changed = evaluate([-3.0, 4.0], [1.0, -2.0])

    assert first == second
    assert changed.measurement_hash != first.measurement_hash
    assert changed.source_artifact_hash != first.source_artifact_hash
    assert changed.evidence_hash != first.evidence_hash
