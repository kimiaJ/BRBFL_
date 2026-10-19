"""Security and determinism tests for specialized committee finalization."""

from dataclasses import replace

import pytest

from brbfl.ca import EvidenceCategory
from brbfl.detectors.committee import finalize_detector_committee
from brbfl.detectors.evidence import DetectorEvidence
from brbfl.selection.roles import RoundRoleAssignment


def fixture():
    """Build a three-member committee and matching severe evidence."""
    assignment = RoundRoleAssignment(
        "experiment",
        2,
        ("a", "b", "c", "target"),
        ("target",),
        (),
        detector_subgroups={"model_update_detector": ("a", "b", "c")},
    )
    records = tuple(
        DetectorEvidence(
            "experiment",
            2,
            node,
            "model_update_detector",
            assignment.assignment_hash,
            "target",
            EvidenceCategory.SEVERE,
            "a" * 64,
            True,
            "b" * 64,
        )
        for node in ("a", "b", "c")
    )
    return assignment, records


def finalize(assignment, records):
    """Finalize evidence against the fixed target update."""
    return finalize_detector_committee(assignment, "model_update_detector", "target", "b" * 64, records)


def test_majority_is_order_independent_and_minority_cannot_override():
    """The same evidence set produces the same decision in every order."""
    assignment, records = fixture()
    records = (*records[:2], replace(records[2], evidence_category=EvidenceCategory.POSITIVE))
    decision = finalize(assignment, records)
    assert decision.category == EvidenceCategory.SEVERE
    assert decision.artifact() == finalize(assignment, tuple(reversed(records))).artifact()


@pytest.mark.parametrize(
    "changes",
    [
        {"round_number": 1},
        {"experiment_id": "other"},
        {"detector_id": "outsider"},
        {"detector_assignment_hash": "c" * 64},
        {"source_artifact_hash": "c" * 64},
        {"evaluated_participant_id": "a"},
        {"detector_capability": "other"},
    ],
)
def test_foreign_or_stale_evidence_is_rejected(changes):
    """Evidence cannot cross round, update, participant, or assignment boundaries."""
    assignment, records = fixture()
    with pytest.raises(ValueError, match="context"):
        finalize(assignment, (*records[:2], replace(records[2], **changes)))


def test_duplicate_votes_and_unverified_majorities_are_rejected():
    """One identity cannot manufacture quorum and invalid votes abstain."""
    assignment, records = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        finalize(assignment, (records[0], records[0]))
    with pytest.raises(RuntimeError, match="quorum"):
        finalize(assignment, (records[0], replace(records[1], verification_result=False)))
    with pytest.raises(RuntimeError, match="quorum"):
        finalize(assignment, (records[0], replace(records[1], evidence_category=EvidenceCategory.POSITIVE)))
