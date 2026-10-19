"""Deterministic finalization of round-scoped specialized detector evidence."""

from collections import Counter
from dataclasses import dataclass

from brbfl.ca import EvidenceCategory
from brbfl.canonical import canonical_hash
from brbfl.detectors.evidence import DetectorEvidence
from brbfl.selection.roles import RoundRoleAssignment


@dataclass(frozen=True)
class DetectorCommitteeDecision:
    """A finalized category bound to one assignment and submitted update."""

    assignment_hash: str
    capability: str
    participant_id: str
    source_artifact_hash: str
    category: EvidenceCategory
    quorum: int
    evidence_hashes: tuple[str, ...]

    def artifact(self) -> dict[str, object]:
        """Return a canonical, JSON-safe decision with its content hash."""
        payload = {
            "assignment_hash": self.assignment_hash,
            "capability": self.capability,
            "participant_id": self.participant_id,
            "source_artifact_hash": self.source_artifact_hash,
            "category": self.category.value,
            "quorum": self.quorum,
            "evidence_hashes": list(self.evidence_hashes),
        }
        return {**payload, "decision_hash": canonical_hash("DetectorCommitteeDecision/v1", payload)}


def finalize_detector_committee(
    assignment: RoundRoleAssignment,
    capability: str,
    participant_id: str,
    source_artifact_hash: str,
    evidence: tuple[DetectorEvidence, ...],
) -> DetectorCommitteeDecision:
    """
    Require a strict committee majority for one verified evidence category.

    Missing or unverified measurements never count as positive or negative votes.
    A split or incomplete quorum raises rather than fabricating neutral evidence.
    """
    if not assignment.verify_hash():
        raise ValueError("invalid assignment hash")
    members = assignment.detector_subgroups.get(capability, ())
    if not members:
        raise ValueError("specialized committee is absent")
    if participant_id not in assignment.selected_contributors:
        raise ValueError("evaluated participant is not a selected contributor")
    if len(source_artifact_hash) != 64 or any(c not in "0123456789abcdef" for c in source_artifact_hash):
        raise ValueError("invalid source artifact hash")
    seen: set[str] = set()
    votes: Counter[EvidenceCategory] = Counter()
    hashes = []
    for record in evidence:
        if record.detector_id in seen:
            raise ValueError("duplicate detector vote")
        seen.add(record.detector_id)
        if (
            record.detector_id not in members
            or record.experiment_id != assignment.experiment_id
            or record.round_number != assignment.round_number
            or record.detector_assignment_hash != assignment.assignment_hash
            or record.detector_capability != capability
            or record.evaluated_participant_id != participant_id
            or record.source_artifact_hash != source_artifact_hash
            or not record.verify_hash()
        ):
            raise ValueError("evidence does not match the committee context")
        hashes.append(record.evidence_hash)
        if record.verification_result:
            votes[record.evidence_category] += 1
    quorum = len(members) // 2 + 1
    winners = [category for category, count in votes.items() if count >= quorum]
    if len(winners) != 1:
        raise RuntimeError("specialized committee has no verified category quorum")
    return DetectorCommitteeDecision(
        assignment.assignment_hash,
        capability,
        participant_id,
        source_artifact_hash,
        winners[0],
        quorum,
        tuple(sorted(hashes)),
    )
