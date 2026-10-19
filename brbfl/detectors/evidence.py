"""Canonical evidence shared by every specialized Byzantine detector."""

from __future__ import annotations

from dataclasses import dataclass, field

from brbfl.ca.model import EvidenceCategory
from brbfl.canonical import canonical_hash


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DetectorEvidence:
    """One immutable, round-scoped result produced by a specialized detector."""

    experiment_id: str
    round_number: int
    detector_id: str
    detector_capability: str
    detector_assignment_hash: str
    evaluated_participant_id: str
    evidence_category: EvidenceCategory
    measurement_hash: str
    verification_result: bool
    source_artifact_hash: str
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate consensus fields and freeze the canonical evidence identity."""
        text_fields = {
            "experiment_id": self.experiment_id,
            "detector_id": self.detector_id,
            "detector_capability": self.detector_capability,
            "evaluated_participant_id": self.evaluated_participant_id,
        }
        if any(not isinstance(value, str) or not value for value in text_fields.values()):
            raise ValueError("detector evidence identifiers must be non-empty strings")
        if isinstance(self.round_number, bool) or not isinstance(self.round_number, int) or self.round_number < 0:
            raise ValueError("round_number must be a non-negative integer")
        if not isinstance(self.evidence_category, EvidenceCategory):
            raise TypeError("evidence_category must be an EvidenceCategory")
        if type(self.verification_result) is not bool:
            raise TypeError("verification_result must be a canonical boolean")
        for name in ("detector_assignment_hash", "measurement_hash", "source_artifact_hash"):
            _require_sha256(name, getattr(self, name))
        object.__setattr__(self, "evidence_hash", canonical_hash("DetectorEvidence/v1", self.canonical_payload()))

    def canonical_payload(self) -> dict[str, object]:
        """Return the complete consensus payload used to derive ``evidence_hash``."""
        return {
            "experiment_id": self.experiment_id,
            "round": self.round_number,
            "detector_id": self.detector_id,
            "detector_capability": self.detector_capability,
            "detector_assignment_hash": self.detector_assignment_hash,
            "evaluated_participant_id": self.evaluated_participant_id,
            "evidence_category": self.evidence_category.value,
            "measurement_hash": self.measurement_hash,
            "verification_result": self.verification_result,
            "source_artifact_hash": self.source_artifact_hash,
        }

    def artifact(self) -> dict[str, object]:
        """Return a JSON-safe ledger representation."""
        return {**self.canonical_payload(), "evidence_hash": self.evidence_hash}

    def verify_hash(self) -> bool:
        """Verify that the record still matches its canonical identity."""
        return self.evidence_hash == canonical_hash("DetectorEvidence/v1", self.canonical_payload())
