"""Specialized model-update detector for direction-reversing attacks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from brbfl.ca import EvidenceCategory
from brbfl.canonical import canonical_hash
from brbfl.detectors.evidence import DetectorEvidence
from brbfl.experiments.sign_flipping_evidence import canonical_parameter_hash


@dataclass(frozen=True, slots=True)
class SignFlippingDetectionPolicy:
    """Deterministic cosine thresholds for model-update classification."""

    severe_cosine_threshold: float = -0.9
    positive_cosine_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Reject non-finite or overlapping classification thresholds."""
        values = (self.severe_cosine_threshold, self.positive_cosine_threshold)
        if any(isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) for value in values):
            raise ValueError("sign-flipping thresholds must be finite numbers")
        if not -1 <= self.severe_cosine_threshold < self.positive_cosine_threshold <= 1:
            raise ValueError("sign-flipping thresholds must satisfy -1 <= severe < positive <= 1")

    @property
    def policy_hash(self) -> str:
        """Return the canonical policy identity."""
        return canonical_hash(
            "SignFlippingDetectionPolicy/v1",
            {
                "severe_cosine_threshold": self.severe_cosine_threshold,
                "positive_cosine_threshold": self.positive_cosine_threshold,
            },
        )


class SignFlippingDetector:
    """Compare one submitted update with a committee-derived reference update."""

    capability = "model_update_detector"

    def __init__(self, detector_id: str, policy: SignFlippingDetectionPolicy | None = None) -> None:
        """Create one identified detector with an immutable classification policy."""
        if not detector_id:
            raise ValueError("detector_id must be non-empty")
        self.detector_id = detector_id
        self.policy = policy or SignFlippingDetectionPolicy()

    def evaluate(
        self,
        *,
        experiment_id: str,
        round_number: int,
        detector_assignment_hash: str,
        evaluated_participant_id: str,
        candidate_update: Mapping[str, Any] | Sequence[Any],
        reference_update: Mapping[str, Any] | Sequence[Any],
        parameter_names: Sequence[str] | None = None,
    ) -> DetectorEvidence:
        """Classify update direction and emit ledger-ready detector evidence."""
        candidate = _flatten(candidate_update, parameter_names)
        reference = _flatten(reference_update, parameter_names)
        candidate_hash = canonical_parameter_hash(candidate_update, parameter_names)
        reference_hash = canonical_parameter_hash(reference_update, parameter_names)
        verification_result = candidate.shape == reference.shape and candidate.size > 0
        candidate_norm = float(np.linalg.norm(candidate))
        reference_norm = float(np.linalg.norm(reference))
        verification_result = verification_result and candidate_norm > 0 and reference_norm > 0
        cosine_similarity = (
            float(np.dot(candidate, reference) / (candidate_norm * reference_norm)) if verification_result else None
        )
        if cosine_similarity is None:
            category = EvidenceCategory.NEUTRAL
        elif cosine_similarity <= self.policy.severe_cosine_threshold:
            category = EvidenceCategory.SEVERE
        elif cosine_similarity >= self.policy.positive_cosine_threshold:
            category = EvidenceCategory.POSITIVE
        else:
            category = EvidenceCategory.NEGATIVE
        measurement = {
            "candidate_update_hash": candidate_hash,
            "reference_update_hash": reference_hash,
            "candidate_l2_norm": candidate_norm,
            "reference_l2_norm": reference_norm,
            "cosine_similarity": cosine_similarity,
            "policy_hash": self.policy.policy_hash,
        }
        return DetectorEvidence(
            experiment_id=experiment_id,
            round_number=round_number,
            detector_id=self.detector_id,
            detector_capability=self.capability,
            detector_assignment_hash=detector_assignment_hash,
            evaluated_participant_id=evaluated_participant_id,
            evidence_category=category,
            measurement_hash=canonical_hash("SignFlippingMeasurement/v1", measurement),
            verification_result=verification_result,
            source_artifact_hash=candidate_hash,
        )


def _flatten(values: Mapping[str, Any] | Sequence[Any], names: Sequence[str] | None) -> np.ndarray:
    """Flatten parameters in the same canonical name order used by the content hash."""
    if isinstance(values, Mapping):
        ordered = [value for _, value in sorted(values.items())]
    else:
        parameter_names = list(names) if names is not None else [str(index) for index in range(len(values))]
        if len(parameter_names) != len(values):
            raise ValueError("parameter names and values must have equal lengths")
        ordered = [value for _, value in sorted(zip(parameter_names, values, strict=True))]
    arrays = [np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64).ravel() for value in ordered]
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
