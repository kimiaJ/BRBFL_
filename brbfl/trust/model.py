# ruff: noqa: D101, D102, D105
"""Immutable values used by deterministic validator trust accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ValidatorTrustState:
    validator_id: str
    alpha: float
    beta: float
    processed_vote_count: int = 0
    agreement_count: int = 0
    disagreement_count: int = 0
    last_finalized_round: int | None = None

    @property
    def score(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def artifact(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "score": self.score,
            "agreement_count": self.agreement_count,
            "disagreement_count": self.disagreement_count,
            "processed_vote_count": self.processed_vote_count,
            "last_finalized_round": self.last_finalized_round,
        }


@dataclass(frozen=True, order=True)
class TrustUpdateEvidence:
    round_id: int
    validator_id: str
    candidate_id: str
    reported_decision: bool
    reference_decision: bool
    agreed: bool
    evidence_sha256: str

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.round_id, self.validator_id, self.candidate_id)

    def artifact(self) -> dict[str, object]:
        return {
            "round": self.round_id,
            "validator_id": self.validator_id,
            "candidate_id": self.candidate_id,
            "reported_decision": self.reported_decision,
            "reference_decision": self.reference_decision,
            "agreed": self.agreed,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class RoundTrustSnapshot:
    experiment_id: str
    round_id: int
    pre_round: Mapping[str, ValidatorTrustState]
    updates: tuple[TrustUpdateEvidence, ...]
    post_round: Mapping[str, ValidatorTrustState]
    snapshot_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pre_round", MappingProxyType(dict(self.pre_round)))
        object.__setattr__(self, "post_round", MappingProxyType(dict(self.post_round)))
