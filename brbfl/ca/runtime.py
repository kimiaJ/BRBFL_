"""Finalized-evidence integration for the pure CA transition engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from brbfl.ca.model import EvidenceCategory
from brbfl.trust.model import RoundTrustSnapshot


class CAEvidenceMapper(Protocol):
    """Extension point for converting verified evidence into CA categories."""

    def categories(
        self, participant_ids: tuple[str, ...], trust_snapshot: RoundTrustSnapshot
    ) -> Mapping[str, EvidenceCategory]:
        """Return one category for every participant."""
        ...


class FinalizedTrustEvidenceMapper:
    """
    Map canonical trust votes, never experiment attack metadata.

    A participant whose evaluated validator votes all agree with their reference
    decisions is positive.  Any verified disagreement is severe.  Participants
    with no evaluated votes are neutral.  ``NEGATIVE`` is deliberately reserved
    for future detectors which can distinguish ordinary from severe rejection.
    """

    def categories(
        self, participant_ids: tuple[str, ...], trust_snapshot: RoundTrustSnapshot
    ) -> Mapping[str, EvidenceCategory]:
        """Derive deterministic categories from the finalized trust updates."""
        updates = {participant: [] for participant in participant_ids}
        for update in trust_snapshot.updates:
            if update.validator_id in updates:
                updates[update.validator_id].append(update.agreed)
        return {
            participant: (
                EvidenceCategory.NEUTRAL
                if not values
                else EvidenceCategory.POSITIVE
                if all(values)
                else EvidenceCategory.SEVERE
            )
            for participant, values in sorted(updates.items())
        }


@dataclass(frozen=True, slots=True)
class CATransitionProvenance:
    """Immutable consensus inputs and output committed for one source round."""

    source_round: int
    source_ledger_hash: str
    source_trust_snapshot_hash: str
    topology_hash: str
    previous_ca_snapshot_hash: str
    transition_policy_hash: str
    resulting_ca_snapshot_hash: str

    def artifact(self) -> dict[str, object]:
        """Return the canonical ledger payload."""
        return {
            "source_round": self.source_round,
            "source_ledger_hash": self.source_ledger_hash,
            "source_trust_snapshot_hash": self.source_trust_snapshot_hash,
            "topology_hash": self.topology_hash,
            "previous_ca_snapshot_hash": self.previous_ca_snapshot_hash,
            "transition_policy_hash": self.transition_policy_hash,
            "resulting_ca_snapshot_hash": self.resulting_ca_snapshot_hash,
        }
