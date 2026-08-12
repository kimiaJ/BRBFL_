# ruff: noqa: D102, D107
"""Experiment-scoped, round-finalized Beta reputation runtime."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from brbfl.trust.model import RoundTrustSnapshot, TrustUpdateEvidence, ValidatorTrustState


def _hash(domain: str, value: object) -> str:
    encoded = json.dumps({"domain": domain, "value": value}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class TrustRuntime:
    """Own trust state for exactly one experiment; selection never consults it."""

    def __init__(self, experiment_id: str, validators: Iterable[str], prior_alpha: float = 1, prior_beta: float = 1) -> None:
        validators = tuple(sorted(validators))
        if not experiment_id or not validators or any(not isinstance(node, str) or not node for node in validators):
            raise ValueError("experiment_id and unique non-empty validators are required")
        if len(set(validators)) != len(validators):
            raise ValueError("validators must be unique")
        if not all(math.isfinite(value) and value > 0 for value in (prior_alpha, prior_beta)):
            raise ValueError("trust priors must be finite and greater than zero")
        self.experiment_id = experiment_id
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self._states = {node: ValidatorTrustState(node, self.prior_alpha, self.prior_beta) for node in validators}
        self._snapshots: dict[int, RoundTrustSnapshot] = {}

    @property
    def states(self) -> Mapping[str, ValidatorTrustState]:
        return MappingProxyType(dict(self._states))

    @property
    def snapshots(self) -> Mapping[int, RoundTrustSnapshot]:
        return MappingProxyType(dict(self._snapshots))

    def finalize_round(
        self,
        round_id: int,
        eligible_validators: Iterable[str],
        candidates: Iterable[str],
        decisions: Iterable[Mapping[str, Any]],
    ) -> RoundTrustSnapshot:
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be a non-negative canonical integer")
        if round_id in self._snapshots:
            raise RuntimeError(f"trust round already finalized: {round_id}")
        if round_id != len(self._snapshots):
            raise RuntimeError(f"trust rounds must finalize consecutively: expected={len(self._snapshots)}, actual={round_id}")
        validators = tuple(sorted(eligible_validators))
        candidate_ids = tuple(sorted(candidates))
        if not validators or not set(validators) <= set(self._states) or not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise RuntimeError("authoritative validator/candidate set is invalid")
        updates: dict[tuple[int, str, str], TrustUpdateEvidence] = {}
        for raw in decisions:
            validator = raw.get("validator_id")
            candidate = raw.get("candidate_id", raw.get("contributor_id"))
            reported = raw.get("reported_decision", raw.get("admitted"))
            reference = raw.get("reference_decision")
            if validator not in self._states or candidate not in candidate_ids:
                raise RuntimeError("trust evidence references an unknown validator or candidate")
            if type(reported) is not bool or type(reference) is not bool:
                raise ValueError("trust decisions must be canonical booleans")
            payload = {
                "round": round_id,
                "validator_id": validator,
                "candidate_id": candidate,
                "reported_decision": reported,
                "reference_decision": reference,
            }
            update = TrustUpdateEvidence(
                round_id, validator, candidate, reported, reference, reported == reference, _hash("TrustVote/v1", payload)
            )
            existing = updates.get(update.key)
            if existing is not None and existing != update:
                raise RuntimeError(f"conflicting trust evidence: key={update.key}")
            updates[update.key] = update
        expected = {(round_id, validator, candidate) for validator in validators for candidate in candidate_ids}
        if set(updates) != expected:
            raise RuntimeError(f"incomplete authoritative trust evidence: missing={sorted(expected - set(updates))}")
        pre = dict(self._states)
        post = dict(pre)
        ordered = tuple(updates[key] for key in sorted(updates))
        for update in ordered:
            state = post[update.validator_id]
            post[update.validator_id] = replace(
                state,
                alpha=state.alpha + int(update.agreed),
                beta=state.beta + int(not update.agreed),
                processed_vote_count=state.processed_vote_count + 1,
                agreement_count=state.agreement_count + int(update.agreed),
                disagreement_count=state.disagreement_count + int(not update.agreed),
                last_finalized_round=round_id,
            )
        payload = self._snapshot_payload(round_id, pre, ordered, post)
        snapshot = RoundTrustSnapshot(self.experiment_id, round_id, pre, ordered, post, _hash("RoundTrustSnapshot/v1", payload))
        self._states = post
        self._snapshots[round_id] = snapshot
        return snapshot

    def _snapshot_payload(self, round_id, pre, updates, post) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "round": round_id,
            "pre_round": {node: pre[node].artifact() for node in sorted(pre)},
            "updates": [update.artifact() for update in updates],
            "post_round": {node: post[node].artifact() for node in sorted(post)},
        }

    def artifact(self, observation_only: bool = True) -> dict[str, object]:
        rounds = {}
        for number, snapshot in sorted(self._snapshots.items()):
            rounds[str(number)] = {
                "finalized": True,
                "pre_round": {node: snapshot.pre_round[node].artifact() for node in sorted(snapshot.pre_round)},
                "updates": [update.artifact() for update in snapshot.updates],
                "post_round": {node: snapshot.post_round[node].artifact() for node in sorted(snapshot.post_round)},
                "snapshot_sha256": snapshot.snapshot_sha256,
            }
        return {
            "enabled": True,
            "method": "beta_reputation",
            "observation_only": observation_only,
            "prior": {"alpha": self.prior_alpha, "beta": self.prior_beta},
            "rounds": rounds,
            "final_states": {node: self._states[node].artifact() for node in sorted(self._states)},
            "verification_result": True,
            "verification_reason": "verified",
        }
