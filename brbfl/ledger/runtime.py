"""Runtime adapter that records the existing P2PFL validation lifecycle."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from brbfl.ca import (
    CAEvidenceMapper,
    CAStateSnapshot,
    CATransitionEngine,
    CATransitionInput,
    CATransitionPolicy,
    CATransitionProvenance,
    FinalizedTrustEvidenceMapper,
)
from brbfl.canonical import canonical_hash
from brbfl.detectors.evidence import DetectorEvidence
from brbfl.detectors.reference import median_reference
from brbfl.detectors.sign_flipping import SignFlippingDetector
from brbfl.ledger import BlockchainLedger, create_ledger, disabled_ledger_artifact
from brbfl.selection.roles import (
    CARoleSelectionPolicy,
    CAStateRoleSelector,
    RoundRoleAssignment,
    SelectionContext,
    StaticRoundRoleSelector,
    TrustRankedContributorSelector,
    TrustRankedValidatorSelector,
)
from brbfl.trust import TrustRuntime


@dataclass(frozen=True)
class RuntimeLedgerConfig:
    """Process-local runtime binding selected by an experiment configuration."""

    enabled: bool = False
    backend: str = "memory"
    fail_closed: bool = True
    trust_enabled: bool = False
    trust_prior_alpha: float = 1.0
    trust_prior_beta: float = 1.0
    trust_observation_only: bool = True
    selection_strategy: str = "static"
    validator_eligible_participants: tuple[str, ...] = ()
    validator_target_count: int = 0
    validator_minimum_trust: float = 0.5
    validator_bootstrap_rounds: int = 1
    ca_enabled: bool = False
    ca_topology: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    ca_policy: CATransitionPolicy = field(default_factory=CATransitionPolicy)
    ca_suspicious_contributors: bool = True
    ca_minimum_contributors: int = 1
    ca_minimum_validators: int = 1
    ca_validator_quorum: int = 1
    ca_bootstrap_rounds: int = 1
    detector_target_counts: Mapping[str, int] = field(default_factory=dict)
    detector_capabilities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    contributor_minimum_count: int = 3


class RuntimeLedgerAdapter:
    """Translate workflow facts into ledger commits without making decisions."""

    def __init__(
        self,
        config: RuntimeLedgerConfig,
        experiment_id: str,
        participants: tuple[str, ...],
        contributors: tuple[str, ...],
        validators: tuple[str, ...],
        *,
        ca_evidence_mapper: CAEvidenceMapper | None = None,
    ) -> None:
        """Initialize and, when enabled, register the canonical experiment."""
        self.config = config
        self.experiment_id = experiment_id
        self.participants = tuple(sorted(participants))
        self.contributors = tuple(sorted(contributors))
        self.validators = tuple(sorted(validators))
        self._lock = threading.RLock()
        self._detector_condition = threading.Condition(self._lock)
        self._detector_models: dict[int, dict[str, tuple]] = {}
        self._detector_results: dict[int, dict[str, bool]] = {}
        self._detector_errors: dict[int, Exception] = {}
        self._ledger = create_ledger(
            enabled=config.enabled,
            backend=config.backend,
            ledger_identifier=f"memory:{experiment_id}",
        )
        self._parents: dict[int, str] = {}
        self._candidate_hashes: dict[int, dict[str, str]] = {}
        self._admissions: dict[int, dict[str, bool]] = {}
        self._aggregates: dict[int, str] = {}
        self._assignments: dict[int, RoundRoleAssignment] = {}
        self._ca_snapshots: dict[int, CAStateSnapshot] = {}
        self._ca_provenance: dict[int, CATransitionProvenance] = {}
        trust_population = config.validator_eligible_participants or self.validators
        self.validator_eligible_participants = tuple(sorted(trust_population))
        if config.detector_target_counts or config.selection_strategy == "trust_contributors":
            trust_population = tuple(sorted(set(trust_population) | set(self.contributors)))
        if config.selection_strategy in ("trust_ranked", "trust_contributors") and (
            not config.enabled or not config.trust_enabled or config.trust_observation_only
        ):
            raise ValueError("trust-ranked selection requires enabled, enforcement-capable trust")
        self._trust = (
            TrustRuntime(experiment_id, trust_population, config.trust_prior_alpha, config.trust_prior_beta)
            if config.trust_enabled
            else None
        )
        if config.ca_enabled and (not config.enabled or not config.trust_enabled):
            raise ValueError("CA requires enabled runtime ledger and finalized trust evidence")
        if config.selection_strategy in ("ca_state", "ca_contributors") and not config.ca_enabled:
            raise ValueError("CA-state selection requires enabled CA")
        if config.selection_strategy not in ("static", "trust_ranked", "ca_state", "trust_contributors", "ca_contributors"):
            raise ValueError(f"unsupported role selection strategy: {config.selection_strategy}")
        if config.detector_target_counts and (not config.enabled or not config.trust_enabled):
            raise ValueError("specialized detector requires ledger and trust")
        if set(config.detector_capabilities) - set(self.participants):
            raise ValueError("detector capabilities reference unknown participants")
        if config.ca_enabled:
            topology = config.ca_topology or {
                node: tuple(other for other in self.participants if other != node) for node in self.participants
            }
            self._ca_topology = {node: tuple(neighbors) for node, neighbors in topology.items()}
            self._ca_mapper = ca_evidence_mapper or FinalizedTrustEvidenceMapper()
            self._ca_snapshots[0] = CATransitionEngine.initialize(experiment_id, self.participants, config.ca_policy, self._ca_topology)
        if self._ledger is not None:
            self._initialize()

    @property
    def ledger(self) -> BlockchainLedger | None:
        """Return the configured backend for tests and artifact generation."""
        return self._ledger

    def _initialize(self) -> None:
        assert self._ledger is not None
        self._ledger.start_experiment(self.experiment_id, self.participants)
        capabilities = {
            node: frozenset(
                role
                for role, members in (
                    ("contributor", self.contributors),
                    ("validator", self.validator_eligible_participants),
                    ("aggregator", self.contributors),
                )
                if node in members
            )
            or frozenset({"participant"})
            for node in self.participants
        }
        for node in self.participants:
            specialized = self.config.detector_capabilities.get(node, ())
            if specialized:
                capabilities[node] = capabilities[node] | frozenset({"detector", *specialized})
            self._ledger.register_participant(self.experiment_id, node, capabilities[node])
        self._capabilities = capabilities
        groups = {}
        for capability, count in sorted(self.config.detector_target_counts.items()):
            eligible = sorted(node for node, caps in capabilities.items() if {"detector", capability} <= caps)
            if type(count) is not int or count < 1 or len(eligible) < count:
                raise ValueError("insufficient specialized detector candidates or invalid target")
            groups[capability] = tuple(eligible[:count])
        self._selector = StaticRoundRoleSelector(self.contributors, self.validators, self.contributors, groups)
        if self.config.selection_strategy == "trust_ranked":
            self._selector = TrustRankedValidatorSelector(
                self._selector,
                self.validator_eligible_participants,
                self.config.validator_target_count,
                self.config.validator_minimum_trust,
                self.config.validator_bootstrap_rounds,
            )
        elif self.config.selection_strategy == "trust_contributors":
            self._selector = TrustRankedContributorSelector(
                self._selector,
                max(3 if groups else 1, self.config.contributor_minimum_count),
                self.config.validator_minimum_trust,
                self.config.validator_bootstrap_rounds,
            )
        elif self.config.selection_strategy in ("ca_state", "ca_contributors"):
            self._selector = CAStateRoleSelector(
                self._selector,
                CARoleSelectionPolicy(
                    contributor_target_count=len(self.contributors),
                    validator_target_count=len(self.validators),
                    minimum_contributors=self.config.ca_minimum_contributors,
                    minimum_validators=self.config.ca_minimum_validators,
                    validator_quorum=self.config.ca_validator_quorum,
                    suspicious_contributors=self.config.ca_suspicious_contributors,
                    bootstrap_rounds=self.config.ca_bootstrap_rounds,
                    detector_target_counts=self.config.detector_target_counts,
                    fixed_sensitive_roles=self.config.selection_strategy == "ca_contributors",
                ),
            )

    def _invoke(self, operation, *args, **kwargs):
        if self._ledger is None:
            return None
        try:
            return operation(*args, **kwargs)
        except Exception:
            if self.config.fail_closed:
                raise
            return None

    def open_round(self, round_number: int, parent_model_hash: str) -> None:
        """Commit static roles and the workflow-observed parent model."""
        with self._lock:
            previous = self._parents.get(int(round_number))
            if previous is not None and previous != parent_model_hash:
                self._invoke(lambda: (_ for _ in ()).throw(RuntimeError("conflicting runtime round parent model hash")))
                return
            if previous is not None:
                return
            if self._ledger is None:
                return
            round_number = int(round_number)
            ca_snapshot = None
            if self.config.ca_enabled:
                ca_snapshot = self._ca_snapshots.get(round_number)
                if ca_snapshot is None or ca_snapshot.generation != round_number:
                    raise RuntimeError(f"verified CA snapshot is unavailable for round selection: {round_number}")
            assignment = self._selector.select_roles(
                SelectionContext(
                    self.experiment_id,
                    round_number,
                    self._capabilities,
                    ca_snapshot.snapshot_hash if ca_snapshot is not None else self._aggregates.get(round_number - 1),
                    trust_scores=(
                        (
                            {
                                node: self._trust.snapshots[round_number - 1].post_round[node].score
                                if node in self._trust.snapshots[round_number - 1].post_round
                                else 0.5
                                for node in self.participants
                            }
                            if round_number > 0 and round_number - 1 in self._trust.snapshots
                            else {node: state.score for node, state in self._trust.states.items()}
                        )
                        if self._trust is not None
                        else {}
                    ),
                    ca_snapshot=ca_snapshot,
                    source_trust_round=(
                        round_number - 1
                        if self._trust is not None and round_number - 1 in self._trust.snapshots else None
                    ),
                    source_trust_hash=(
                        self._trust.snapshots[round_number - 1].snapshot_sha256
                        if self._trust is not None and round_number - 1 in self._trust.snapshots
                        else None
                    ),
                )
            )
            self._invoke(self._ledger.commit_round_roles, assignment)
            self._invoke(self._ledger.open_round, self.experiment_id, int(round_number), parent_model_hash)
            self._assignments[int(round_number)] = assignment
            self._parents[int(round_number)] = parent_model_hash

    def detect_model_submission(
        self, round_number: int, participant_id: str, parameters: list, *, publisher_id: str, timeout: float = 60,
    ) -> bool:
        """Coordinate a complete observable candidate set before model admission."""
        from brbfl.validation.byzantine_gate import canonical_parameters, parameter_hash

        capability = "model_update_detector"
        if capability not in self.config.detector_target_counts:
            return True
        with self._detector_condition:
            if round_number in self._detector_errors:
                raise RuntimeError("model-update detection previously failed") from self._detector_errors[round_number]
            assignment = self.role_assignment(round_number)
            if participant_id not in assignment.selected_contributors:
                raise ValueError("detector submission is not a selected contributor")
            if publisher_id == participant_id:
                snapshot = canonical_parameters(parameters)
                committed = self._candidate_hashes.get(round_number, {}).get(participant_id)
                if committed != parameter_hash(snapshot):
                    raise ValueError("detector submission differs from the committed candidate")
                models = self._detector_models.setdefault(round_number, {})
                existing = models.get(participant_id)
                if existing is not None and parameter_hash(existing) != committed:
                    raise RuntimeError("conflicting detector submission")
                models[participant_id] = snapshot
                if set(models) == set(assignment.selected_contributors) and round_number not in self._detector_results:
                    try:
                        reference, reference_provenance = median_reference(models)
                        results = {}
                        for candidate, values in sorted(models.items()):
                            source_hash = self._candidate_hashes[round_number][candidate]
                            evidence = tuple(
                                replace(
                                    SignFlippingDetector(member).evaluate(
                                        experiment_id=self.experiment_id, round_number=round_number,
                                        detector_assignment_hash=assignment.assignment_hash,
                                        evaluated_participant_id=candidate, candidate_update=values, reference_update=reference,
                                    ),
                                    source_artifact_hash=source_hash,
                                )
                                for member in assignment.detector_subgroups[capability]
                            )
                            evidence = tuple(replace(row, measurement_hash=canonical_hash(
                                "CommitteeReferenceMeasurement/v1",
                                {"measurement_hash": row.measurement_hash, "reference_provenance_hash": reference_provenance},
                            )) for row in evidence)
                            self.record_detector_committee(round_number, capability, candidate, source_hash, evidence)
                            assert self._ledger is not None
                            committed_rows = self._ledger.get_round_record(self.experiment_id, round_number)["detector_committees"]
                            committed_decision = next(
                                row["decision"] for row in committed_rows
                                if row["decision"]["participant_id"] == candidate and row["decision"]["capability"] == capability
                            )
                            results[candidate] = committed_decision["category"] not in ("severe", "negative")
                        self._detector_results[round_number] = results
                    except Exception as exc:
                        self._detector_errors[round_number] = exc
                    finally:
                        self._detector_condition.notify_all()
            ready = self._detector_condition.wait_for(
                lambda: round_number in self._detector_results or round_number in self._detector_errors, timeout=timeout,
            )
            if not ready:
                error = RuntimeError("timed out waiting for complete detector submissions")
                self._detector_errors[round_number] = error
                self._detector_condition.notify_all()
                raise error
            if round_number in self._detector_errors:
                raise RuntimeError("model-update detection failed") from self._detector_errors[round_number]
            return self._detector_results[round_number][participant_id]

    def record_detector_committee(
        self, round_number: int, capability: str, participant_id: str,
        source_artifact_hash: str, evidence: tuple[DetectorEvidence, ...],
    ) -> None:
        """Commit a committee result through the authoritative ledger boundary."""
        with self._lock:
            if self._ledger is None:
                return
            self._invoke(
                self._ledger.commit_detector_committee, self.experiment_id, round_number,
                capability, participant_id, source_artifact_hash, evidence,
            )

    def role_assignment(self, round_number: int) -> RoundRoleAssignment:
        """Return the immutable authoritative assignment for an opened round."""
        with self._lock:
            try:
                return self._assignments[int(round_number)]
            except KeyError as exc:
                raise RuntimeError(f"round role assignment is unavailable before open_round: {round_number}") from exc

    def selected_validators(self, round_number: int) -> tuple[str, ...]:
        """Return validators from the frozen assignment for an opened round."""
        return self.role_assignment(round_number).selected_validators

    def record_candidate(
        self,
        round_number: int,
        contributor_id: str,
        parent_model_hash: str,
        candidate_hash: str,
        votes: list[dict[str, Any]],
        *,
        publisher_id: str | None = None,
    ) -> None:
        """Record decisions only from the candidate owner's authoritative callback."""
        with self._lock:
            # Every process-local node gate observes transported candidates, but
            # the TrainStage callback of the candidate owner is the workflow's
            # sole producer.  Receiver gates verify locally and must not turn
            # their independently ordered audit rows into ledger commitments.
            publisher_id = contributor_id if publisher_id is None else publisher_id
            if publisher_id != contributor_id:
                return
            self.open_round(round_number, parent_model_hash)
            if self._ledger is None:
                return
            self._invoke(
                self._ledger.commit_candidate,
                self.experiment_id,
                int(round_number),
                contributor_id,
                parent_model_hash,
                candidate_hash,
            )
            self._candidate_hashes.setdefault(int(round_number), {})[contributor_id] = candidate_hash
            for vote in sorted(votes, key=lambda row: row["validator_node_id"]):
                self._invoke(
                    self._ledger.record_validator_decision,
                    self.experiment_id,
                    int(round_number),
                    vote["validator_node_id"],
                    contributor_id,
                    candidate_hash,
                    bool(vote["reported_decision"]),
                    {
                        key: vote[key]
                        for key in (
                            "vote_sha256",
                            "reference_decision",
                            "byzantine",
                            "strategy",
                            "attack_group_id",
                            "order_index",
                        )
                        if key in vote
                    },
                )

    def finalize_admission(self, round_number: int, decisions: dict[str, bool]) -> None:
        """Commit the admission map calculated by workflow coordination."""
        with self._lock:
            if self._ledger is None:
                return
            canonical = {node: bool(decisions[node]) for node in sorted(decisions)}
            self._invoke(self._ledger.finalize_admission, self.experiment_id, int(round_number), canonical)
            self._admissions[int(round_number)] = canonical

    def confirm_installation(self, round_number: int, participant_id: str, aggregate_hash: str) -> None:
        """Commit the workflow aggregate once, then its verified installation callback."""
        with self._lock:
            if self._ledger is None:
                return
            round_number = int(round_number)
            if round_number not in self._aggregates:
                admission = self._admissions.get(round_number)
                if admission is None:
                    self._invoke(lambda: (_ for _ in ()).throw(RuntimeError("aggregate observed before admission finalization")))
                    return
                inputs = {node: self._candidate_hashes[round_number][node] for node, admitted in admission.items() if admitted}
                self._invoke(self._ledger.commit_aggregate, self.experiment_id, round_number, inputs, aggregate_hash)
                self._aggregates[round_number] = aggregate_hash
            elif self._aggregates[round_number] != aggregate_hash:
                self._invoke(lambda: (_ for _ in ()).throw(RuntimeError("conflicting runtime aggregate hash")))
                return
            self._invoke(
                self._ledger.confirm_model_installation,
                self.experiment_id,
                round_number,
                participant_id,
                aggregate_hash,
            )

    def finalize_round(self, round_number: int) -> None:
        """Verify and finalize only after the workflow's network-wide barrier."""
        with self._lock:
            if self._ledger is None:
                return
            round_number = int(round_number)
            self._invoke(self._ledger.finalize_round, self.experiment_id, round_number)
            verified = self._invoke(self._ledger.verify_round, self.experiment_id, round_number)
            if (self.config.ca_enabled or self.config.selection_strategy == "trust_contributors"
                    or self.config.detector_target_counts) and verified is not True:
                raise RuntimeError("CA/trust cannot consume an unverified ledger round")
            if self._trust is not None and int(round_number) not in self._trust.snapshots:
                record = self._ledger.get_round_record(self.experiment_id, int(round_number))
                decisions = [
                    {
                        "validator_id": row["validator_id"],
                        "candidate_id": row["contributor_id"],
                        "reported_decision": row["admitted"],
                        "reference_decision": row["evidence"].get("reference_decision") if row.get("evidence") else None,
                    }
                    for row in record["decisions"].values()
                ]
                self._trust.finalize_round(
                    int(round_number),
                    record["assignment"]["selected_validators"],
                    record["candidates"],
                    decisions,
                    detector_decisions=[row["decision"] for row in record.get("detector_committees", ())],
                )
            if self.config.ca_enabled and round_number not in self._ca_provenance:
                self._finalize_ca(round_number)

    def _finalize_ca(self, round_number: int) -> None:
        """Derive and atomically commit the next generation from finalized inputs."""
        assert self._ledger is not None and self._trust is not None
        previous = self._ca_snapshots.get(round_number)
        trust = self._trust.snapshots.get(round_number)
        if previous is None or previous.generation != round_number or previous.source_round not in (None, round_number - 1):
            raise RuntimeError("CA source snapshot is absent or stale")
        if trust is None or trust.round_id != round_number or trust.experiment_id != self.experiment_id:
            raise RuntimeError("trust snapshot is absent, unverified, or stale")
        ledger_hash = self._ledger.final_event_chain_hash
        if not ledger_hash:
            raise RuntimeError("finalized ledger hash is absent")
        categories = self._ca_mapper.categories(self.participants, trust)
        trust_scores = {node: trust.post_round[node].score if node in trust.post_round else 0.5 for node in self.participants}
        result = CATransitionEngine.transition(
            previous,
            CATransitionInput(round_number, trust_scores, categories, self._ca_topology),
            self.config.ca_policy,
        )
        provenance = CATransitionProvenance(
            round_number,
            ledger_hash,
            trust.snapshot_sha256,
            result.topology_hash,
            previous.snapshot_hash,
            self.config.ca_policy.policy_hash,
            result.snapshot_hash,
        )
        payload = {
            **provenance.artifact(),
            "transition_records": [
                {
                    "participant_id": record.participant_id,
                    "previous_state": record.previous_state.value,
                    "next_state": record.next_state.value,
                    "evidence": record.evidence_category.value,
                    "reason_code": record.reason_code,
                }
                for record in result.transition_records
            ],
        }
        self._invoke(self._ledger.commit_ca_transition, self.experiment_id, round_number, payload)
        self._ca_snapshots[result.generation] = result
        self._ca_provenance[round_number] = provenance

    def ca_snapshot(self, generation: int) -> CAStateSnapshot:
        """Return an immutable generation snapshot, failing closed if unavailable."""
        with self._lock:
            try:
                return self._ca_snapshots[int(generation)]
            except KeyError as exc:
                raise RuntimeError(f"CA snapshot is unavailable: generation={generation}") from exc

    def ca_snapshot_for_round(self, round_number: int) -> CAStateSnapshot:
        """Return the snapshot used to select and execute ``round_number``."""
        return self.ca_snapshot(round_number)

    def ca_provenance(self, source_round: int) -> CATransitionProvenance:
        """Return immutable provenance retained for a finalized source round."""
        with self._lock:
            try:
                return self._ca_provenance[int(source_round)]
            except KeyError as exc:
                raise RuntimeError(f"CA transition is unavailable: source_round={source_round}") from exc

    def ca_artifact(self) -> dict[str, object] | None:
        """Return deterministic CA evidence only when CA is explicitly enabled."""
        with self._lock:
            if not self.config.ca_enabled:
                return None
            return {
                "enabled": True,
                "mapping_policy": "all_agree=positive;any_disagreement=severe;unevaluated=neutral",
                "generations": {
                    str(generation): {
                        "source_round": snapshot.source_round,
                        "snapshot_hash": snapshot.snapshot_hash,
                        "previous_snapshot_hash": snapshot.previous_snapshot_hash,
                        "topology_hash": snapshot.topology_hash,
                        "transition_policy_hash": snapshot.policy_hash,
                        "participant_states": {
                            node: {
                                "state": snapshot.participant_states[node].state.value,
                                "consecutive_positive": snapshot.participant_states[node].consecutive_positive_rounds,
                                "consecutive_negative": snapshot.participant_states[node].consecutive_negative_rounds,
                                "rounds_in_state": snapshot.participant_states[node].rounds_in_state,
                                "last_transition_round": snapshot.participant_states[node].last_transition_round,
                                "unresolved_severe_since_round": snapshot.participant_states[node].unresolved_severe_since_round,
                            }
                            for node in sorted(snapshot.participant_states)
                        },
                        "transition_records": [
                            {
                                "participant_id": record.participant_id,
                                "previous_state": record.previous_state.value,
                                "next_state": record.next_state.value,
                                "evidence_category": record.evidence_category.value,
                                "reason_code": record.reason_code,
                            }
                            for record in snapshot.transition_records
                        ],
                    }
                    for generation, snapshot in sorted(self._ca_snapshots.items())
                },
                "transitions": {
                    str(source_round): provenance.artifact() for source_round, provenance in sorted(self._ca_provenance.items())
                },
            }

    def validation_artifact(self) -> dict[str, Any]:
        """Return deterministic ledger evidence for ``validation.json``."""
        with self._lock:
            if self._ledger is None:
                return {**disabled_ledger_artifact(), "fail_closed": self.config.fail_closed}
            artifact = self._ledger.validation_artifact(self.experiment_id)
            artifact["fail_closed"] = self.config.fail_closed
            if isinstance(self._selector, TrustRankedValidatorSelector):
                artifact["selection"] = self._selector.artifact()
            return artifact

    def trust_artifact(self) -> dict[str, Any] | None:
        """Return trust only when explicitly enabled, preserving old artifacts."""
        with self._lock:
            return self._trust.artifact(self.config.trust_observation_only) if self._trust is not None else None


_runtime_adapter: RuntimeLedgerAdapter | None = None
_runtime_lock = threading.RLock()


def install_runtime_ledger(adapter: RuntimeLedgerAdapter) -> None:
    """Bind one ledger adapter to the current process experiment."""
    global _runtime_adapter
    with _runtime_lock:
        _runtime_adapter = adapter


def get_runtime_ledger() -> RuntimeLedgerAdapter | None:
    """Return the active adapter, including a configured disabled adapter."""
    with _runtime_lock:
        return _runtime_adapter


def clear_runtime_ledger() -> None:
    """Remove runtime state during deterministic experiment shutdown."""
    global _runtime_adapter
    with _runtime_lock:
        _runtime_adapter = None
