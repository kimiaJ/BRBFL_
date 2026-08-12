# ruff: noqa: D103
"""Focused deterministic trust-ranked validator selection tests."""

import numpy as np
import pytest

from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig, clear_runtime_ledger, install_runtime_ledger
from brbfl.selection import SelectionContext, StaticRoundRoleSelector, TrustRankedValidatorSelector
from brbfl.validation import AdmissionPolicy, ValidatorSubgroupGate

CAPS = {f"node-{i}": frozenset({"contributor", "validator", "aggregator"}) for i in range(5)}


def selector():
    return TrustRankedValidatorSelector(
        StaticRoundRoleSelector(("node-0", "node-1", "node-2"), ("node-0", "node-3", "node-4"), ("node-0", "node-1", "node-2")),
        tuple(CAPS),
        3,
    )


def test_bootstrap_and_next_round_use_only_prior_finalized_trust():
    value = selector()
    first = value.select_roles(SelectionContext("x", 0, CAPS, trust_scores={n: 0.99 for n in CAPS}))
    assert first.selected_validators == ("node-0", "node-3", "node-4")
    scores = {"node-0": 0.8, "node-1": 0.5, "node-2": 0.5, "node-3": 0.2, "node-4": 0.2}
    second = value.select_roles(SelectionContext("x", 1, CAPS, "finalized-round-0", scores))
    assert second.selected_validators == ("node-0", "node-1", "node-2")
    assert second.selected_contributors == first.selected_contributors
    assert second.aggregation_eligible_nodes == first.aggregation_eligible_nodes


def test_ranking_is_order_independent_and_fail_closed():
    scores = dict(reversed(list({"node-0": 0.8, "node-1": 0.5, "node-2": 0.5, "node-3": 0.2, "node-4": 0.2}.items())))
    value = selector()
    value.select_roles(SelectionContext("x", 0, CAPS))
    assert value.select_roles(SelectionContext("x", 1, CAPS, "h", scores)).selected_validators == ("node-0", "node-1", "node-2")
    value = TrustRankedValidatorSelector(StaticRoundRoleSelector((), ("node-0",), ()), tuple(CAPS), 1, minimum_trust=0.9)
    value.select_roles(SelectionContext("x", 0, CAPS))
    with pytest.raises(RuntimeError, match="insufficient"):
        value.select_roles(SelectionContext("x", 1, CAPS, "h", {n: 0.5 for n in CAPS}))


def test_gate_uses_each_rounds_frozen_dynamic_validator_assignment():
    participants = tuple(CAPS)
    contributors = ("node-0", "node-1", "node-2")
    bootstrap = ("node-0", "node-3", "node-4")
    runtime = RuntimeLedgerAdapter(
        RuntimeLedgerConfig(
            enabled=True,
            trust_enabled=True,
            trust_observation_only=False,
            selection_strategy="trust_ranked",
            validator_eligible_participants=participants,
            validator_target_count=3,
        ),
        "dynamic-gate-lifecycle",
        participants,
        contributors,
        bootstrap,
    )
    install_runtime_ledger(runtime)
    gate = ValidatorSubgroupGate(
        AdmissionPolicy(contributors, bootstrap, ("node-3", "node-4"), quorum=3, acceptance_threshold=2)
    )

    def submit_round(round_number, parent, *, check_unselected=False):
        for index, candidate in enumerate(reversed(contributors)):
            gate.submit_and_decide(
                round_number,
                candidate,
                [np.array([index + round_number], dtype=np.float32)],
                current_node=candidate,
                parent_global_model_sha256=parent,
            )
        rows = [row for row in gate.evidence() if row["round"] == round_number]
        if check_unselected:
            record = runtime.ledger.get_round_record(runtime.experiment_id, round_number)
            with pytest.raises(RuntimeError, match="decision submitter is not a selected validator: node-3"):
                runtime.ledger.record_validator_decision(
                    runtime.experiment_id,
                    round_number,
                    "node-3",
                    "node-0",
                    record["candidates"]["node-0"]["candidate_hash"],
                    True,
                )
        runtime.finalize_admission(round_number, {row["candidate_node_id"]: row["admitted"] for row in rows})
        aggregate_hash = f"aggregate-{round_number}"
        for participant in participants:
            runtime.confirm_installation(round_number, participant, aggregate_hash)
        runtime.finalize_round(round_number)
        return rows

    try:
        round_zero = submit_round(0, "genesis")
        assert runtime.selected_validators(0) == bootstrap
        assert sum(len(row["votes"]) for row in round_zero) == 9
        assert sum(vote["attack_application_count"] for row in round_zero for vote in row["votes"]) == 6
        assert {node: state.score for node, state in runtime._trust.states.items()} == {
            "node-0": 0.8,
            "node-1": 0.5,
            "node-2": 0.5,
            "node-3": 0.2,
            "node-4": 0.2,
        }

        round_one = submit_round(1, "aggregate-0", check_unselected=True)
        selected = ("node-0", "node-1", "node-2")
        assert runtime.selected_validators(1) == selected
        assert sum(len(row["votes"]) for row in round_one) == 9
        assert {vote["validator_node_id"] for row in round_one for vote in row["votes"]} == set(selected)
        assert not any(vote["byzantine"] for row in round_one for vote in row["votes"])
        assert all(vote["reported_decision"] == vote["reference_decision"] for row in round_one for vote in row["votes"])

        assert runtime.validation_artifact()["round_verification"] == {"0": True, "1": True}
        states = runtime._trust.states
        assert (states["node-0"].alpha, states["node-0"].beta, states["node-0"].processed_vote_count) == (7, 1, 6)
        for node in ("node-1", "node-2"):
            assert (states[node].alpha, states[node].beta, states[node].processed_vote_count) == (4, 1, 3)
        for node in ("node-3", "node-4"):
            assert (states[node].alpha, states[node].beta, states[node].processed_vote_count) == (1, 4, 3)
    finally:
        clear_runtime_ledger()
