"""Three-round integration through the production gate, trust, CA and selection."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from brbfl.ca import CATransitionPolicy
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig, clear_runtime_ledger, install_runtime_ledger
from brbfl.validation.byzantine_gate import (
    AdmissionPolicy,
    ValidatorSubgroupGate,
    canonical_aggregation_inputs,
    clear_validator_gate,
    get_validator_gate,
    install_validator_gate,
    publish_admission_decision,
    wait_for_admitted_contributors,
)


@pytest.fixture
def paired_artifacts():
    """Construct controlled numeric integration artifacts, without claiming MNIST execution."""
    artifacts = []
    for attacked in (False, True):
        nodes = ("a", "b", "c", "d", "v", "w")
        contributors, validators = nodes[:4], nodes[4:]
        runtime = RuntimeLedgerAdapter(
            RuntimeLedgerConfig(
                enabled=True,
                trust_enabled=True,
                trust_observation_only=False,
                ca_enabled=True,
                selection_strategy="ca_state",
                validator_eligible_participants=validators,
                ca_policy=CATransitionPolicy(recovery_enabled=False),
                detector_target_counts={"model_update_detector": 2},
                detector_capabilities={node: ("model_update_detector",) for node in ("a", "v", "w")},
            ),
            "numeric-integration",
            nodes,
            contributors,
            validators,
        )
        install_runtime_ledger(runtime)
        install_validator_gate(ValidatorSubgroupGate(AdmissionPolicy(contributors, validators, quorum=2, acceptance_threshold=2)))
        rounds = []
        try:
            parent = "initial"
            for number in range(3):
                runtime.open_round(number, parent)
                selected = runtime.role_assignment(number).selected_contributors
                models = {node: [np.array([1.0, 2.0]) * (-3 if attacked and node == "b" else 1)] for node in selected}
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {
                        node: pool.submit(
                            get_validator_gate(node).submit_and_decide,
                            number,
                            node,
                            model,
                            current_node=node,
                            parent_global_model_sha256=parent,
                        )
                        for node, model in models.items()
                    }
                    decisions = {node: future.result(timeout=3) for node, future in futures.items()}
                for node, admitted in decisions.items():
                    publish_admission_decision(number, node, admitted)
                    if admitted:
                        get_validator_gate(node).observe_aggregation_input(number, node, models[node])
                admitted = wait_for_admitted_contributors(number, timeout=0)
                assert set(admitted) == set(selected) - ({"b"} if attacked else set())
                canonical_aggregation_inputs(number, list(admitted))
                if number == 2 and attacked:
                    with pytest.raises(RuntimeError, match="unselected"):
                        publish_admission_decision(number, "b", True)
                digest = str(number) * 64
                for node in nodes:
                    runtime.confirm_installation(number, node, digest)
                runtime.finalize_round(number)
                parent = digest
                roles = {
                    node: {
                        "selected_trainers": list(selected),
                        "actually_trained": node in selected,
                        "submitted_candidate": node in selected,
                    }
                    for node in nodes
                }
                rounds.append(
                    {
                        "round": number,
                        "trainer_roles": {
                            "per_node_selection": roles,
                            "actual_training_nodes": list(selected),
                            "submitted_candidates": list(selected),
                        },
                    }
                )
            artifacts.append(
                {
                    "execution_mode": "real_p2pfl_runtime",  # Schema fixture only; not a real MNIST run.
                    "configured_rounds": 3,
                    "configuration": {"rounds": 3},
                    "provenance": {"producing_commit": "fixture", "partition_manifest_sha256": "partitions"},
                    "attack_type": "sign_flipping" if attacked else "none",
                    "malicious_node_ids": ["b"] if attacked else [],
                    "ledger": runtime.validation_artifact(),
                    "trust": runtime.trust_artifact(),
                    "ca": runtime.ca_artifact(),
                    "rounds": rounds,
                    "final_model_consensus": True,
                    "per_node_final_installed_model_hashes": dict.fromkeys(nodes, parent),
                }
            )
        finally:
            clear_validator_gate()
            clear_runtime_ledger()
    return json.loads(json.dumps(artifacts))


def test_third_round_excludes_attacker_without_waiting_for_its_submission(paired_artifacts):
    """Exclusion changes real gate membership and still permits finalization."""
    clean, attacked = paired_artifacts
    for number in range(3):
        key = str(number)
        assert attacked["ledger"]["round_verification"][key]
        assert clean["ledger"]["round_verification"][key]
    assert attacked["ca"]["generations"]["1"]["participant_states"]["b"]["state"] == "suspicious"
    assert attacked["ca"]["generations"]["2"]["participant_states"]["b"]["state"] == "excluded"
    assert "b" not in attacked["ledger"]["per_round_role_assignment"]["2"]["selected_contributors"]
    assert "b" not in attacked["ledger"]["rounds"]["2"]["candidate_commitments"]
    assert "b" in clean["ledger"]["per_round_role_assignment"]["2"]["selected_contributors"]


def test_comparison_accepts_complete_chain(paired_artifacts):
    """The comparison checks the numeric integration fixture's recorded chain."""
    from brbfl.experiments.compare_model_detector_runtime import compare_evidence

    assert compare_evidence(*paired_artifacts)["verification_result"] is True


@pytest.mark.parametrize("mutation", ["selection", "training", "decision", "installation", "trust", "provenance"])
def test_comparison_rejects_inconsistent_receipts(paired_artifacts, mutation):
    """Success flags cannot hide a contradictory lifecycle receipt."""
    import copy

    from brbfl.experiments.compare_model_detector_runtime import compare_evidence

    clean, attacked = copy.deepcopy(paired_artifacts)
    if mutation == "selection":
        attacked["ledger"]["per_round_role_assignment"]["2"]["selected_contributors"].append("b")
    elif mutation == "training":
        attacked["rounds"][2]["trainer_roles"]["per_node_selection"]["b"]["actually_trained"] = True
    elif mutation == "decision":
        attacked["ledger"]["rounds"]["0"]["detector_committees"][0]["decision"]["category"] = "tampered"
    elif mutation == "installation":
        attacked["ledger"]["rounds"]["2"]["installation_confirmations"]["b"] = "wrong"
    elif mutation == "trust":
        attacked["trust"]["rounds"]["0"]["detector_updates"] = []
    else:
        attacked["provenance"]["partition_manifest_sha256"] = "wrong"
    with pytest.raises(AssertionError):
        compare_evidence(clean, attacked)


def test_workflow_uses_frozen_assignment_instead_of_election(monkeypatch):
    """The actual election stage installs ledger-selected trainers before fit."""
    import random
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    from p2pfl.stages.base_node.vote_train_set_stage import VoteTrainSetStage
    from p2pfl.stages.stage_factory import StageFactory

    runtime = RuntimeLedgerAdapter(RuntimeLedgerConfig(enabled=True), "workflow", ("a", "b", "v"), ("a", "b"), ("v",))
    install_runtime_ledger(runtime)
    state = SimpleNamespace(
        addr="a", round=0, aggregated_model_event=threading.Event(), eligible_trainers=("a", "b"), trainer_role_evidence={}
    )
    protocol = Mock()
    protocol.get_neighbors.return_value = ["b", "v"]
    learner = Mock()
    learner.get_model.return_value.get_parameters.return_value = [np.array([1.0])]
    monkeypatch.setattr(VoteTrainSetStage, "_VoteTrainSetStage__vote", Mock())
    monkeypatch.setattr(VoteTrainSetStage, "_VoteTrainSetStage__aggregate_votes", Mock(return_value=["a"]))
    monkeypatch.setattr(StageFactory, "get_stage", Mock(return_value="train"))
    try:
        result = VoteTrainSetStage.execute(
            trainset_size=2, state=state, communication_protocol=protocol, generator=random.Random(0), learner=learner
        )
        assert result == "train"
        assert state.train_set == ["a", "b"]
        assert state.trainer_role_evidence[0]["protocol_selected_trainers"] == ["a"]
    finally:
        clear_runtime_ledger()
