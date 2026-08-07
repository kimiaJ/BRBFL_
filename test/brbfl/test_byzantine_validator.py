"""Focused tests for the real validator-subgroup admission decision."""
# ruff: noqa: D103

from unittest.mock import Mock

import numpy as np
import pytest

from brbfl.experiments.config import load_experiment_config
from brbfl.validation import AdmissionPolicy, ValidatorSubgroupGate, canonical_parameters, parameter_hash
from p2pfl.node_state import NodeState
from p2pfl.stages.base_node.train_stage import TrainStage
from p2pfl.stages.base_node.vote_train_set_stage import VoteTrainSetStage


def _gate(byzantine=()):
    return ValidatorSubgroupGate(
        AdmissionPolicy(
            contributors=("node-0", "node-1", "node-2"),
            validators=("node-0", "node-3", "node-4"),
            byzantine_validators=byzantine,
            quorum=3,
            acceptance_threshold=2,
            group_id="test-group",
            reference_reject_candidates=("node-0",),
        )
    )


def test_controlled_configs_match_except_attack_and_output():
    clean = load_experiment_config("configs/smoke/mnist_byzantine_validator_clean.yaml")
    attacked = load_experiment_config("configs/smoke/mnist_byzantine_validator.yaml")
    assert clean.validation == attacked.validation
    assert clean.validation.contributors == ("node-0", "node-1", "node-2")
    assert clean.validation.validators == ("node-0", "node-3", "node-4")
    assert attacked.attack.adversaries == (3, 4)
    assert clean.eligible_trainers == attacked.eligible_trainers == ("node-0", "node-1", "node-2")
    assert set(clean.validation.validators) - set(clean.validation.contributors) == {"node-3", "node-4"}


class _SingleNodeProtocol:
    def get_neighbors(self, only_direct=False):
        return []

    def build_msg(self, *args, **kwargs):
        return (args, kwargs)

    def broadcast(self, message):
        return None


def test_workflow_election_allowlist_and_pre_fit_role_invariant():
    """Reach the real election and prove a policy violation fails before fit."""
    state = NodeState("node-0")
    state.set_eligible_trainers(("node-0", "node-1", "node-2"))
    state.set_experiment("role-test", 1)
    protocol = _SingleNodeProtocol()
    next_stage = VoteTrainSetStage.execute(
        trainset_size=5,
        state=state,
        communication_protocol=protocol,
        generator=__import__("random").Random(666),
    )
    assert next_stage is TrainStage
    assert state.train_set == ["node-0"]
    assert state.trainer_role_evidence[0]["eligible_trainers"] == ["node-0", "node-1", "node-2"]

    learner = Mock()
    state.eligible_trainers = ("node-1", "node-2")  # simulate corrupt selection/config state
    with pytest.raises(RuntimeError, match=r"refusing to train before learner\.fit"):
        TrainStage.execute(state=state, communication_protocol=protocol, learner=learner, aggregator=Mock())
    learner.fit.assert_not_called()


def test_default_election_remains_unrestricted_and_gate_remains_strict():
    state = NodeState("node-4")
    state.set_experiment("default-role-test", 1)
    next_stage = VoteTrainSetStage.execute(
        trainset_size=5,
        state=state,
        communication_protocol=_SingleNodeProtocol(),
        generator=__import__("random").Random(666),
    )
    assert next_stage is TrainStage
    assert state.eligible_trainers is None
    with pytest.raises(RuntimeError, match="candidate is not an eligible contributor"):
        _gate().submit_and_decide(0, "node-4", [np.array([1.0])])


def test_byzantine_votes_change_admission_without_mutating_candidate():
    candidate = [np.array([1.0, 2.0])]
    before = candidate[0].copy()
    clean, attacked = _gate(), _gate(("node-3", "node-4"))
    assert clean.submit_and_decide(0, "node-1", candidate) is True
    assert attacked.submit_and_decide(0, "node-1", candidate) is False
    assert np.array_equal(candidate[0], before)
    votes = attacked.evidence()[0]["votes"]
    assert [vote["reported_decision"] for vote in votes] == [True, False, False]
    assert [vote["attack_application_count"] for vote in votes] == [0, 1, 1]


def test_accepted_snapshot_reaches_boundary_and_rejected_is_blocked():
    accepted = _gate()
    parameters = [np.array([3.0])]
    assert accepted.submit_and_decide(1, "node-2", parameters)
    accepted.observe_aggregation_input(1, "node-2", parameters)
    row = accepted.evidence()[0]
    assert row["reached_aggregator_add_model"] and row["aggregation_matches_submitted_snapshot"]
    rejected = _gate(("node-3", "node-4"))
    assert not rejected.submit_and_decide(1, "node-2", parameters)
    with pytest.raises(RuntimeError, match="rejected candidate cannot reach aggregation"):
        rejected.observe_aggregation_input(1, "node-2", parameters)


def test_snapshot_aliasing_lifecycle_and_vote_errors_are_descriptive():
    gate = _gate()
    parameters = [np.array([1.0])]
    gate.submit_and_decide(0, "node-1", parameters)
    parameters[0][0] = 9.0
    with pytest.raises(RuntimeError, match="candidate snapshot changed"):
        gate.submit_and_decide(0, "node-1", parameters)
    with pytest.raises(RuntimeError, match="ineligible validator"):
        gate.publish_vote(0, "node-1", "node-9", True)
    with pytest.raises(RuntimeError, match="duplicate validator vote"):
        gate.publish_vote(0, "node-1", "node-0", True)
    with pytest.raises(RuntimeError, match="before candidate submission"):
        _gate().observe_aggregation_input(0, "node-1", [np.array([1.0])])


def test_canonical_snapshot_survives_real_parameter_transport_and_is_immutable():
    torch = pytest.importorskip("torch")
    try:
        from p2pfl.learning.compression.manager import CompressionManager
    except ImportError:
        pytest.skip("optional compression dependency is not installed")
    tensor = torch.tensor([[1.0, -2.0]], dtype=torch.float32, requires_grad=True)
    source = canonical_parameters([tensor, np.array([7], dtype=np.int16)])
    encoded = CompressionManager.apply(list(source), {"evidence": "kept"}, {})
    decoded, metadata = CompressionManager.reverse(encoded)
    received = canonical_parameters(decoded)
    assert parameter_hash(source) == parameter_hash(received)
    assert metadata["evidence"] == "kept"
    assert [item.dtype for item in received] == [np.dtype("float32"), np.dtype("int16")]
    assert [item.shape for item in received] == [(1, 2), (1,)]
    assert all(not item.flags.writeable for item in source)
    assert not np.shares_memory(source[0], tensor.detach().numpy())
    tensor.detach()[0, 0] = 99
    assert source[0][0, 0] == 1


def test_transport_hash_is_verified_and_true_mutation_fails_descriptively():
    sender = _gate()
    receiver = _gate()
    values = [np.array([1.0, 2.0], dtype=np.float32)]
    assert sender.submit_and_decide(0, "node-1", values, current_node="node-1", lifecycle_path="TrainStage")
    expected = sender.submitted_hash(0, "node-1")
    assert receiver.submit_and_decide(
        0,
        "node-1",
        values,
        current_node="node-3",
        lifecycle_path="PartialModelCommand",
        expected_hash=expected,
        transport_occurred=True,
    )
    mutated = [values[0].copy()]
    mutated[0][1] += np.float32(1)
    mutation_receiver = _gate()
    mutation_receiver.submit_and_decide(
        0,
        "node-1",
        values,
        current_node="node-4",
        lifecycle_path="PartialModelCommand",
        expected_hash=expected,
        transport_occurred=True,
    )
    with pytest.raises(RuntimeError, match=r"current_node=node-4.*first_difference=parameter\[0\]\(1,\)"):
        mutation_receiver.submit_and_decide(
            0,
            "node-1",
            mutated,
            current_node="node-4",
            lifecycle_path="PartialModelCommand",
            expected_hash=expected,
            transport_occurred=True,
        )


def test_hash_is_order_sensitive_and_repeated_hashing_is_stable():
    values = [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)]
    assert parameter_hash(values) == parameter_hash(values)
    assert parameter_hash(values) != parameter_hash(list(reversed(values)))


def _comparison_fixture():
    """Return the real two-round causal shape in a compact regression fixture."""
    from copy import deepcopy

    validators = ["node-0", "node-3", "node-4"]
    contributors = ["node-0", "node-1", "node-2"]
    config = {
        "nodes": 5,
        "rounds": 2,
        "epochs": 1,
        "seed": 666,
        "protocol": "memory",
        "framework": "pytorch",
        "aggregator": "fedavg",
        "topology": "full",
        "batch_size": 128,
        "eligible_trainers": contributors,
        "dataset": {"name": "MNIST", "distribution": "iid"},
        "validation": {"contributors": contributors, "validators": validators, "quorum": 3, "acceptance_threshold": 2},
    }
    roles = {
        "configured_contributors": contributors,
        "configured_validators": validators,
        "actual_training_nodes": contributors,
        "eligible_trainers": contributors,
    }

    def vote(candidate, validator, reference, attacked):
        byzantine = validator in {"node-3", "node-4"} and attacked
        return {
            "validator_node_id": validator,
            "reference_decision": reference,
            "reported_decision": not reference if byzantine else reference,
            "attack_application_count": int(byzantine),
            "byzantine": byzantine,
        }

    def build(attacked):
        rows = []
        rounds = []
        parent = "initial"
        for round_number in range(2):
            admitted = []
            inputs = {}
            result = (
                ("attack-global-0" if attacked else "clean-global-0")
                if round_number == 0
                else ("attack-global-1" if attacked else "clean-global-1")
            )
            for candidate in contributors:
                reference = candidate != "node-0"
                is_admitted = reference if not attacked else not reference
                submitted = f"same-0-{candidate}" if round_number == 0 else f"{'attack' if attacked else 'clean'}-1-{candidate}"
                if is_admitted:
                    admitted.append(candidate)
                    inputs[candidate] = submitted
                rows.append(
                    {
                        "round": round_number,
                        "candidate_node_id": candidate,
                        "current_node": candidate,
                        "parent_global_model_sha256": parent,
                        "submitted_model_sha256": submitted,
                        "votes": [vote(candidate, validator, reference, attacked) for validator in validators],
                        "admitted": is_admitted,
                        "reached_aggregator_add_model": is_admitted,
                        "aggregation_input_sha256": submitted if is_admitted else None,
                        "aggregation_matches_submitted_snapshot": True if is_admitted else None,
                    }
                )
            rounds.append(
                {
                    "round": round_number,
                    "trainer_roles": dict(roles, byzantine_validators=["node-3", "node-4"] if attacked else []),
                    "aggregation_lineage": {
                        "contributors": admitted,
                        "input_hashes": inputs,
                        "installed_global_model_sha256": result,
                        "canonical_hash_source": "fixture",
                    },
                }
            )
            parent = result
        from brbfl.experiments.partition_evidence import canonical_partition_manifest

        partitions = canonical_partition_manifest(
            {
                "entries": [
                    {
                        "node_id": node,
                        "split": "train",
                        "partition_index": index,
                        "sample_count": 10,
                        "ordered_sample_indices_sha256": node,
                        "ordered_targets_sha256": f"labels-{node}",
                        "partitioning_strategy": "random_iid",
                        "dataset_identity_sha256": "mnist",
                        "configured_seed": 666,
                        "effective_worker_seed": 666,
                    }
                    for index, node in enumerate(contributors)
                ]
            }
        )
        return {
            "configuration": deepcopy(config),
            "seeds": {"experiment": 666, "partition": 666},
            "partitions": partitions,
            "provenance": {
                "evidence_schema_version": "brbfl.validation.v2",
                "producing_commit": "fixture",
                "controlled_configuration_sha256": "same",
                "dataset_identity": {"name": "MNIST"},
                "partitioning_strategy": "random_iid",
                "configured_seeds": {"experiment": 666, "partition": 666},
                "partition_manifest_sha256": partitions["sha256"],
            },
            "malicious_node_ids": ["node-3", "node-4"] if attacked else [],
            "validator_admission": rows,
            "rounds": rounds,
            "final_model_sha256": parent,
        }

    return build(False), build(True)


def test_causal_comparator_accepts_real_round_one_node_zero_shape():
    from brbfl.experiments.compare_byzantine_validator import compare_evidence

    clean, attacked = _comparison_fixture()
    result = compare_evidence(clean, attacked)
    assert result["first_changed_admission"] == {
        "round": 0,
        "candidate_node_id": "node-0",
        "clean_admitted": False,
        "attacked_admitted": True,
    }
    assert result["first_downstream_candidate_difference"]["candidate_node_id"] == "node-0"
    assert result["first_downstream_candidate_difference"]["expected_downstream_effect"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda clean, attacked: attacked["validator_admission"][2].update(submitted_model_sha256="early-difference"),
            "pre-intervention candidate/parent mismatch",
        ),
        (
            lambda clean, attacked: attacked["partitions"]["entries"][0].update(ordered_sample_indices_sha256="different"),
            "controlled partitions differ",
        ),
        (lambda clean, attacked: attacked["seeds"].update(experiment=777), "controlled seeds differ"),
        (
            lambda clean, attacked: attacked["rounds"][0]["trainer_roles"].update(actual_training_nodes=["node-0"]),
            "controlled trainer_roles differ",
        ),
        (
            lambda clean, attacked: attacked["validator_admission"][0]["votes"][0].update(reported_decision=True),
            "honest validator falsified",
        ),
        (
            lambda clean, attacked: attacked["validator_admission"][0]["votes"][0].update(reference_decision=True),
            "reference decision differs",
        ),
        (
            lambda clean, attacked: attacked["validator_admission"][3].update(parent_global_model_sha256="wrong"),
            "parent is not the prior installed",
        ),
        (
            lambda clean, attacked: attacked["validator_admission"][0].update(aggregation_input_sha256="wrong"),
            "aggregation-input hash differs",
        ),
    ],
)
def test_causal_comparator_rejects_broken_invariants(mutation, match):
    from brbfl.experiments.compare_byzantine_validator import compare_evidence

    clean, attacked = _comparison_fixture()
    mutation(clean, attacked)
    with pytest.raises(AssertionError, match=match):
        compare_evidence(clean, attacked)


def test_later_candidate_difference_requires_prior_global_divergence():
    from brbfl.experiments.compare_byzantine_validator import compare_evidence

    clean, attacked = _comparison_fixture()
    attacked["rounds"][0]["aggregation_lineage"]["installed_global_model_sha256"] = "clean-global-0"
    for row in attacked["validator_admission"]:
        if row["round"] == 1:
            row["parent_global_model_sha256"] = "clean-global-0"
    with pytest.raises(AssertionError, match="pre-intervention candidate/parent mismatch"):
        compare_evidence(clean, attacked)


def test_lineage_is_required():
    from brbfl.experiments.compare_byzantine_validator import compare_evidence

    clean, attacked = _comparison_fixture()
    del attacked["validator_admission"][0]["parent_global_model_sha256"]
    with pytest.raises(AssertionError, match="lacks required parent-model lineage"):
        compare_evidence(clean, attacked)
