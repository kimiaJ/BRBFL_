"""Focused tests for the real validator-subgroup admission decision."""
# ruff: noqa: D103

import numpy as np
import pytest

from brbfl.experiments.config import load_experiment_config
from brbfl.validation import AdmissionPolicy, ValidatorSubgroupGate


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
