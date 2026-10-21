"""Aggregate gossip must preserve verified lineage without local input rows."""

import numpy as np
import pytest

from brbfl.validation.byzantine_gate import (
    AdmissionPolicy,
    ValidatorSubgroupGate,
    canonical_aggregation_inputs,
    clear_validator_gate,
    get_validator_gate,
    install_validator_gate,
    parameter_hash,
    publish_admission_decision,
)


@pytest.fixture
def owners():
    """Publish decisions on separate contributor gates."""
    install_validator_gate(ValidatorSubgroupGate(AdmissionPolicy(("a", "b", "c"), ("v",), quorum=1, acceptance_threshold=1)))
    models = {node: [np.array([float(index)])] for index, node in enumerate(("a", "b", "c"))}
    for node, model in models.items():
        gate = get_validator_gate(node)
        publish_admission_decision(0, node, gate.submit_and_decide(0, node, model))
    yield models
    clear_validator_gate()


def test_aggregate_gossip_resolves_remote_owner_inputs(owners):
    """A receiver with only its own input still obtains all verified hashes."""
    for node, model in owners.items():
        get_validator_gate(node).observe_aggregation_input(0, node, model)
    receiver = get_validator_gate("c")
    assert len(receiver.evidence()) == 1
    inputs = canonical_aggregation_inputs(0, list(owners))
    assert inputs == {node: parameter_hash(model) for node, model in owners.items()}
    receiver.observe_round_result(0, [np.array([1.0])], list(owners), canonical_hash_source="test", aggregation_input_hashes=inputs)
    row = receiver.evidence()[0]
    assert row["round_aggregation_input_hashes"] == inputs
    assert len(receiver.evidence()) == 1


def test_admission_alone_is_not_input_evidence(owners):
    """An accepted but unobserved owner input cannot supply lineage."""
    with pytest.raises(RuntimeError, match="missing verified owner aggregation input"):
        canonical_aggregation_inputs(0, list(owners))


def test_aggregate_membership_must_equal_admission(owners):
    """Neither missing nor extra contributors may pass the guard."""
    for contributors in (["a", "b"], ["a", "b", "c", "x"]):
        with pytest.raises(RuntimeError, match="disagrees with admission"):
            canonical_aggregation_inputs(0, contributors)
