"""Regression tests for CA-4 causal artifacts."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from brbfl.experiments.compare_ca_state_transition import compare_evidence, generate_artifact, validate_artifact

CONFIGS = Path("configs/smoke")


def load(name: str):
    """Load a smoke fixture."""
    return yaml.safe_load((CONFIGS / name).read_text())


@pytest.fixture
def pair():
    """Return verified clean and attacked evidence."""
    return (
        generate_artifact(load("mnist_byzantine_validator_ca_clean.yaml")),
        generate_artifact(load("mnist_byzantine_validator_ca_attacked.yaml")),
    )


def test_causal_comparison_and_exact_paths(pair):
    """The paired run proves escalation, stability, and promotion semantics."""
    clean, attacked = pair
    result = compare_evidence(clean, attacked)
    assert result["verification_result"] is True
    assert result["causal_status"] == "proven_ca_state_transitions_excluded_byzantine_participants"
    assert result["attacked_state_paths"]["node-3"][:2] == ["suspicious", "excluded"]
    assert clean["rounds"][2]["next_ca_states"]["node-0"] == "trusted"
    assert clean["rounds"][2]["next_ca_states"]["node-1"] == "observation"


@pytest.mark.parametrize("path,value", [
    (("rounds", 0, "transition_records"), []),
    (("rounds", 0, "source_ca_hash"), "forged"),
    (("rounds", 0, "source_trust_hash"), "unverified"),
    (("rounds", 0, "role_assignment", "source_ca_snapshot_hash"), None),
    (("rounds", 0, "transition_records", 0, "reason_code"), "altered"),
])
def test_forged_evidence_is_rejected(pair, path, value):
    """Any missing or altered causal evidence invalidates the artifact."""
    artifact = copy.deepcopy(pair[1])
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        validate_artifact(artifact)


def test_deterministic_strict_json_ordering_and_no_mutation():
    """Generation is canonical, strict, and side-effect free."""
    config = load("mnist_byzantine_validator_ca_attacked.yaml")
    original = copy.deepcopy(config)
    first = generate_artifact(config)
    reordered = dict(reversed(list(config.items())))
    second = generate_artifact(reordered)
    assert config == original
    assert first["artifact_sha256"] == second["artifact_sha256"]
    json.dumps(compare_evidence(generate_artifact(load("mnist_byzantine_validator_ca_clean.yaml")), first), allow_nan=False)


def test_incompatible_control_and_missing_rotation_fail(pair):
    """Comparison fails closed on confounding fields and absent causality."""
    clean, attacked = pair
    incompatible = copy.deepcopy(attacked)
    incompatible["controlled_fields"]["seed"] = 1
    with pytest.raises(AssertionError):
        compare_evidence(clean, incompatible)
    with pytest.raises(AssertionError):
        compare_evidence(clean, clean)
