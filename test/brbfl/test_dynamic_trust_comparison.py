# ruff: noqa: D103
"""Focused regression tests for dynamic trust-ranked comparison evidence."""

import json
import math
from copy import deepcopy

import pytest

from brbfl.experiments.compare_byzantine_validator_trust import (
    _digest,
    canonicalize_json,
    compare_evidence,
    controlled_fields,
)
from brbfl.experiments.compare_dynamic_trust_selection import compare_dynamic

NODES = [f"node-{index}" for index in range(5)]
CANDIDATES = ["node-0", "node-1", "node-2"]
BOOTSTRAP = ["node-0", "node-3", "node-4"]


def _state(alpha, beta, votes, last):
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "score": alpha / (alpha + beta),
        "agreement_count": int(alpha - 1),
        "disagreement_count": int(beta - 1),
        "processed_vote_count": votes,
        "last_finalized_round": last,
    }


def _updates(round_number, validators, agreed):
    rows = []
    for validator in validators:
        for candidate in CANDIDATES:
            payload = {
                "round": round_number,
                "validator_id": validator,
                "candidate_id": candidate,
                "reported_decision": agreed[validator],
                "reference_decision": True,
            }
            rows.append({**payload, "agreed": agreed[validator], "evidence_sha256": _digest("TrustVote/v1", payload)})
    return rows


def _trust(attacked):
    prior = {node: _state(1, 1, 0, None) for node in NODES}
    round_zero_post = deepcopy(prior)
    for node in BOOTSTRAP:
        round_zero_post[node] = _state(4 if not attacked or node == "node-0" else 1, 1 if not attacked or node == "node-0" else 4, 3, 0)
    second_validators = ["node-0", "node-1", "node-2"] if attacked else BOOTSTRAP
    final = deepcopy(round_zero_post)
    for node in second_validators:
        before = final[node]
        final[node] = _state(before["alpha"] + 3, before["beta"], before["processed_vote_count"] + 3, 1)
    rounds = {
        "0": {
            "pre_round": prior,
            "updates": _updates(0, BOOTSTRAP, {node: not attacked or node == "node-0" for node in BOOTSTRAP}),
            "post_round": round_zero_post,
            "snapshot_sha256": "round-zero",
        },
        "1": {
            "pre_round": round_zero_post,
            "updates": _updates(1, second_validators, dict.fromkeys(second_validators, True)),
            "post_round": final,
            "snapshot_sha256": "round-one",
        },
    }
    return {
        "method": "beta_reputation",
        "prior": {"alpha": 1, "beta": 1},
        "rounds": rounds,
        "final_states": final,
        "verification_result": True,
        "verification_reason": "verified",
    }


def artifacts():
    def make(attacked):
        experiment_id = "mnist:666:byzantine_validator" if attacked else "mnist:666:none"
        assignment = {
            "experiment_id": experiment_id,
            "round_number": 0,
            "network_participants": NODES,
            "selected_contributors": CANDIDATES,
            "selected_validators": BOOTSTRAP,
            "aggregation_eligible_nodes": CANDIDATES,
            "detector_subgroups": {},
            "selection_source": "bootstrap",
            "previous_state_hash": None,
        }
        rotated = ["node-0", "node-1", "node-2"] if attacked else BOOTSTRAP
        selection = {
            "verification_result": True,
            "rounds": {
                "0": {"selected_validators": BOOTSTRAP},
                "1": {
                    "selected_validators": rotated,
                    "excluded_participants": ["node-3", "node-4"] if attacked else ["node-1", "node-2"],
                    "trust_source_round": 0,
                },
            },
        }
        return {
            "experiment_id": experiment_id,
            "configuration": {"seed": 666, "nodes": 5, "rounds": 2, "aggregator": "fedavg", "validation": {"quorum": 3}},
            "partitions": {"sha256": "partitions"},
            "initial_model_sha256": "initial",
            "rounds": [{"trainer_roles": {"submitted_candidates": CANDIDATES}} for _ in range(2)],
            "provenance": {"producing_commit": "b9ccae6", "controlled_configuration_sha256": "controlled"},
            "ledger": {
                "verification_result": True,
                "ledger_round_consensus": True,
                "per_round_role_assignment": {"0": assignment},
                "selection": selection,
            },
            "trust": _trust(attacked),
            "final_model_consensus": True,
        }

    return make(False), make(True)


def test_experiment_id_is_an_intervention_not_a_control():
    clean, attacked = artifacts()
    assert controlled_fields(clean) == controlled_fields(attacked)
    assert compare_dynamic(clean, attacked)["verification_result"] is True


@pytest.mark.parametrize(
    "field",
    [
        "round_number",
        "network_participants",
        "selected_contributors",
        "aggregation_eligible_nodes",
        "detector_subgroups",
        "selection_source",
        "previous_state_hash",
    ],
)
def test_every_bootstrap_role_field_remains_controlled(field):
    clean, attacked = artifacts()
    attacked["ledger"]["per_round_role_assignment"]["0"][field] = "different"
    with pytest.raises(AssertionError, match="controlled experiment"):
        compare_dynamic(clean, attacked)


def test_changed_bootstrap_validators_fail_compatibility():
    clean, attacked = artifacts()
    attacked["ledger"]["per_round_role_assignment"]["0"]["selected_validators"] = ["node-0"]
    with pytest.raises(AssertionError, match="controlled experiment"):
        compare_dynamic(clean, attacked)


@pytest.mark.parametrize("mutation", ["partitions", "seed", "validation", "candidates", "prior"])
def test_other_controlled_fields_remain_controlled(mutation):
    clean, attacked = artifacts()
    if mutation == "partitions":
        attacked["partitions"] = {"sha256": "other"}
    elif mutation == "seed":
        attacked["configuration"]["seed"] = 42
    elif mutation == "validation":
        attacked["configuration"]["validation"] = {"quorum": 2}
    elif mutation == "candidates":
        attacked["rounds"][1]["trainer_roles"]["submitted_candidates"] = ["node-0"]
    else:
        attacked["trust"]["prior"] = {"alpha": 2, "beta": 1}
    with pytest.raises(AssertionError, match="controlled"):
        compare_dynamic(clean, attacked)


def test_clean_unselected_nodes_may_remain_at_prior():
    clean, attacked = artifacts()
    result = compare_dynamic(clean, attacked)
    assert result["clean_final_states"]["node-1"]["score"] == result["clean_final_states"]["node-2"]["score"] == 0.5


def test_static_causal_behavior_is_unchanged():
    clean, attacked = artifacts()
    with pytest.raises(AssertionError, match="trust causal assertions failed"):
        compare_evidence(clean, attacked)


def test_dynamic_comparison_reports_exact_rotation_and_status():
    clean, attacked = artifacts()
    result = compare_dynamic(clean, attacked)
    assert result["selection_comparison"]["attacked"]["rounds"]["1"]["selected_validators"] == ["node-0", "node-1", "node-2"]
    assert result["role_assignment_comparison"] != "unchanged"
    assert result["causal_status"] == "proven_trust_based_selection_removed_low_trust_byzantine_validators"


def test_dynamic_comparison_rejects_missing_rotation():
    clean, attacked = artifacts()
    attacked["ledger"]["selection"]["rounds"]["1"]["selected_validators"] = BOOTSTRAP
    with pytest.raises(AssertionError, match="not removed"):
        compare_dynamic(clean, attacked)


@pytest.mark.parametrize("section", ["ledger", "trust"])
def test_dynamic_comparison_rejects_unverified_evidence(section):
    clean, attacked = artifacts()
    attacked[section]["verification_result"] = False
    with pytest.raises(AssertionError, match="unverified|verification failed"):
        compare_dynamic(clean, attacked)


def test_input_dictionary_order_does_not_change_hash():
    clean, attacked = artifacts()
    expected = compare_dynamic(clean, attacked)["comparison_sha256"]
    reordered_clean = dict(reversed(list(clean.items())))
    reordered_attacked = dict(reversed(list(attacked.items())))
    assert compare_dynamic(reordered_clean, reordered_attacked)["comparison_sha256"] == expected
    assert compare_dynamic(clean, attacked)["comparison_sha256"] == expected


def test_nonfinite_policy_bound_is_preserved_as_strict_json_and_hashes_deterministically():
    clean, attacked = artifacts()
    for artifact in (clean, attacked):
        artifact["configuration"]["validation"]["max_l2_norm"] = math.inf
    original_clean, original_attacked = deepcopy(clean), deepcopy(attacked)

    first = compare_dynamic(clean, attacked)
    second = compare_dynamic(dict(reversed(list(clean.items()))), dict(reversed(list(attacked.items()))))

    assert first["controlled_fields"]["validation"]["max_l2_norm"] == {
        "__nonfinite_float__": "positive_infinity"
    }
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert first["comparison_sha256"] == second["comparison_sha256"]
    assert clean == original_clean
    assert attacked == original_attacked


def test_canonicalization_distinguishes_nonfinite_values_and_preserves_finite_numbers_without_mutation():
    source = {
        "positive": math.inf,
        "negative": -math.inf,
        "not_a_number": math.nan,
        "finite_float": 1.25,
        "finite_int": 7,
        "nested": [True, None, "text"],
    }

    encoded = canonicalize_json(source)

    assert encoded["positive"] == {"__nonfinite_float__": "positive_infinity"}
    assert encoded["negative"] == {"__nonfinite_float__": "negative_infinity"}
    assert encoded["not_a_number"] == {"__nonfinite_float__": "nan"}
    assert encoded["finite_float"] == 1.25
    assert isinstance(encoded["finite_float"], float)
    assert encoded["finite_int"] == 7
    assert isinstance(encoded["finite_int"], int)
    assert math.isinf(source["positive"]) and source["positive"] > 0
    assert math.isinf(source["negative"]) and source["negative"] < 0
    assert math.isnan(source["not_a_number"])
    assert encoded is not source and encoded["nested"] is not source["nested"]


def test_digest_canonicalizes_equivalent_dictionary_order_without_mutating_input():
    first = {"z": math.inf, "a": {"number": 2.5}}
    second = {"a": {"number": 2.5}, "z": math.inf}

    assert _digest("ordering/v1", first) == _digest("ordering/v1", second)
    assert math.isinf(first["z"]) and math.isinf(second["z"])
