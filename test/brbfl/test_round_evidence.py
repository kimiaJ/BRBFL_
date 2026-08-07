"""Regression tests for round-level evidence aggregation."""

# ruff: noqa: D103

import copy

import pytest

from brbfl.experiments.compare_backdoor import compare_evidence
from brbfl.experiments.round_evidence import assert_round_evidence, malicious_participants, triggered_round_metrics


def rows(values):
    result = []
    for node_id, targets, eligible in values:
        result.extend(
            [
                {"node_id": node_id, "metric": "triggered_test_target_prediction_count", "value": targets},
                {"node_id": node_id, "metric": "eligible_triggered_examples", "value": eligible},
                {"node_id": node_id, "metric": "triggered_test_asr", "value": targets / eligible if eligible else 0.0},
            ]
        )
    return result


def test_malicious_participants_are_configuration_intersection():
    assert malicious_participants(["node-0", "node-1", "node-2"], ["node-1"]) == ["node-1"]
    assert malicious_participants(["node-0", "node-2"], ["node-1"]) == []
    assert malicious_participants(["node-0", "node-1"], []) == []


def test_unequal_denominators_use_summed_counts_and_expose_macro_average():
    result = triggered_round_metrics(rows([("node-1", 2, 57), ("node-2", 5, 59), ("node-0", 8, 59)]))
    assert result["triggered_test_target_prediction_count"] == 15
    assert result["eligible_triggered_examples"] == 175
    assert result["triggered_test_asr"] == 15 / 175
    assert result["triggered_test_asr_macro_average"] == pytest.approx((2 / 57 + 5 / 59 + 8 / 59) / 3)
    assert result["triggered_test_asr"] != result["triggered_test_asr_macro_average"]


def test_zero_eligible_examples_have_null_asr():
    assert triggered_round_metrics(rows([("node-0", 0, 0)])) == {
        "triggered_test_target_prediction_count": 0,
        "eligible_triggered_examples": 0,
        "triggered_test_asr": None,
        "triggered_test_asr_macro_average": None,
    }


def evidence(model, malicious, targets):
    metric_rows = rows([("node-0", targets, 10)])
    return {
        "final_model_sha256": model,
        "malicious_node_ids": malicious,
        "per_node_poisoning_evidence": [
            {"node_id": "node-0", "attack_application_count": 1 if malicious else 0},
            {"node_id": "node-1", "attack_application_count": 0},
        ],
        "rounds": [
            {
                "participating_node_ids": ["node-0", "node-1"],
                "malicious_participant_ids": malicious,
                "clean_test_accuracy": 0.5,
                "triggered_test_target_prediction_count": targets,
                "eligible_triggered_examples": 10,
                "triggered_test_asr": targets / 10,
                "per_node_metrics": metric_rows,
            }
        ],
    }


def test_comparison_uses_authoritative_micro_asr():
    clean = evidence("clean", [], 1)
    attacked = evidence("attacked", ["node-0"], 3)
    # compare_backdoor's fixed poisoning invariant is node-1.
    attacked["malicious_node_ids"] = ["node-1"]
    attacked["rounds"][0]["malicious_participant_ids"] = ["node-1"]
    attacked["per_node_poisoning_evidence"][0]["attack_application_count"] = 0
    attacked["per_node_poisoning_evidence"][1]["attack_application_count"] = 1
    assert compare_evidence(clean, attacked)["genuine_triggered_asr_differences_by_round"] == pytest.approx([0.2])


def test_consistency_assertions_reject_averaged_count_fields():
    round_data = evidence("clean", [], 1)["rounds"][0]
    broken = copy.deepcopy(round_data)
    broken["eligible_triggered_examples"] = 5
    with pytest.raises(AssertionError, match="sum of per-node counts"):
        assert_round_evidence(broken, [])
