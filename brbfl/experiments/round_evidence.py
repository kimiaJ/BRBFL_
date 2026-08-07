"""Construction and validation of authoritative round-level evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def malicious_participants(participants: list[str], configured_malicious: list[str]) -> list[str]:
    """Return configured malicious nodes which participated, in participant order."""
    malicious = set(configured_malicious)
    return [node_id for node_id in participants if node_id in malicious]


def triggered_round_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Micro-aggregate triggered counts and retain a diagnostic macro ASR."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"targets": 0, "eligible": 0})
    for row in metric_rows:
        metric = row["metric"]
        if metric not in {"triggered_test_target_prediction_count", "eligible_triggered_examples"}:
            continue
        value = row["value"]
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise AssertionError(f"{metric} must be a non-negative integer")
        key = "targets" if metric == "triggered_test_target_prediction_count" else "eligible"
        counts[str(row["node_id"])][key] += int(value)

    target_total = sum(item["targets"] for item in counts.values())
    eligible_total = sum(item["eligible"] for item in counts.values())
    if target_total > eligible_total:
        raise AssertionError("triggered target predictions exceed eligible examples")
    per_node_asr = [item["targets"] / item["eligible"] for item in counts.values() if item["eligible"]]
    return {
        "triggered_test_target_prediction_count": target_total,
        "eligible_triggered_examples": eligible_total,
        "triggered_test_asr": target_total / eligible_total if eligible_total else None,
        "triggered_test_asr_macro_average": sum(per_node_asr) / len(per_node_asr) if per_node_asr else None,
    }


def assert_round_evidence(round_evidence: dict[str, Any], configured_malicious: list[str]) -> None:
    """Assert participant and triggered-metric evidence is internally consistent."""
    expected = malicious_participants(round_evidence["participating_node_ids"], configured_malicious)
    if round_evidence["malicious_participant_ids"] != expected:
        raise AssertionError("malicious participants do not equal configured/participating intersection")
    totals = triggered_round_metrics(round_evidence["per_node_metrics"])
    for field in ("triggered_test_target_prediction_count", "eligible_triggered_examples"):
        value = round_evidence[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AssertionError(f"{field} must be a non-negative integer")
        if value != totals[field]:
            raise AssertionError(f"{field} does not equal the sum of per-node counts")
    if round_evidence["triggered_test_target_prediction_count"] > round_evidence["eligible_triggered_examples"]:
        raise AssertionError("triggered target predictions exceed eligible examples")
    actual = round_evidence["triggered_test_asr"]
    expected_asr = totals["triggered_test_asr"]
    if expected_asr is None:
        if actual is not None:
            raise AssertionError("ASR must be null when there are no eligible examples")
    elif actual is None or not math.isclose(actual, expected_asr, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("ASR does not equal target predictions / eligible examples")
