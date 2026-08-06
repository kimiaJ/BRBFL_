"""Lifecycle-level regression tests for sign-flipping publication."""

import numpy as np
import pytest

from brbfl.attacks import create_attack
from brbfl.experiments.sign_flipping_evidence import AuditedModelUpdateAttack, canonical_parameter_hash


def audit(round_provider=lambda: 0):
    """Build a one-parameter audited attack."""
    return AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=round_provider)


def test_training_completion_publishes_one_immutable_attack():
    """Publication freezes benign evidence and applies the formula once."""
    live = [np.array([2.0, -4.0], dtype=np.float32)]
    recorder = audit()
    recorder.trace("local_training_completed")
    attacked = recorder.publish_update(live)
    event = recorder.events[0]
    live[0][...] = 99

    assert len(recorder.events) == 1
    assert np.array_equal(attacked[0], [-6.0, 12.0])
    assert canonical_parameter_hash(recorder._original_snapshots[event["update_id"]], ["weight"]) == event["pre_attack_sha256"]
    assert event["pre_attack_sha256"] != event["post_attack_sha256"]
    assert event["post_attack_l2_norm"] == pytest.approx(3 * event["pre_attack_l2_norm"])
    assert event["cosine_similarity"] == pytest.approx(-1.0)
    assert [item["event_type"] for item in recorder.event_trace] == [
        "local_training_completed",
        "benign_snapshot_created",
        "attack_applied",
        "update_published",
    ]


def test_multiple_transmissions_are_observations_not_applications():
    """Repeated sends cannot execute logical attacks."""
    recorder = audit()
    attacked = recorder.publish_update([np.array([2.0], dtype=np.float32)])
    for recipient in ("node-0", "node-2", "node-0"):
        recorder.record_transmission(recipient, attacked)

    assert len(recorder.events) == 1
    assert len(recorder.transmissions) == 3
    assert all(item["matches_published_snapshot"] for item in recorder.transmissions)
    assert len({item["update_id"] for item in recorder.transmissions}) == 1


def test_aggregation_receives_cached_attacked_snapshot():
    """Aggregation must see the exact cached attacked payload."""
    recorder = audit()
    attacked = recorder.publish_update([np.array([2.0], dtype=np.float32)])
    recorder.observe_aggregation(attacked)

    assert recorder.validate_eligible_updates() == {recorder.events[0]["update_id"]: 1}
    assert recorder.aggregations[0]["payload_sha256"] == recorder.events[0]["post_attack_sha256"]
    with pytest.raises(AssertionError, match="aggregation-visible"):
        recorder.observe_aggregation([np.array([2.0], dtype=np.float32)])


def test_two_rounds_have_distinct_identity_and_one_application_each():
    """Each trained round creates a content-addressed publication."""
    current_round = 0
    recorder = audit(lambda: current_round)
    for round_id, value in enumerate((2.0, 5.0)):
        current_round = round_id
        attacked = recorder.publish_update([np.array([value], dtype=np.float32)])
        recorder.observe_aggregation(attacked)
        recorder.record_transmission("peer", attacked)

    assert [event["round_id"] for event in recorder.events] == ["0", "1"]
    assert len({event["pre_attack_sha256"] for event in recorder.events}) == 2
    assert [recorder.evidence_for_round(i)["logical_application_count"] for i in range(2)] == [1, 1]
    assert all(recorder.evidence_for_round(i)["aggregation_matches_attacked_snapshot"] for i in range(2))


def test_evolving_and_noncomparable_transport_is_not_falsely_grouped():
    """Only an equal full payload is a transmitted publication copy."""
    recorder = audit()
    attacked = recorder.publish_update([np.array([2.0], dtype=np.float32)])
    recorder.record_transmission("exact", attacked)
    recorder.record_transmission("aggregate", [np.array([7.0], dtype=np.float32)])
    recorder.record_transmission("metadata-only")

    exact, evolving, absent = recorder.transmissions
    assert exact["update_id"] == recorder.events[0]["update_id"]
    assert evolving["payload_comparable"] is True and evolving["update_id"] is None
    assert absent["payload_comparable"] is False and absent["payload_sha256"] is None


def test_canonical_hash_uses_names_dtype_shape_and_exact_bytes():
    """Canonical identity includes all model-value semantics."""
    first = {"b": np.array([1.0], dtype=np.float32), "a": np.array([[2.0]], dtype=np.float64)}
    reordered = {"a": first["a"].copy(), "b": first["b"].copy()}
    assert canonical_parameter_hash(first) == canonical_parameter_hash(reordered)
    assert canonical_parameter_hash(first) != canonical_parameter_hash({"a": first["a"].astype(np.float32), "b": first["b"]})
    assert canonical_parameter_hash(first) != canonical_parameter_hash({"a": first["a"].reshape(1), "b": first["b"]})


def test_second_publication_in_one_round_is_rejected_not_deduplicated():
    """A second completion in one round is a lifecycle violation."""
    recorder = audit()
    recorder.publish_update([np.array([1.0], dtype=np.float32)])
    with pytest.raises(AssertionError, match="already published"):
        recorder.publish_update([np.array([2.0], dtype=np.float32)])
