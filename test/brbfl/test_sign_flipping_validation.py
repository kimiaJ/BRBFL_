"""Controlled assertions for sign-flipping validation evidence."""

import json
import threading

import numpy as np
import pytest

from brbfl.attacks import create_attack
from brbfl.experiments.compare_sign_flipping import compare
from brbfl.experiments.config import load_experiment_config
from brbfl.experiments.sign_flipping_evidence import AuditedModelUpdateAttack, canonical_parameter_hash


def test_sign_flipping_is_exact_once_and_preserves_original():
    """Prove the configured formula on named tensors without input mutation."""
    update = [np.array([1.5, -2.0], dtype=np.float32), np.array([[3.0]], dtype=np.float32)]
    before = [value.copy() for value in update]
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight", "bias"])
    attack.record_update_created(update)

    transformed = attack.manipulate_update(update)

    assert len(attack.events) == 1
    assert all(np.array_equal(value, saved) for value, saved in zip(update, before, strict=True))
    assert all(np.array_equal(value, -3.0 * saved) for value, saved in zip(transformed, before, strict=True))
    event = attack.events[0]
    assert event["cosine_similarity"] == pytest.approx(-1.0)
    assert event["maximum_transformation_error"] <= event["numerical_tolerance"]
    assert event["original_pre_attack_update_preserved"] is True
    assert event["post_attack_l2_norm"] == pytest.approx(3 * event["pre_attack_l2_norm"])
    assert [item["name"] for item in event["parameters"]] == ["weight", "bias"]


def test_one_update_can_be_transmitted_repeatedly_without_reapplying_attack():
    """Recipient sends and retries reuse one attacked update byte-for-byte."""
    update = [np.array([2.0, -4.0], dtype=np.float32)]
    before = [value.copy() for value in update]
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    attack.record_update_created(update)

    sent = []
    for recipient in ("node-0", "node-2", "node-0"):
        sent.append(attack.manipulate_update(update))
        attack.record_transmission(recipient)
    # Even accidentally feeding an already attacked copy through the hook must
    # not turn -3 into 9 (or a later retry into -27).
    resent = attack.manipulate_update(sent[0])
    attack.record_transmission("node-2")

    assert len(attack.events) == 1
    assert len(attack.transmissions) == 4
    assert all(np.array_equal(value, before_value) for value, before_value in zip(update, before, strict=True))
    assert all(np.array_equal(copy[0], -3.0 * before[0]) for copy in [*sent, resent])
    assert len({event["update_id"] for event in attack.transmissions}) == 1
    assert len({event["post_attack_sha256"] for event in attack.transmissions}) == 1
    assert attack.evidence_for_round(0)["logical_application_count"] == 1
    assert attack.evidence_for_round(0)["transmission_count"] == 4
    assert len({item["logical_application_id"] for item in attack.transmissions}) == 1
    assert len({item["hook_invocation_id"] for item in attack.transmissions}) == 4
    assert all(item["cached_result_reused"] for item in attack.transmissions)
    assert not any(item["transformation_newly_applied"] for item in attack.transmissions)
    assert all(item["lifecycle_stage"] == "transmission" for item in attack.transmissions)


def test_canonical_parameter_hash_ignores_mapping_and_transport_metadata_order():
    """Only sorted, named parameter content participates in the digest."""
    weight = np.array([1.0, -2.0], dtype=np.float32)
    bias = np.array([3.0], dtype=np.float64)
    first = {"weight": weight, "bias": bias}
    second = {"bias": bias.copy(), "weight": weight.copy()}

    assert canonical_parameter_hash(first) == canonical_parameter_hash(second)
    assert canonical_parameter_hash(first) == canonical_parameter_hash(first, names=["ignored", "metadata"])


def test_transmission_copies_and_cached_snapshot_are_independent():
    """Mutating one recipient's payload cannot alter another send or the cache."""
    update = [np.array([2.0, -4.0], dtype=np.float32)]
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    attack.record_update_created(update)

    first = attack.manipulate_update(update)
    attack.record_transmission("node-0")
    first[0][0] = 999.0
    second = attack.manipulate_update(update)
    attack.record_transmission("node-2")

    assert np.array_equal(second[0], np.array([-6.0, 12.0], dtype=np.float32))
    assert np.array_equal(attack._cache[attack.events[0]["update_id"]][0], second[0])
    assert len({item["post_attack_sha256"] for item in attack.transmissions}) == 1


def test_live_model_mutation_after_snapshot_does_not_change_logical_update():
    """A trained-model mutation cannot rewrite the immutable audit snapshot."""
    live = [np.array([2.0, -4.0], dtype=np.float32)]
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    attack.record_update_created(live)
    update_id = attack.events[0]["update_id"]
    pre_hash = attack.events[0]["pre_attack_sha256"]

    live[0][...] = 100.0
    sent = attack.manipulate_update(attack._original_snapshots[update_id])
    attack.record_transmission("node-0")

    assert canonical_parameter_hash(attack._original_snapshots[update_id], ["weight"]) == pre_hash
    assert np.array_equal(sent[0], np.array([-6.0, 12.0], dtype=np.float32))


def test_snapshot_and_cached_transform_do_not_alias_source_or_each_other():
    """Storage diagnostics prove shallow-copy aliasing is not present."""
    source = [np.array([1.0, 2.0], dtype=np.float32)]
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    attack.record_update_created(source)
    event = attack.events[0]

    assert event["original_storage"][0]["owns_storage"] is True
    assert event["original_storage"][0]["storage_address"] != event["transformed_storage"][0]["storage_address"]
    assert not np.shares_memory(attack._original_snapshots[event["update_id"]][0], attack._cache[event["update_id"]][0])


def test_recipient_metadata_is_not_part_of_canonical_hash():
    """Transport envelopes never enter canonical parameter hashing."""
    params = {"weight": np.array([1.0], dtype=np.float32)}
    envelopes = [{"recipient": "node-0", "timestamp": 1}, {"recipient": "node-2", "timestamp": 999}]

    assert canonical_parameter_hash(params) == canonical_parameter_hash(params)
    assert envelopes[0] != envelopes[1]


def test_already_attacked_payload_keeps_original_pre_attack_evidence():
    """A retry's attacked bytes are recognized, never relabeled as original."""
    original = [np.array([2.0], dtype=np.float32)]
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    attack.record_update_created(original)
    attacked = attack.manipulate_update(original)
    attack.record_transmission("node-0")
    attack.manipulate_update(attacked)
    attack.record_transmission("node-2")

    assert {item["pre_attack_sha256"] for item in attack.transmissions} == {attack.events[0]["pre_attack_sha256"]}
    assert attack.events[0]["pre_attack_sha256"] != canonical_parameter_hash(attacked, ["weight"])


def test_concurrent_transmission_keeps_its_own_hook_observation():
    """A concurrent aggregate serialization cannot relabel a local-update send."""
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    local = [np.array([2.0], dtype=np.float32)]
    attack.record_update_created(local)
    local_hook_finished = threading.Event()
    aggregate_hook_finished = threading.Event()

    def send_local():
        attack.manipulate_update(local)
        local_hook_finished.set()
        aggregate_hook_finished.wait()
        attack.record_transmission("node-0")

    def send_aggregate():
        local_hook_finished.wait()
        attack.manipulate_update([np.array([7.0], dtype=np.float32)])
        aggregate_hook_finished.set()
        attack.record_transmission("node-2")

    threads = [threading.Thread(target=send_local), threading.Thread(target=send_aggregate)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    local_transmission = next(item for item in attack.transmissions if item["recipient"] == "node-0")
    aggregate_transmission = next(item for item in attack.transmissions if item["recipient"] == "node-2")
    assert local_transmission["update_id"] == attack.events[0]["update_id"]
    assert local_transmission["post_attack_sha256"] == attack.events[0]["post_attack_sha256"]
    assert aggregate_transmission["update_id"] is None


def test_different_pre_attack_updates_have_different_logical_ids():
    """Two local states in one producer/round cannot be grouped by metadata."""
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    first = [np.array([1.0], dtype=np.float32)]
    second = [np.array([2.0], dtype=np.float32)]
    attack.record_update_created(first)
    attack.record_update_created(second)
    attack.manipulate_update(first)
    attack.record_transmission("node-0")
    attack.manipulate_update(second)
    attack.record_transmission("node-2")

    assert len(attack.events) == 2
    assert len({event["pre_attack_sha256"] for event in attack.events}) == 2
    assert len({event["update_id"] for event in attack.events}) == 2


def test_distinct_round_updates_each_have_one_logical_application():
    """Round identity prevents equal metadata or reused objects suppressing round two."""
    current_round = 0
    attack = AuditedModelUpdateAttack(
        create_attack("sign_flipping", {"scale": -3.0}),
        ["weight"],
        round_provider=lambda: current_round,
    )
    update = [np.array([1.0, -2.0], dtype=np.float32)]
    originals = []
    transmitted = []

    for round_id in range(2):
        current_round = round_id
        # Deliberately reuse the list/array identities and identical metadata.
        update[0][...] = np.array([round_id + 1.0, -(round_id + 2.0)], dtype=np.float32)
        originals.append(update[0].copy())
        attack.record_update_created(update)
        first = attack.manipulate_update(update)
        attack.record_transmission("node-0")
        retry = attack.manipulate_update(update)
        attack.record_transmission("node-2")
        transmitted.append((first, retry))

    assert [event["round_id"] for event in attack.events] == ["0", "1"]
    assert len({event["update_id"] for event in attack.events}) == 2
    for round_id, ((first, retry), original) in enumerate(zip(transmitted, originals, strict=True)):
        evidence = attack.evidence_for_round(round_id)
        round_transmissions = [item for item in attack.transmissions if item["round_id"] == str(round_id)]
        assert evidence["logical_application_count"] == 1
        assert evidence["transmission_count"] == 2
        assert evidence["pre_attack_sha256"] != evidence["post_attack_sha256"]
        assert evidence["post_attack_l2_norm"] == pytest.approx(3 * evidence["pre_attack_l2_norm"])
        assert evidence["cosine_similarity"] == pytest.approx(-1.0)
        assert evidence["maximum_transformation_error"] <= evidence["numerical_tolerance"]
        assert evidence["original_pre_attack_update_preserved"] is True
        assert np.array_equal(first[0], -3.0 * original)
        assert np.array_equal(retry[0], first[0])
        assert len({item["post_attack_sha256"] for item in round_transmissions}) == 1
        assert round_transmissions[-1]["transmission_count"] == 2


def test_prior_round_attacked_bytes_are_a_new_update_in_the_next_round():
    """The retry index cannot collide across rounds, even when update bytes do."""
    current_round = 0
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: current_round)
    attack.record_update_created([np.array([2.0], dtype=np.float32)])
    first = attack.manipulate_update([np.array([2.0], dtype=np.float32)])

    current_round = 1
    attack.record_update_created(first)
    second = attack.manipulate_update(first)

    assert len(attack.events) == 2
    assert [attack.evidence_for_round(round_id)["logical_application_count"] for round_id in range(2)] == [1, 1]
    assert np.array_equal(first[0], np.array([-6.0], dtype=np.float32))
    assert np.array_equal(second[0], np.array([18.0], dtype=np.float32))


def test_partial_aggregate_hook_observations_are_not_eligible_updates():
    """Changing aggregate payloads explain raw hook counts without extra attacks."""
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 0)
    local = [np.array([2.0], dtype=np.float32)]
    attack.record_update_created(local)

    attacked = attack.manipulate_update(local)
    attack.record_transmission("node-0")
    partial_one = attack.manipulate_update([np.array([5.0], dtype=np.float32)])
    attack.record_transmission("node-2")
    partial_two = attack.manipulate_update([np.array([7.0], dtype=np.float32)])
    attack.record_transmission("node-0")

    assert len(attack.hook_invocations) == 3
    assert len(attack.transmissions) == 3
    assert len(attack.events) == 1
    assert attack.validate_eligible_updates() == {attack.events[0]["update_id"]: 1}
    assert np.array_equal(attacked[0], np.array([-6.0], dtype=np.float32))
    assert np.array_equal(partial_one[0], np.array([5.0], dtype=np.float32))
    assert np.array_equal(partial_two[0], np.array([7.0], dtype=np.float32))


def test_terminal_round_without_outbound_update_is_not_eligible():
    """A configured terminal lifecycle event does not fabricate an attack."""
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 1)
    attack.trace("round_entered", participating=True)
    attack.trace("local_training_completed")
    attack.trace("evaluation_completed")

    assert attack.eligible_round_ids() == set()
    assert attack.validate_eligible_updates() == {}
    assert attack.events == []


def test_participant_absence_is_not_an_eligible_update():
    """A malicious node omitted by voting has no expected transformation."""
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 1)
    attack.trace("round_entered", participating=False)

    assert attack.eligible_round_ids() == set()
    assert attack.validate_eligible_updates() == {}


def test_eligible_transmission_that_bypasses_hook_fails_validation():
    """Observed training plus serialization cannot silently evade the attack."""
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight"], round_provider=lambda: 1)
    attack.record_update_created([np.array([1.0], dtype=np.float32)])
    attack.trace("local_training_completed")
    attack.record_transmission("node-0")

    with pytest.raises(AssertionError, match="inconsistent pre-attack hashes"):
        attack.validate_eligible_updates()


def test_sign_flipping_configs_match_controls():
    """The attack and isolated output directory are the only treatments."""
    clean = load_experiment_config("configs/smoke/mnist_sign_flipping_clean.yaml")
    attacked = load_experiment_config("configs/smoke/mnist_sign_flipping.yaml")
    controls = ("seed", "nodes", "rounds", "epochs", "protocol", "framework", "aggregator", "topology", "batch_size", "dataset")
    assert all(getattr(clean, field) == getattr(attacked, field) for field in controls)
    assert clean.output_dir != attacked.output_dir
    assert clean.attack.name == "none"
    assert attacked.attack.adversaries == (1,)


def test_comparison_rejects_attack_on_benign_node(tmp_path):
    """Comparison enforces participants and per-round application counts."""
    round_data = {
        "round": 0,
        "participating_node_ids": ["node-0", "node-1", "node-2"],
        "malicious_participant_ids": [],
        "attack_application_counts": {"node-0": 0, "node-1": 0, "node-2": 0},
        "model_update_transformations": {},
        "per_node_metrics": [],
    }
    clean = {"malicious_node_ids": [], "rounds": [round_data], "final_model_sha256": "clean"}
    attacked_round = dict(
        round_data, malicious_participant_ids=["node-1"], attack_application_counts={"node-0": 1, "node-1": 0, "node-2": 0}
    )
    attacked = {"malicious_node_ids": ["node-1"], "rounds": [attacked_round], "final_model_sha256": "attacked"}
    clean_path, attacked_path = tmp_path / "clean.json", tmp_path / "attacked.json"
    clean_path.write_text(json.dumps(clean), encoding="utf-8")
    attacked_path.write_text(json.dumps(attacked), encoding="utf-8")

    try:
        compare(clean_path, attacked_path, tmp_path / "comparison.json")
    except AssertionError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("invalid benign-node transformation was accepted")
