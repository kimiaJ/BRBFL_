"""Controlled assertions for sign-flipping validation evidence."""

import json

import numpy as np
import pytest

from brbfl.attacks import create_attack
from brbfl.experiments.compare_sign_flipping import compare
from brbfl.experiments.config import load_experiment_config
from brbfl.experiments.sign_flipping_evidence import AuditedModelUpdateAttack


def test_sign_flipping_is_exact_once_and_preserves_original():
    """Prove the configured formula on named tensors without input mutation."""
    update = [np.array([1.5, -2.0], dtype=np.float32), np.array([[3.0]], dtype=np.float32)]
    before = [value.copy() for value in update]
    attack = AuditedModelUpdateAttack(create_attack("sign_flipping", {"scale": -3.0}), ["weight", "bias"])

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

    sent = [attack.manipulate_update(update) for _recipient_or_retry in range(3)]
    # Even accidentally feeding an already attacked copy through the hook must
    # not turn -3 into 9 (or a later retry into -27).
    resent = attack.manipulate_update(sent[0])

    assert len(attack.events) == 1
    assert len(attack.transmissions) == 4
    assert all(np.array_equal(value, before_value) for value, before_value in zip(update, before, strict=True))
    assert all(np.array_equal(copy[0], -3.0 * before[0]) for copy in [*sent, resent])
    assert len({event["update_id"] for event in attack.transmissions}) == 1
    assert len({event["post_attack_sha256"] for event in attack.transmissions}) == 1


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
