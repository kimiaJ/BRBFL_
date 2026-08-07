"""Focused coordinated-collusion tests."""
# ruff: noqa: D103

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from brbfl.attacks.collusion import CollusionAttack
from brbfl.attacks.registry import create_attack
from brbfl.experiments.attack_evidence import audit_partition, snapshot_partition
from brbfl.experiments.collusion_evidence import CollusionLifecycleAudit, completed_collusion_rows, cosine, delta
from brbfl.experiments.config import load_experiment_config
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


def attack():
    return CollusionAttack("g", [1, 2], 42, 1.0)


def partition():
    rows = {"image": [np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)], "label": [1, 2]}
    return P2PFLDataset(DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)}))


def test_collusion_partition_dispatch_preserves_all_data_for_attacked_and_clean_nodes():
    nodes = [partition() for _ in range(3)]
    snapshots = [snapshot_partition(node) for node in nodes]

    attacked = [
        audit_partition(
            node_id=i,
            attack_type="collusion",
            before=snapshots[i],
            after=nodes[i],
            malicious=i in {1, 2},
            attack=attack() if i in {1, 2} else None,
        )
        for i in range(3)
    ]
    clean = [audit_partition(node_id=i, attack_type="none", before=snapshots[i], after=nodes[i], malicious=False) for i in range(3)]

    for index in (1, 2):
        evidence = attacked[index]
        assert evidence["source_sample_count"] == evidence["result_sample_count"] == 2
        assert evidence["before_image_sha256"] == evidence["after_image_sha256"]
        assert evidence["before_label_sha256"] == evidence["after_label_sha256"]
        assert evidence["source_partition_sha256"] == evidence["result_partition_sha256"]
        assert evidence["poisoned_image_count"] == evidence["poisoned_label_count"] == 0
        assert evidence["samples_poisoned"] == 0
        assert evidence["backdoor_trigger_applied"] is evidence["label_flipping_applied"] is False
    assert attacked[0]["source_partition_sha256"] == attacked[0]["result_partition_sha256"]
    assert attacked[0]["image_changed_indices"] == attacked[0]["label_changed_indices"] == []
    assert [row["result_partition_sha256"] for row in clean] == [row["result_partition_sha256"] for row in attacked]


def test_unknown_attack_does_not_inherit_model_poisoning_partition_validation():
    node = partition()
    with pytest.raises(AssertionError, match="node-1 has no partition audit validator for attack type 'unknown_model_attack'"):
        audit_partition(
            node_id=1,
            attack_type="unknown_model_attack",
            before=snapshot_partition(node),
            after=node,
            malicious=True,
        )


@pytest.mark.parametrize("attack_type", ["sign_flipping", "free_rider"])
def test_existing_model_poisoning_partition_dispatch_remains_unchanged(attack_type):
    node = partition()
    evidence = audit_partition(node_id=1, attack_type=attack_type, before=snapshot_partition(node), after=node, malicious=True)
    assert evidence["samples_poisoned"] == evidence["attack_application_count"] == 0
    assert evidence["source_partition_unchanged"] is True


def test_configuration_and_registry():
    clean = load_experiment_config("configs/smoke/mnist_collusion_clean.yaml")
    attacked = load_experiment_config("configs/smoke/mnist_collusion.yaml")
    assert clean.attack.name == "none" and clean.attack.adversaries == ()
    assert attacked.attack.adversaries == (1, 2)
    assert isinstance(create_attack(attacked.attack.name, attacked.attack.parameters), CollusionAttack)
    assert clean.nodes == attacked.nodes == 3 and clean.rounds == attacked.rounds == 2
    assert clean.seed == attacked.seed and clean.dataset == attacked.dataset


def test_shared_direction_is_deterministic_shared_and_round_specific():
    values = [np.zeros((2, 3), dtype=np.float32)]
    a, b = attack(), attack()
    assert all(np.array_equal(x, y) for x, y in zip(a.shared_direction(values, 0), b.shared_direction(values, 0), strict=True))
    assert not np.array_equal(a.shared_direction(values, 0)[0], a.shared_direction(values, 1)[0])


def complete(audit, before, trained):
    audit.begin_local_training(0, before)
    audit.record_optimizer_steps(2)
    audit.complete_local_training(trained, False)
    submitted = audit.publish_update(trained)
    audit.record_submission(submitted, 0)
    audit.observe_aggregation(submitted)
    audit.observe_global_model(submitted)
    return submitted


def test_genuine_training_then_aligned_poisoning_and_receipt():
    before = [np.array([1.0, 2.0], dtype=np.float32)]
    left = CollusionLifecycleAudit(attack(), "node-1", 1, 2)
    right = CollusionLifecycleAudit(attack(), "node-2", 1, 2)
    ls = complete(left, before, [np.array([2.0, 2.5], dtype=np.float32)])
    rs = complete(right, before, [np.array([0.5, 4.0], dtype=np.float32)])
    assert cosine(delta(before, ls), delta(before, rs)) == pytest.approx(1.0, abs=1e-6)
    for audit in (left, right):
        row = audit.evidence_for_round(0)
        assert row["optimizer_step_count"] == 2 and row["attack_application_count"] == 1
        assert row["submitted_update_l2_norm"] == pytest.approx(row["genuine_update_l2_norm"], rel=1e-6)
        assert row["aggregation_receipt"] and row["submitted_shared_direction_cosine_similarity"] == pytest.approx(1, abs=1e-6)


def test_zero_update_cosine_and_immutable_snapshot():
    assert cosine([np.zeros(2)], [np.ones(2)]) == 0.0
    before = [np.ones(2, dtype=np.float32)]
    audit = CollusionLifecycleAudit(attack(), "node-1", 1, 1)
    submitted = complete(audit, before, before)
    digest = audit.evidence_for_round(0)["submitted_model_sha256"]
    submitted[0][0] = 99
    assert audit.evidence_for_round(0)["submitted_model_sha256"] == digest
    assert audit.evidence_for_round(0)["submitted_update_l2_norm"] == 0


@pytest.mark.parametrize("alpha", [-1, float("nan"), float("inf")])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CollusionAttack("g", [1, 2], 1, alpha)


def test_invalid_groups_rejected():
    with pytest.raises(ValueError, match="group_id"):
        CollusionAttack("", [1, 2], 1)
    with pytest.raises(ValueError, match="at least two"):
        CollusionAttack("g", [1], 1)


def test_benign_never_applies_attack():
    audit = CollusionLifecycleAudit(None, "node-0", 1, 1)
    complete(audit, [np.zeros(1)], [np.ones(1)])
    assert audit.evidence_for_round(0)["attack_application_count"] == 0


def test_completed_rows_distinguish_configured_participating_and_completed():
    before = [np.zeros(2, dtype=np.float32)]
    left = CollusionLifecycleAudit(attack(), "node-1", 1, 1)
    complete(left, before, [np.ones(2, dtype=np.float32)])

    rows, participating, missing = completed_collusion_rows({"node-1": left}, ["node-1", "node-2"], ["node-0", "node-1"], 0)

    assert list(rows) == participating == ["node-1"]
    assert missing == ["node-2"]
    assert rows["node-1"]["shared_direction_sha256"]


@pytest.mark.parametrize(
    ("finish", "missing_key", "message"),
    [
        (False, None, "finalized=False"),
        (True, "shared_direction_sha256", "shared_direction_sha256"),
    ],
)
def test_participating_colluder_incomplete_evidence_fails_descriptively(finish, missing_key, message):
    audit = CollusionLifecycleAudit(attack(), "node-1", 1, 1)
    audit.begin_local_training(0, [np.zeros(1)])
    if finish:
        audit.record_optimizer_steps(1)
        audit.complete_local_training([np.ones(1)], False)
        submitted = audit.publish_update([np.ones(1)])
        audit.record_submission(submitted, 0)
        audit.observe_aggregation(submitted)
        audit.observe_global_model(submitted)
        del audit.rounds[0][missing_key]

    with pytest.raises(RuntimeError, match=message) as error:
        completed_collusion_rows({"node-1": audit}, ["node-1"], ["node-1"], 0)
    text = str(error.value)
    assert "node=node-1, round=0" in text
    assert "participant=True, completed=False" in text
    assert "available_keys=" in text and "attack_application_count=" in text


def test_clean_and_benign_rows_need_no_collusion_fields():
    benign = CollusionLifecycleAudit(None, "node-0", 1, 1)
    complete(benign, [np.zeros(1)], [np.ones(1)])
    row = benign.evidence_for_round(0)
    assert "shared_direction_sha256" not in row
    assert completed_collusion_rows({"node-0": benign}, [], ["node-0"], 0) == ({}, [], [])
