"""Controlled genuine MNIST backdoor tests."""  # noqa: D100

# ruff: noqa: D103

import copy

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value, load_from_disk
from PIL import Image as PILImage

from brbfl.attacks import create_attack
from brbfl.attacks.backdoor import BackdoorAttack
from brbfl.evaluation.metrics import MNISTTrigger, apply_mnist_trigger, triggered_asr_counts
from brbfl.experiments.attack_evidence import audit_partition, snapshot_partition
from brbfl.experiments.compare_backdoor import compare_evidence
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


def partition(size: int = 10) -> P2PFLDataset:
    images = [np.full((28, 28), index / 20, dtype=np.float32) for index in range(size)]
    labels = [index % 10 for index in range(size)]
    return P2PFLDataset(
        DatasetDict(
            {"train": Dataset.from_dict({"image": images, "label": labels}), "test": Dataset.from_dict({"image": images, "label": labels})}
        )
    )


@pytest.mark.parametrize("reconstruct", [False, True])
def test_image_feature_poisoning_preserves_schema_and_source(tmp_path, reconstruct):
    images = [PILImage.fromarray(np.full((28, 28), index, dtype=np.uint8)) for index in range(10)]
    labels = list(range(10))
    features = Features({"image": Image(), "label": Value("int64")})
    train = Dataset.from_dict({"image": images, "label": labels}, features=features)
    if reconstruct:
        cache_path = tmp_path / "mnist"
        train.save_to_disk(cache_path)
        train = load_from_disk(cache_path)
    source = P2PFLDataset(DatasetDict({"train": train, "test": train}))
    before_images = [np.asarray(row["image"], dtype=np.uint8).copy() for row in train]
    before_labels = list(train["label"])

    attack = BackdoorAttack(poison_rate=0.3, target_class=2, trigger_value=255, seed=12)
    poisoned_train = attack.poison_data(source)._data["train"]
    selected = set(attack.poisoning_evidence["changed_image_indices"])

    assert isinstance(train.features["image"], Image)
    assert isinstance(poisoned_train.features["image"], Image)
    assert len(selected) == 3
    for index, row in enumerate(poisoned_train):
        assert isinstance(row["image"], PILImage.Image)
        assert row["image"].mode == "L"
        pixels = np.asarray(row["image"])
        assert pixels.dtype == np.uint8
        assert pixels.shape == (28, 28)
        if index in selected:
            assert np.all(pixels[25:28, 25:28] == 255)
            assert row["label"] == 2
        else:
            np.testing.assert_array_equal(pixels, before_images[index])
            assert row["label"] == before_labels[index]

    for index, row in enumerate(train):
        np.testing.assert_array_equal(np.asarray(row["image"]), before_images[index])
        assert row["label"] == before_labels[index]


def test_deterministic_trigger_placement_and_normalization():
    images = torch.zeros(2, 28, 28)
    trigger = MNISTTrigger(size=3, value=1.0, normalization_mean=0.5, normalization_std=0.5)
    result = apply_mnist_trigger(images, trigger)
    assert trigger.coordinates() == [[row, column] for row in (25, 26, 27) for column in (25, 26, 27)]
    assert torch.all(result[:, 25:28, 25:28] == 1.0)
    assert torch.count_nonzero(result).item() == 18
    assert torch.count_nonzero(images).item() == 0


def test_fraction_relabeling_source_copy_and_deterministic_evidence():
    source = partition()
    snapshot = copy.deepcopy(source._data["train"].to_dict())
    first = BackdoorAttack(poison_rate=0.3, target_class=9, seed=12)
    poisoned = first.poison_data(source)
    second = BackdoorAttack(poison_rate=0.3, target_class=9, seed=12)
    second.poison_data(partition())
    evidence = first.poisoning_evidence
    assert evidence == second.poisoning_evidence
    assert evidence["changed_image_indices"] == [1, 5, 7]
    assert evidence["samples_examined"] == 10
    assert evidence["samples_poisoned"] == 3
    assert source._data["train"].to_dict() == snapshot
    assert isinstance(poisoned._data["train"].features["image"], Sequence)
    for index in range(10):
        before = np.asarray(source._data["train"][index]["image"])
        encoded = poisoned._data["train"][index]["image"]
        assert isinstance(encoded, list)
        assert all(isinstance(row, list) for row in encoded)
        after = np.asarray(encoded, dtype=np.float32)
        assert after.shape == (28, 28)
        assert after.dtype == np.float32
        if index in evidence["changed_image_indices"]:
            assert poisoned._data["train"][index]["label"] == 9
            assert np.all(after[25:28, 25:28] == 1.0)
        else:
            np.testing.assert_array_equal(after, before)
            assert poisoned._data["train"][index]["label"] == source._data["train"][index]["label"]


def test_attack_aware_audit_handles_target_class_and_distinct_change_sets():
    source = partition()
    before = snapshot_partition(source)
    # This deterministic selection contains index 2, whose original label is already the target.
    attack = BackdoorAttack(poison_rate=0.5, target_class=2, seed=0)
    poisoned = attack.poison_data(source)
    evidence = audit_partition(node_id=1, attack_type="backdoor", before=before, after=poisoned, malicious=True, attack=attack)

    assert 2 in evidence["poisoned_indices"]
    assert 2 in evidence["image_changed_indices"]
    assert 2 not in evidence["label_changed_indices"]
    assert evidence["poisoned_indices"] == evidence["image_changed_indices"]
    assert set(evidence["label_changed_indices"]) < set(evidence["image_changed_indices"])
    assert evidence["resulting_labels_at_poisoned_indices"] == [2] * 5
    assert evidence["trigger_validation_passed"]
    assert evidence["non_trigger_pixels_preserved"]
    assert evidence["source_partition_unchanged"]
    assert source._data["train"][2]["label"] == 2


def _replace_train(dataset, transform):
    data = DatasetDict(dict(dataset._data))
    data[dataset._train_split_name] = data[dataset._train_split_name].map(transform, with_indices=True)
    return P2PFLDataset(data)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("label", "does not match all-to-one replacement|does not equal target label"),
        ("trigger", "missing or malformed trigger|image changes do not match poisoned indices"),
        ("outside", "outside the trigger"),
    ],
)
def test_backdoor_audit_rejects_semantically_invalid_poisoning(tamper, message):
    source = partition()
    before = snapshot_partition(source)
    attack = BackdoorAttack(poison_rate=0.4, target_class=9, seed=12)
    poisoned = attack.poison_data(source)
    selected = attack.poisoning_evidence["changed_image_indices"][0]

    def corrupt(row, index):
        if index != selected:
            return row
        changed = dict(row)
        if tamper == "label":
            changed["label"] = 8
        else:
            image = np.asarray(row["image"]).copy()
            image[27, 27] = 0 if tamper == "trigger" else image[27, 27]
            if tamper == "outside":
                image[0, 0] += 0.25
            changed["image"] = image
        return changed

    corrupted = _replace_train(poisoned, corrupt)
    with pytest.raises(AssertionError, match=message):
        audit_partition(node_id=1, attack_type="backdoor", before=before, after=corrupted, malicious=True, attack=attack)


def test_backdoor_audit_rejects_wrong_attack_dispatch():
    source = partition()
    before = snapshot_partition(source)
    backdoor = BackdoorAttack(poison_rate=0.3, seed=12)
    poisoned = backdoor.poison_data(source)
    label_attack = create_attack("label_flipping", {"flip_map": {1: 7}})
    with pytest.raises(AssertionError, match="changed indices do not match the configured label map"):
        audit_partition(
            node_id=1,
            attack_type="label_flipping",
            before=before,
            after=poisoned,
            malicious=True,
            attack=label_attack,
        )


def test_benign_partition_is_unchanged_when_only_node_one_is_poisoned():
    benign = partition(4)
    snapshot = benign._data["train"].to_dict()
    malicious = BackdoorAttack(poison_rate=0.5).poison_data(partition(4))
    assert benign._data["train"].to_dict() == snapshot
    assert malicious.get_num_samples() == 4


def test_asr_excludes_target_and_is_exact_and_handles_zero_eligible():
    result = triggered_asr_counts(torch.tensor([2, 2, 1, 2]), torch.tensor([0, 2, 3, 4]), 2)
    assert result == {"triggered_test_target_prediction_count": 2, "eligible_triggered_examples": 3, "triggered_test_asr": 2 / 3}
    assert triggered_asr_counts(torch.tensor([2, 1]), torch.tensor([2, 2]), 2) == {
        "triggered_test_target_prediction_count": 0,
        "eligible_triggered_examples": 0,
        "triggered_test_asr": None,
    }


def test_comparison_reports_participants_and_poison_application():
    base = {
        "final_model_sha256": "clean",
        "malicious_node_ids": [],
        "per_node_poisoning_evidence": [{"node_id": f"node-{i}", "attack_application_count": 0} for i in range(3)],
        "rounds": [
            {
                "participating_node_ids": ["node-0", "node-1", "node-2"],
                "malicious_participant_ids": [],
                "clean_test_accuracy": 0.5,
                "triggered_test_asr": 0.1,
                "triggered_test_target_prediction_count": 1,
                "eligible_triggered_examples": 10,
                "per_node_metrics": [
                    {"node_id": "node-0", "metric": "triggered_test_target_prediction_count", "value": 1},
                    {"node_id": "node-0", "metric": "eligible_triggered_examples", "value": 10},
                ],
            }
        ],
    }
    attacked = copy.deepcopy(base)
    attacked["final_model_sha256"] = "attacked"
    attacked["malicious_node_ids"] = ["node-1"]
    attacked["per_node_poisoning_evidence"][1]["attack_application_count"] = 1
    attacked["rounds"][0].update(
        clean_test_accuracy=0.4,
        malicious_participant_ids=["node-1"],
        triggered_test_asr=0.3,
        triggered_test_target_prediction_count=3,
    )
    attacked["rounds"][0]["per_node_metrics"][0]["value"] = 3
    result = compare_evidence(base, attacked)
    assert result["final_models_differ"]
    assert result["participants_equal"]
    assert result["genuine_triggered_asr_differences_by_round"] == [0.19999999999999998]
