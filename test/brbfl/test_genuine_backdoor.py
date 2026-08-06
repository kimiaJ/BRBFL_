"""Controlled genuine MNIST backdoor tests."""  # noqa: D100

# ruff: noqa: D103

import copy

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from brbfl.attacks.backdoor import BackdoorAttack
from brbfl.evaluation.metrics import MNISTTrigger, apply_mnist_trigger, triggered_asr_counts
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
    assert evidence["samples_examined"] == 10
    assert evidence["samples_poisoned"] == 3
    assert source._data["train"].to_dict() == snapshot
    for index in range(10):
        before = np.asarray(source._data["train"][index]["image"])
        after = np.asarray(poisoned._data["train"][index]["image"])
        if index in evidence["changed_image_indices"]:
            assert poisoned._data["train"][index]["label"] == 9
            assert np.all(after[25:28, 25:28] == 1.0)
        else:
            np.testing.assert_array_equal(after, before)
            assert poisoned._data["train"][index]["label"] == source._data["train"][index]["label"]


def test_benign_partition_is_unchanged_when_only_node_one_is_poisoned():
    benign = partition(4)
    snapshot = benign._data["train"].to_dict()
    malicious = BackdoorAttack(poison_rate=0.5).poison_data(partition(4))
    assert benign._data["train"].to_dict() == snapshot
    assert malicious.get_num_samples() == 4


def test_asr_excludes_target_and_is_exact_and_handles_zero_eligible():
    result = triggered_asr_counts(torch.tensor([2, 2, 1, 2]), torch.tensor([0, 2, 3, 4]), 2)
    assert result == {"triggered_test_target_prediction_count": 2, "eligible_triggered_examples": 3, "triggered_test_asr": 2 / 3}
    assert triggered_asr_counts(torch.tensor([2, 1]), torch.tensor([2, 2]), 2)["triggered_test_asr"] == 0.0


def test_comparison_reports_participants_and_poison_application():
    base = {
        "final_model_sha256": "clean",
        "malicious_node_ids": [],
        "per_node_poisoning_evidence": [{"node_id": f"node-{i}", "attack_application_count": 0} for i in range(3)],
        "rounds": [{"participating_node_ids": ["node-0", "node-1", "node-2"], "clean_test_accuracy": 0.5, "triggered_test_asr": 0.1}],
    }
    attacked = copy.deepcopy(base)
    attacked["final_model_sha256"] = "attacked"
    attacked["malicious_node_ids"] = ["node-1"]
    attacked["per_node_poisoning_evidence"][1]["attack_application_count"] = 1
    attacked["rounds"][0].update(clean_test_accuracy=0.4, triggered_test_asr=0.3)
    result = compare_evidence(base, attacked)
    assert result["final_models_differ"]
    assert result["participants_equal"]
    assert result["genuine_triggered_asr_differences_by_round"] == [0.19999999999999998]
