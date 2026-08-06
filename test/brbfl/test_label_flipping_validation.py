"""Controlled assertions for label-flipping evidence."""

from datasets import Dataset, DatasetDict

from brbfl.attacks import create_attack, prepare_dataset
from brbfl.experiments.attack_evidence import audit_partition, labels
from brbfl.experiments.config import load_experiment_config
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


def partition(values):
    """Build a minimal P2PFL partition."""
    return P2PFLDataset(DatasetDict({"train": Dataset.from_dict({"label": values}), "test": Dataset.from_dict({"label": values})}))


def test_only_malicious_partition_is_poisoned_once_without_mutating_source():
    """Poison only the malicious copy and preserve its independent source."""
    source = partition([1, 2, 1, 3])
    benign = partition([1, 2, 1, 3])
    original = labels(source)
    attack = create_attack("label_flipping", {"flip_map": {1: 7}})

    poisoned = prepare_dataset(source, attack)
    malicious_audit = audit_partition(1, original, labels(poisoned), {1: 7}, True)
    benign_audit = audit_partition(0, labels(benign), labels(benign), {1: 7}, False)

    assert malicious_audit["attack_application_count"] == 1
    assert malicious_audit["labels_examined"] == 4
    assert malicious_audit["labels_changed"] == 2
    assert labels(poisoned) == [7, 2, 7, 3]
    assert labels(benign) == original
    assert benign_audit["labels_changed"] == 0


def test_clean_configuration_changes_zero_labels():
    """A clean lifecycle leaves all labels untouched."""
    clean = partition([1, 2, 1])
    before = labels(clean)
    assert prepare_dataset(clean, create_attack("none")) is clean
    assert audit_partition(0, before, labels(clean), {}, False)["labels_changed"] == 0


def test_clean_and_attacked_configs_have_identical_controls():
    """The attack is the only experimental treatment."""
    clean = load_experiment_config("configs/smoke/mnist_clean.yaml")
    attacked = load_experiment_config("configs/smoke/mnist_label_flipping.yaml")
    controlled_fields = ("seed", "nodes", "rounds", "epochs", "protocol", "framework", "aggregator", "topology", "batch_size", "dataset")
    assert all(getattr(clean, field) == getattr(attacked, field) for field in controlled_fields)
    assert clean.output_dir != attacked.output_dir
    assert clean.attack.name == "none"
    assert attacked.attack.name == "label_flipping"
