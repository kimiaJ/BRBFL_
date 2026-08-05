from datasets import Dataset, DatasetDict

from brbfl.experiments.config import DatasetConfig, ExperimentConfig
from brbfl.experiments.datasets import partition_dataset
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


def _toy_dataset():
    train = Dataset.from_dict({"image": list(range(12)), "label": [x % 3 for x in range(12)]})
    test = Dataset.from_dict({"image": list(range(6)), "label": [x % 3 for x in range(6)]})
    return P2PFLDataset(DatasetDict({"train": train, "test": test}), dataset_name="toy")


def _train_values(partitions):
    return [partition._data["train"]["image"] for partition in partitions]


def test_dataset_partitioning_is_deterministic_for_seed():
    config = ExperimentConfig(nodes=3, seed=123, dataset=DatasetConfig(name="toy", reduced=False))

    first = _train_values(partition_dataset(_toy_dataset(), config))
    second = _train_values(partition_dataset(_toy_dataset(), config))

    assert first == second


def test_dataset_partitioning_changes_with_seed():
    config_a = ExperimentConfig(nodes=3, seed=123, dataset=DatasetConfig(name="toy", reduced=False))
    config_b = ExperimentConfig(nodes=3, seed=456, dataset=DatasetConfig(name="toy", reduced=False))

    assert _train_values(partition_dataset(_toy_dataset(), config_a)) != _train_values(partition_dataset(_toy_dataset(), config_b))
