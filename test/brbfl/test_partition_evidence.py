"""Canonical controlled-partition evidence tests."""
# ruff: noqa: D103

from __future__ import annotations

from copy import deepcopy

import pytest

from brbfl.experiments.config import ExperimentConfig
from brbfl.experiments.partition_evidence import canonical_partition_manifest


def _manifest() -> dict:
    rows = []
    for node, digest in (("node-0", "aaa"), ("node-1", "bbb")):
        for split in ("train", "test"):
            rows.append(
                {
                    "node_id": node,
                    "split": split,
                    "partition_index": int(node[-1]),
                    "sample_count": 4,
                    "ordered_sample_indices_sha256": digest + split,
                    "ordered_targets_sha256": "labels-" + digest + split,
                    "partitioning_strategy": "random_iid",
                    "dataset_identity_sha256": "mnist-v1",
                    "configured_seed": 666,
                    "effective_worker_seed": 666,
                }
            )
    return {"entries": rows}


def test_order_and_unstable_runtime_metadata_do_not_change_identity():
    left = _manifest()
    right = deepcopy(left)
    right["entries"].reverse()
    right["entries"] = [dict(reversed(list(row.items())), actor_id="ray-volatile", timestamp="now") for row in right["entries"]]
    assert canonical_partition_manifest(left) == canonical_partition_manifest(right)


@pytest.mark.parametrize(
    "field,value", [("ordered_sample_indices_sha256", "changed"), ("configured_seed", 777), ("dataset_identity_sha256", "other")]
)
def test_assignment_seed_and_dataset_changes_fail_identity(field, value):
    left, right = _manifest(), _manifest()
    right["entries"][0][field] = value
    assert canonical_partition_manifest(left) != canonical_partition_manifest(right)


def test_swapped_node_partitions_fail_even_when_counts_match():
    left, right = _manifest(), _manifest()
    for row in right["entries"]:
        row["node_id"] = "node-1" if row["node_id"] == "node-0" else "node-0"
    assert canonical_partition_manifest(left) != canonical_partition_manifest(right)


def test_duplicate_node_local_evidence_cannot_overwrite_identity():
    value = _manifest()
    value["entries"].append(deepcopy(value["entries"][0]))
    with pytest.raises(AssertionError, match="duplicate partition record cannot overwrite"):
        canonical_partition_manifest(value)


def test_stale_schema_fails_descriptively():
    with pytest.raises(AssertionError, match="canonical v2 manifest"):
        canonical_partition_manifest([{"node_id": "node-0", "samples_examined": 4}])


def test_real_mnist_partition_construction_path_is_repeatable_and_startup_order_independent():
    datasets = pytest.importorskip("datasets")
    from brbfl.experiments.datasets import partition_dataset
    from brbfl.experiments.partition_evidence import build_partition_manifest, dataset_identity
    from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

    train = datasets.Dataset.from_dict({"image": list(range(20)), "label": [index % 10 for index in range(20)]})
    test = datasets.Dataset.from_dict({"image": list(range(10)), "label": list(range(10))})

    def construct():
        data = P2PFLDataset(datasets.DatasetDict({"train": train, "test": test}), dataset_name="p2pfl/MNIST")
        config = ExperimentConfig(nodes=2, seed=666)
        partitions = partition_dataset(data, config)
        return build_partition_manifest(partitions, config, dataset_identity(data, config.dataset.name))

    first, second = construct(), construct()
    simulated_worker_startup_order = ["node-1", "node-0"]
    assert simulated_worker_startup_order != ["node-0", "node-1"]
    assert first == second
