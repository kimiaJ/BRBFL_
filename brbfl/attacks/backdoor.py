"""Deterministic all-to-one MNIST backdoor poisoning."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict

from brbfl.evaluation.metrics import MNISTTrigger, apply_mnist_trigger
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


def _image_bytes(image: Any) -> bytes:
    """Return stable bytes for PIL images, arrays, and tensors."""
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    return np.ascontiguousarray(array).tobytes()


def _partition_hashes(dataset: P2PFLDataset) -> tuple[str, str]:
    split = dataset._data[dataset._train_split_name]
    images = hashlib.sha256()
    labels = hashlib.sha256()
    for row in split:
        images.update(_image_bytes(row["image"]))
        labels.update(np.asarray([row["label"]], dtype=np.int64).tobytes())
    return images.hexdigest(), labels.hexdigest()


class BackdoorAttack:
    """
    Poison one deterministic fraction with a bottom-right 3x3 trigger.

    For 28x28 MNIST the default coordinates are rows 25..27 and columns
    25..27. ``trigger_value`` is expressed in the input pixel domain and is
    converted to normalized space when ``normalization_mean/std`` are set.
    """

    def __init__(
        self,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        target_class: int = 2,
        poison_rate: float = 0.2,
        seed: int = 666,
        normalization_mean: float | None = None,
        normalization_std: float | None = None,
        source_labels: list[int] | None = None,
    ) -> None:
        """Configure deterministic poisoning and trigger evaluation semantics."""
        if not 0 <= poison_rate <= 1:
            raise ValueError("poison_rate must be between zero and one")
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.seed = seed
        self.source_labels = tuple(source_labels) if source_labels is not None else None
        self.trigger = MNISTTrigger(
            size=trigger_size,
            value=trigger_value,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )
        self.node = None
        self.poisoning_evidence: dict[str, Any] | None = None
        self.application_count = 0

    def on_attach(self, node: Any) -> None:
        """Attach this attack to its malicious node."""
        self.node = node

    def poison_data(self, dataset: P2PFLDataset) -> P2PFLDataset:
        """Return an independently materialized poisoned partition exactly once."""
        if self.application_count:
            raise RuntimeError("backdoor poisoning may only be applied once")
        before_image_hash, before_label_hash = _partition_hashes(dataset)
        split_name = dataset._train_split_name
        train = dataset._data[split_name]
        examined = len(train)
        count = int(examined * self.poison_rate)
        indices = sorted(np.random.default_rng(self.seed).permutation(examined)[:count].tolist())
        poisoned_set = set(indices)

        def poison(row: dict[str, Any], index: int) -> dict[str, Any]:
            if index not in poisoned_set:
                return row
            changed = dict(row)
            triggered = apply_mnist_trigger(torch.as_tensor(np.array(row["image"], copy=True)), self.trigger)
            changed["image"] = triggered.cpu().numpy()
            changed["label"] = self.target_class
            return changed

        poisoned_train = train.map(poison, with_indices=True)
        copied_data = DatasetDict({name: (poisoned_train if name == split_name else value) for name, value in dataset._data.items()})
        result = P2PFLDataset(
            copied_data,
            train_split_name=dataset._train_split_name,
            test_split_name=dataset._test_split_name,
            batch_size=dataset.batch_size,
            dataset_name=dataset.dataset_name,
        )
        after_image_hash, after_label_hash = _partition_hashes(result)
        source_image_hash, source_label_hash = _partition_hashes(dataset)
        if (source_image_hash, source_label_hash) != (before_image_hash, before_label_hash):
            raise AssertionError("source partition was mutated during backdoor poisoning")
        self.application_count = 1
        self.poisoning_evidence = {
            "samples_examined": examined,
            "samples_poisoned": count,
            "changed_image_indices": indices,
            "changed_label_indices": [i for i in indices if int(train[i]["label"]) != self.target_class],
            "before_image_sha256": before_image_hash,
            "after_image_sha256": after_image_hash,
            "before_label_sha256": before_label_hash,
            "after_label_sha256": after_label_hash,
            "source_partition_unchanged": True,
            "attack_application_count": 1,
        }
        return result

    def poison_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility hook; offline-poisoned data must not be poisoned again."""
        return batch

    def manipulate_update(self, params: list[np.ndarray]) -> list[np.ndarray]:
        """Backdoor is a pure data-poisoning attack."""
        return params
