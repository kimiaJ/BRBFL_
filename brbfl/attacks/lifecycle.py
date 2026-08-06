"""Single integration points for MNIST data and model poisoning."""
from typing import Any

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

from .registry import Attack


def prepare_dataset(dataset: P2PFLDataset, attack: Attack | None) -> P2PFLDataset:
    """Apply offline data poisoning once during dataset preparation."""
    hook = getattr(attack, "poison_data", None)
    return hook(dataset) if hook is not None else dataset


def poison_training_batch(batch: Any, attack: Attack | None) -> Any:
    """Apply an online data attack once immediately before local training."""
    hook = getattr(attack, "poison_batch", None)
    return hook(batch) if hook is not None else batch


def poison_model_update(parameters: Any, attack: Attack | None) -> Any:
    """Apply model poisoning once when a trained update is serialized."""
    hook = getattr(attack, "manipulate_update", None)
    return hook(parameters) if hook is not None else parameters
