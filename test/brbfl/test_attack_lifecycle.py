"""Regression tests for the centralized MNIST attack lifecycle."""

import numpy as np

from brbfl.attacks import create_attack, poison_model_update, poison_training_batch, prepare_dataset


class CountingAttack:
    """Spy implementing every supported lifecycle hook."""

    def __init__(self):
        """Initialize hook counters."""
        self.data_calls = 0
        self.batch_calls = 0
        self.model_calls = 0

    def poison_data(self, dataset):
        """Count offline data poisoning."""
        self.data_calls += 1
        return dataset

    def poison_batch(self, batch):
        """Count online data poisoning."""
        self.batch_calls += 1
        return batch

    def manipulate_update(self, parameters):
        """Count model poisoning."""
        self.model_calls += 1
        return parameters


def test_clean_configuration_applies_no_attack():
    """A clean config leaves all lifecycle values untouched."""
    dataset = object()
    batch = object()
    parameters = [np.array([1.0])]
    attack = create_attack("none")

    assert attack is None
    assert prepare_dataset(dataset, attack) is dataset
    assert poison_training_batch(batch, attack) is batch
    assert poison_model_update(parameters, attack) is parameters


def test_enabled_attack_is_applied_once_at_each_applicable_stage():
    """Every central integration point invokes its hook exactly once."""
    attack = CountingAttack()
    dataset = object()
    batch = object()
    parameters = [np.array([1.0])]

    prepare_dataset(dataset, attack)
    poison_training_batch(batch, attack)
    poison_model_update(parameters, attack)

    assert (attack.data_calls, attack.batch_calls, attack.model_calls) == (1, 1, 1)


def test_registry_constructs_existing_attack_with_configured_parameters():
    """The registry creates and configures preserved implementations."""
    attack = create_attack("sign_flipping", {"scale": -2.0})
    result = poison_model_update([np.array([3.0])], attack)

    assert result[0].tolist() == [-6.0]
