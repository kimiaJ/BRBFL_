"""Tests for loading experiment configurations."""

from brbfl.experiments.config import TopologyType, load_experiment_config


def test_load_smoke_mnist_config():
    """The clean smoke configuration loads all reference settings."""
    config = load_experiment_config("configs/smoke/mnist_clean.yaml")

    assert config.nodes == 3
    assert config.rounds == 2
    assert config.seed == 666
    assert config.framework == "numpy"
    assert config.dataset.name == "synthetic-mnist"
    assert config.dataset.reduced is False
    assert config.attack.name == "none"
    assert config.attack.adversaries == ()
    assert config.topology == TopologyType.FULL
