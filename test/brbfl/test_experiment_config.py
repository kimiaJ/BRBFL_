from brbfl.experiments.config import TopologyType, load_experiment_config


def test_load_smoke_mnist_config():
    config = load_experiment_config("configs/smoke/mnist_clean.yaml")

    assert config.nodes == 3
    assert config.rounds == 1
    assert config.seed == 666
    assert config.dataset.name == "p2pfl/MNIST"
    assert config.dataset.reduced is True
    assert config.attack.name == "none"
    assert config.attack.adversaries == ()
    assert config.topology == TopologyType.FULL
