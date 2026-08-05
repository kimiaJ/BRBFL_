"""Compatibility adapter for the current MNIST experiment."""

from __future__ import annotations

from brbfl.experiments.config import ExperimentConfig, load_experiment_config
from p2pfl.examples.mnist.mnist import mnist


def run_mnist(config: ExperimentConfig) -> None:
    mnist(config=config)


def run_mnist_from_yaml(path: str) -> None:
    run_mnist(load_experiment_config(path))
