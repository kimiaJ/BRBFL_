"""Dataset partition helpers for reproducible experiments."""

from __future__ import annotations

from brbfl.experiments.config import ExperimentConfig
from brbfl.experiments.reproducibility import seed_everything
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy


def partition_dataset(data: P2PFLDataset, config: ExperimentConfig) -> list[P2PFLDataset]:
    """Generate deterministic IID partitions for an experiment config."""
    seed_everything(config.seed)
    data.set_batch_size(config.batch_size)
    num_partitions = config.nodes * config.dataset.partition_multiplier if config.dataset.reduced else config.nodes
    return data.generate_partitions(num_partitions, RandomIIDPartitionStrategy, seed=config.seed)
