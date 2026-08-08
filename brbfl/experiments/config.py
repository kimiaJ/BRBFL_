"""Typed experiment configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class TopologyType(Enum):
    """Supported topology names without importing the runnable P2PFL stack."""

    STAR = "star"
    FULL = "full"
    LINE = "line"
    RING = "ring"
    RANDOM_2 = "random_2"
    RANDOM_3 = "random_3"
    RANDOM_4 = "random_4"


@dataclass(frozen=True)
class DatasetConfig:
    """Federated dataset and deterministic partition settings."""

    name: str = "p2pfl/MNIST"
    distribution: str = "iid"
    reduced: bool = False
    partition_multiplier: int = 50


@dataclass(frozen=True)
class AttackConfig:
    """Attack selection and attack-specific parameters."""

    name: str = "none"
    adversaries: tuple[int, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationConfig:
    """Optional pre-aggregation validator-subgroup policy."""

    enabled: bool = False
    contributors: tuple[str, ...] = ()
    validators: tuple[str, ...] = ()
    quorum: int = 0
    acceptance_threshold: int = 0
    max_l2_norm: float = float("inf")
    reference_reject_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockchainConfig:
    """Optional lifecycle-ledger backend settings."""

    enabled: bool = False
    backend: str = "memory"
    fail_closed: bool = True


@dataclass(frozen=True)
class ParticipantSelectionConfig:
    """Round-role selection strategy (static until a future CA milestone)."""

    mode: str = "static"


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete runnable experiment settings."""

    nodes: int = 10
    rounds: int = 15
    epochs: int = 1
    seed: int = 666
    protocol: str = "grpc"
    framework: str = "pytorch"
    aggregator: str = "fedavg"
    topology: TopologyType = TopologyType.FULL
    batch_size: int = 128
    show_metrics: bool = True
    measure_time: bool = False
    save_csv: bool = True
    output_dir: str = "results/mnist"
    eligible_trainers: tuple[str, ...] | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    participant_selection: ParticipantSelectionConfig = field(default_factory=ParticipantSelectionConfig)


def _topology(value: str | TopologyType) -> TopologyType:
    return value if isinstance(value, TopologyType) else TopologyType(value)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    dataset_raw = raw.get("dataset", {}) or {}
    attack_raw = raw.get("attack", {}) or {}
    validation_raw = raw.get("validation", {}) or {}
    blockchain_raw = raw.get("blockchain", {}) or {}
    selection_raw = raw.get("participant_selection", {}) or {}

    dataset = DatasetConfig(
        name=dataset_raw.get("name", DatasetConfig.name),
        distribution=dataset_raw.get("distribution", DatasetConfig.distribution),
        reduced=dataset_raw.get("reduced", DatasetConfig.reduced),
        partition_multiplier=dataset_raw.get("partition_multiplier", DatasetConfig.partition_multiplier),
    )
    attack = AttackConfig(
        name=attack_raw.get("name", AttackConfig.name),
        adversaries=tuple(attack_raw.get("adversaries", ()) or ()),
        parameters=dict(attack_raw.get("parameters", {}) or {}),
    )
    validation = ValidationConfig(
        enabled=bool(validation_raw.get("enabled", False)),
        contributors=tuple(validation_raw.get("contributors", ()) or ()),
        validators=tuple(validation_raw.get("validators", ()) or ()),
        quorum=int(validation_raw.get("quorum", 0)),
        acceptance_threshold=int(validation_raw.get("acceptance_threshold", 0)),
        max_l2_norm=float(validation_raw.get("max_l2_norm", float("inf"))),
        reference_reject_candidates=tuple(validation_raw.get("reference_reject_candidates", ()) or ()),
    )
    blockchain = BlockchainConfig(
        enabled=bool(blockchain_raw.get("enabled", False)),
        backend=str(blockchain_raw.get("backend", "memory")),
        fail_closed=bool(blockchain_raw.get("fail_closed", True)),
    )
    if blockchain.enabled and blockchain.backend != "memory":
        raise ValueError(f"unsupported blockchain ledger backend: {blockchain.backend}")
    participant_selection = ParticipantSelectionConfig(mode=str(selection_raw.get("mode", "static")))
    if participant_selection.mode != "static":
        raise ValueError(f"unsupported participant selection mode: {participant_selection.mode}")

    return ExperimentConfig(
        nodes=raw.get("nodes", ExperimentConfig.nodes),
        rounds=raw.get("rounds", ExperimentConfig.rounds),
        epochs=raw.get("epochs", ExperimentConfig.epochs),
        seed=raw.get("seed", ExperimentConfig.seed),
        protocol=raw.get("protocol", ExperimentConfig.protocol),
        framework=raw.get("framework", ExperimentConfig.framework),
        aggregator=raw.get("aggregator", ExperimentConfig.aggregator),
        topology=_topology(raw.get("topology", ExperimentConfig.topology)),
        batch_size=raw.get("batch_size", ExperimentConfig.batch_size),
        show_metrics=raw.get("show_metrics", ExperimentConfig.show_metrics),
        measure_time=raw.get("measure_time", ExperimentConfig.measure_time),
        save_csv=raw.get("save_csv", ExperimentConfig.save_csv),
        output_dir=raw.get("output_dir", ExperimentConfig.output_dir),
        eligible_trainers=(tuple(raw["eligible_trainers"]) if raw.get("eligible_trainers") is not None else None),
        dataset=dataset,
        attack=attack,
        validation=validation,
        blockchain=blockchain,
        participant_selection=participant_selection,
    )
