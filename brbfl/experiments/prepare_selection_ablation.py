"""Prepare controlled MNIST configs without running training or inventing results."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from brbfl.experiments.config import load_experiment_config

STRATEGIES = ("static", "trust_contributors", "ca_contributors")
DEFAULT_SEEDS = (666, 667, 668)


def configuration(strategy: str, attacked: bool, seed: int) -> dict:
    """Keep data, models, validation and detector identical across selection modes."""
    if strategy not in STRATEGIES:
        raise ValueError("unsupported contributor ablation strategy")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    trainers = [f"node-{i}" for i in range(4)]
    validators = ["node-4", "node-5"]
    scenario = "attacked" if attacked else "clean"
    return {
        "nodes": 6,
        "rounds": 3,
        "epochs": 1,
        "seed": seed,
        "protocol": "memory",
        "framework": "pytorch",
        "aggregator": "fedavg",
        "topology": "full",
        "batch_size": 128,
        "show_metrics": False,
        "save_csv": True,
        "output_dir": f"results/selection-ablation/{seed}/{strategy}/{scenario}",
        "eligible_trainers": trainers,
        "dataset": {"name": "p2pfl/MNIST", "distribution": "iid", "reduced": True, "partition_multiplier": 50},
        "validation": {
            "enabled": True,
            "contributors": trainers.copy(),
            "validators": validators.copy(),
            "quorum": 2,
            "acceptance_threshold": 2,
        },
        "blockchain": {"enabled": True, "backend": "memory", "fail_closed": True},
        "trust": {"enabled": True, "observation_only": False, "prior_alpha": 1, "prior_beta": 1},
        "selection": {
            "strategy": strategy,
            "eligible_participants": validators.copy(),
            "bootstrap_rounds": 1,
            "minimum_trust": 0.5,
            "minimum_contributors": 3,
        },
        "ca": {
            "enabled": strategy == "ca_contributors",
            "minimum_contributors": 3,
            "suspicious_contributors": True,
            "policy": {"severe_suspicious_probation_rounds": 1, "recovery_enabled": False},
        },
        "model_detector": {"enabled": True, "committee_size": 2, "eligible_participants": validators.copy()},
        "attack": {"name": "sign_flipping", "adversaries": [1], "parameters": {"scale": -3.0}}
        if attacked
        else {"name": "none", "adversaries": []},
    }


def prepare(output: Path, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[Path, ...]:
    """Write the full matrix into new paths, refusing to overwrite earlier configs."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("non-empty unique seeds are required")
    pending = []
    for seed in seeds:
        for strategy in STRATEGIES:
            for attacked in (False, True):
                name = f"{strategy}_{'attacked' if attacked else 'clean'}_{seed}.yaml"
                pending.append((output / name, configuration(strategy, attacked, seed)))
    if any(path.exists() for path, _ in pending):
        raise FileExistsError("refusing to replace existing selection-ablation configs")
    output.mkdir(parents=True, exist_ok=True)
    for path, config in pending:
        path.write_text(yaml.safe_dump(deepcopy(config), sort_keys=False), encoding="utf-8")
        load_experiment_config(path)
    return tuple(path for path, _ in pending)


def main() -> None:
    """Print commands for the existing training environment; never launch them implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/ablation/selection"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    for path in prepare(args.output, tuple(args.seeds)):
        print(f'python -m p2pfl.examples.mnist.mnist --config "{path.as_posix()}"')


if __name__ == "__main__":
    main()
