"""Run and compare two small, dependency-light federated experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from brbfl.experiments.config import ExperimentConfig, load_experiment_config


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _json_config(config: ExperimentConfig) -> dict[str, Any]:
    value = asdict(config)
    value["topology"] = config.topology.value
    return value


def run_once(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    """Run the deterministic NumPy smoke workload and persist its evidence."""
    if config.framework != "numpy" or config.dataset.name != "synthetic-mnist":
        raise ValueError("The reproducibility smoke runner requires framework=numpy and dataset=synthetic-mnist")

    rng = np.random.default_rng(config.seed)
    sample_count, feature_count, class_count = 240, 28 * 28, 10
    features = rng.normal(0.0, 1.0, (sample_count, feature_count)).astype(np.float64)
    teacher = rng.normal(0.0, 0.2, (feature_count, class_count)).astype(np.float64)
    labels = np.argmax(features @ teacher, axis=1).astype(np.int64)
    indices = rng.permutation(sample_count)
    partitions = [part.astype(np.int64) for part in np.array_split(indices, config.nodes)]

    weights = rng.normal(0.0, 0.01, (feature_count, class_count)).astype(np.float64)
    bias = np.zeros(class_count, dtype=np.float64)
    initial_weights, initial_bias = weights.copy(), bias.copy()
    metrics: list[dict[str, Any]] = []
    participants: list[dict[str, Any]] = []
    nodes_per_round = min(config.nodes, max(1, config.nodes - 1))

    for round_number in range(config.rounds):
        selected = np.sort(rng.choice(config.nodes, size=nodes_per_round, replace=False))
        participants.append({"round": round_number, "nodes": [f"node-{int(node)}" for node in selected]})
        local_models: list[tuple[np.ndarray, np.ndarray]] = []
        for node in selected:
            local_weights, local_bias = weights.copy(), bias.copy()
            local_indices = partitions[int(node)]
            for _ in range(config.epochs):
                logits = features[local_indices] @ local_weights + local_bias
                shifted = logits - logits.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                probabilities[np.arange(len(local_indices)), labels[local_indices]] -= 1.0
                gradient_weights = features[local_indices].T @ probabilities / len(local_indices)
                gradient_bias = probabilities.mean(axis=0)
                local_weights -= 0.05 * gradient_weights
                local_bias -= 0.05 * gradient_bias
            local_models.append((local_weights, local_bias))
        weights = np.mean([model[0] for model in local_models], axis=0)
        bias = np.mean([model[1] for model in local_models], axis=0)
        logits = features @ weights + bias
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(probabilities[np.arange(sample_count), labels]).mean()
        accuracy = np.mean(np.argmax(logits, axis=1) == labels)
        metrics.append({"round": round_number, "loss": float(loss), "accuracy": float(accuracy)})

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "parameters.npz", initial_weights=initial_weights, initial_bias=initial_bias, final_weights=weights, final_bias=bias
    )
    result = {
        "config": _json_config(config),
        "partition_indices": [part.tolist() for part in partitions],
        "partition_sha256": [_digest(part) for part in partitions],
        "initial_parameters_sha256": _digest(initial_weights, initial_bias),
        "participating_nodes": participants,
        "round_metrics": metrics,
        "final_parameters_sha256": _digest(weights, bias),
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def compare_runs(first_dir: Path, second_dir: Path) -> dict[str, Any]:
    """Compare persisted run evidence using exact and numerical comparisons."""
    first = json.loads((first_dir / "run.json").read_text(encoding="utf-8"))
    second = json.loads((second_dir / "run.json").read_text(encoding="utf-8"))
    first_parameters = np.load(first_dir / "parameters.npz")
    second_parameters = np.load(second_dir / "parameters.npz")

    def parameter_comparison(prefix: str) -> dict[str, Any]:
        names = (f"{prefix}_weights", f"{prefix}_bias")
        exact = all(np.array_equal(first_parameters[name], second_parameters[name]) for name in names)
        maximum = max(float(np.max(np.abs(first_parameters[name] - second_parameters[name]))) for name in names)
        return {
            "classification": "identical" if exact else "approximately_equal" if maximum <= 1e-12 else "different",
            "exact": exact,
            "max_absolute_difference": maximum,
        }

    fields = {
        "data_partitions": first["partition_indices"] == second["partition_indices"],
        "participating_nodes": first["participating_nodes"] == second["participating_nodes"],
        "per_round_loss": [row["loss"] for row in first["round_metrics"]] == [row["loss"] for row in second["round_metrics"]],
        "per_round_accuracy": [row["accuracy"] for row in first["round_metrics"]] == [row["accuracy"] for row in second["round_metrics"]],
    }
    comparisons: dict[str, Any] = {
        name: {"classification": "identical" if exact else "different", "exact": exact} for name, exact in fields.items()
    }
    comparisons["initial_model_parameters"] = parameter_comparison("initial")
    comparisons["final_model_parameters"] = parameter_comparison("final")
    return {
        "configuration_identical": first["config"] == second["config"],
        "comparisons": comparisons,
        "all_requested_outputs_identical": all(item["classification"] == "identical" for item in comparisons.values()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "numerical_tolerance": {"absolute": 1e-12, "relative": 0.0},
    }


def run_twice(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Remove previous evidence, execute twice in fresh processes, and compare."""
    config = load_experiment_config(config_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for run_number in (1, 2):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(config.seed)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "brbfl.experiments.smoke_reproducibility",
                "--single-run",
                str(config_path),
                str(output_dir / f"run-{run_number}"),
            ],
            check=True,
            env=environment,
        )
    comparison = compare_runs(output_dir / "run-1", output_dir / "run-2")
    comparison["config_path"] = str(config_path)
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Clean smoke reproducibility comparison",
        "",
        f"Configuration: `{config_path}`; seed: `{config.seed}`.",
        "",
        "| Output | Classification | Exact | Maximum absolute difference |",
        "|---|---|---:|---:|",
    ]
    for name, result in comparison["comparisons"].items():
        label = name.replace("_", " ")
        exact = str(result["exact"]).lower()
        difference = result.get("max_absolute_difference", "n/a")
        lines.append(f"| {label} | {result['classification']} | {exact} | {difference} |")
    lines += ["", "Both runs completed in separate processes after the previous output directory was removed.", ""]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-run", nargs=2, metavar=("CONFIG", "OUTPUT_DIR"))
    parser.add_argument("--config", type=Path, default=Path("configs/smoke/mnist_clean.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproducibility"))
    arguments = parser.parse_args()
    if arguments.single_run:
        run_once(load_experiment_config(arguments.single_run[0]), Path(arguments.single_run[1]))
    else:
        comparison = run_twice(arguments.config, arguments.output_dir)
        identical = sum(
            result["classification"] == "identical" for result in comparison["comparisons"].values()
        )
        total = len(comparison["comparisons"])
        status = "identical" if comparison["all_requested_outputs_identical"] else "different"
        print(f"Reproducibility: {status} ({identical}/{total} outputs); artifacts: {arguments.output_dir}")


if __name__ == "__main__":
    main()
