"""Tests for the clean reproducibility smoke runner."""

import subprocess
import sys
from pathlib import Path

import pytest

from brbfl.experiments.config import DatasetConfig, ExperimentConfig, load_experiment_config
from brbfl.experiments.smoke_reproducibility import compare_runs, run_once, run_twice

COMPATIBLE_CONFIG = Path("configs/smoke/mnist_numpy_clean.yaml")


def test_two_smoke_runs_are_identical(tmp_path: Path):
    """Equal seeds produce exact evidence in two runs."""
    config = ExperimentConfig(nodes=3, rounds=2, epochs=1, seed=666, framework="numpy", dataset=DatasetConfig(name="synthetic-mnist"))
    run_once(config, tmp_path / "one")
    run_once(config, tmp_path / "two")

    comparison = compare_runs(tmp_path / "one", tmp_path / "two")

    assert comparison["all_requested_outputs_identical"]
    assert all(result["classification"] == "identical" for result in comparison["comparisons"].values())


def test_config_boundary_accepts_synthetic_numpy_and_rejects_real_pytorch(tmp_path: Path):
    """The dedicated config is valid while the real training config fails clearly."""
    run_once(load_experiment_config(COMPATIBLE_CONFIG), tmp_path / "compatible")
    incompatible = load_experiment_config(Path("configs/smoke/mnist_clean.yaml"))
    with pytest.raises(ValueError, match="requires framework=numpy and dataset=synthetic-mnist"):
        run_once(incompatible, tmp_path / "incompatible")


def test_two_isolated_processes_produce_comparison(tmp_path: Path):
    """Fresh child processes produce the complete exact comparison."""
    comparison = run_twice(COMPATIBLE_CONFIG, tmp_path / "isolated")
    assert comparison["all_requested_outputs_identical"]
    assert len(comparison["comparisons"]) == 6


def test_module_execution_invokes_experiment(tmp_path: Path):
    """The module entry point executes both runs and reports their result."""
    output_dir = tmp_path / "reproducibility"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "brbfl.experiments.smoke_reproducibility",
            "--config",
            str(COMPATIBLE_CONFIG),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Reproducibility: identical (6/6 outputs)" in completed.stdout
    assert (output_dir / "run-1" / "run.json").is_file()
    assert (output_dir / "run-2" / "run.json").is_file()
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "comparison.md").is_file()
