"""Tests for the clean reproducibility smoke runner."""

from pathlib import Path

from brbfl.experiments.config import DatasetConfig, ExperimentConfig
from brbfl.experiments.smoke_reproducibility import compare_runs, run_once


def test_two_smoke_runs_are_identical(tmp_path: Path):
    """Equal seeds produce exact evidence in two runs."""
    config = ExperimentConfig(nodes=3, rounds=2, epochs=1, seed=666, framework="numpy", dataset=DatasetConfig(name="synthetic-mnist"))
    run_once(config, tmp_path / "one")
    run_once(config, tmp_path / "two")

    comparison = compare_runs(tmp_path / "one", tmp_path / "two")

    assert comparison["all_requested_outputs_identical"]
    assert all(result["classification"] == "identical" for result in comparison["comparisons"].values())
