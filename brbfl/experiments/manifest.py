"""Experiment manifest writing."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from brbfl.experiments.config import ExperimentConfig


def current_git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("p2pfl", "torch", "numpy", "pandas", "pyyaml"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_manifest(config: ExperimentConfig, start_time: datetime | None = None) -> dict[str, object]:
    started = start_time or datetime.now(timezone.utc)
    return {
        "git_commit": current_git_commit(),
        "seed": config.seed,
        "node_count": config.nodes,
        "malicious_node_ids": list(config.attack.adversaries),
        "dataset": config.dataset.name,
        "distribution": config.dataset.distribution,
        "topology": config.topology.value,
        "attack": {"name": config.attack.name, "parameters": config.attack.parameters},
        "aggregation_method": config.aggregator,
        "start_time": started.isoformat(),
        "software_versions": software_versions(),
    }


def write_manifest(output_dir: str | Path, config: ExperimentConfig, start_time: datetime | None = None) -> Path:
    path = Path(output_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(config, start_time), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
