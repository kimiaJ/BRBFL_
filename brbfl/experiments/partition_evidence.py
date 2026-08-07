"""Stable, value-based evidence for controlled dataset assignments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from typing import Any

from brbfl.experiments.config import ExperimentConfig

EVIDENCE_SCHEMA_VERSION = "brbfl.validation.v2"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def dataset_identity(dataset: Any, configured_name: str) -> dict[str, Any]:
    """Identify a Hugging Face dataset without runtime paths or wrapper reprs."""
    splits = {}
    for name, split in dataset._data.items():
        splits[name] = {"sample_count": len(split), "fingerprint": getattr(split, "_fingerprint", None)}
    return {"name": configured_name, "splits": splits}


def build_partition_manifest(partitions: list[Any], config: ExperimentConfig, identity: dict[str, Any]) -> dict[str, Any]:
    """Build canonical node/split assignments in canonical node order."""
    entries = []
    for node_index, partition in enumerate(partitions[: config.nodes]):
        source = partition._source_partition_indices
        for split_name in sorted(source):
            indices = list(source[split_name])
            split = partition._data[split_name]
            labels = [int(value) for value in split["label"]] if "label" in split.column_names else None
            entries.append(
                {
                    "node_id": f"node-{node_index}",
                    "split": split_name,
                    "partition_index": int(partition._partition_index),
                    "sample_count": len(indices),
                    "ordered_sample_indices_sha256": _json_hash(indices),
                    "ordered_targets_sha256": _json_hash(labels) if labels is not None else None,
                    "partitioning_strategy": "random_iid",
                    "dataset_identity_sha256": _json_hash(identity),
                    "configured_seed": config.seed,
                    "effective_worker_seed": config.seed,
                }
            )
    entries.sort(key=lambda row: (row["node_id"], row["split"]))
    return {"entries": entries, "sha256": _json_hash(entries)}


def controlled_configuration(config: ExperimentConfig) -> dict[str, Any]:
    """Return only settings which must agree across clean and attack runs."""
    return {
        "nodes": config.nodes,
        "rounds": config.rounds,
        "epochs": config.epochs,
        "seed": config.seed,
        "protocol": config.protocol,
        "framework": config.framework,
        "aggregator": config.aggregator,
        "topology": config.topology.value,
        "batch_size": config.batch_size,
        "eligible_trainers": list(config.eligible_trainers or ()),
        "dataset": vars(config.dataset),
        "validation": {
            **vars(config.validation),
            "contributors": list(config.validation.contributors),
            "validators": list(config.validation.validators),
            "reference_reject_candidates": list(config.validation.reference_reject_candidates),
        },
    }


def build_provenance(config: ExperimentConfig, identity: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Record compatible-schema and controlled-configuration provenance."""
    controlled = controlled_configuration(config)
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producing_commit": _commit(),
        "controlled_configuration_sha256": _json_hash(controlled),
        "dataset_identity": identity,
        "partitioning_strategy": "random_iid",
        "configured_seeds": {"experiment": config.seed, "partition": config.seed},
        "participant_roles": {
            "contributors": sorted(config.validation.contributors),
            "validators": sorted(config.validation.validators),
            "validator_only": sorted(set(config.validation.validators) - set(config.validation.contributors)),
            "byzantine": sorted(f"node-{index}" for index in config.attack.adversaries if f"node-{index}" in config.validation.validators),
        },
        "partition_manifest_sha256": manifest["sha256"],
    }


def canonical_partition_manifest(value: Any) -> dict[str, Any]:
    """Validate and canonicalize a partition manifest by semantic keys."""
    if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
        raise AssertionError("partition evidence lacks canonical v2 manifest entries; rerun both artifacts")
    by_key = {}
    required = {
        "node_id",
        "split",
        "partition_index",
        "sample_count",
        "ordered_sample_indices_sha256",
        "ordered_targets_sha256",
        "partitioning_strategy",
        "dataset_identity_sha256",
        "configured_seed",
        "effective_worker_seed",
    }
    for position, raw in enumerate(value["entries"]):
        if not isinstance(raw, dict) or not required <= raw.keys():
            raise AssertionError(f"partition manifest entry {position} is schema-incompatible")
        row = {key: raw[key] for key in sorted(required)}  # exclude unstable/unknown runtime metadata
        key = (row["node_id"], row["split"])
        if key in by_key:
            raise AssertionError(f"duplicate partition record cannot overwrite node/split identity: {key}")
        by_key[key] = row
    entries = [by_key[key] for key in sorted(by_key)]
    return {"entries": entries, "sha256": _json_hash(entries)}
