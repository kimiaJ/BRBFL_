"""Machine-readable evidence helpers for controlled label-flipping runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def labels(dataset: Any) -> list[int]:
    """Return the untransformed training labels from a P2PFL partition."""
    split = dataset._data[dataset._train_split_name]
    return [int(value) for value in split["label"]]


def label_hash(values: list[int]) -> str:
    """Hash labels in their deterministic partition order."""
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def parameter_hash(parameters: Any) -> str:
    """Hash serialized model tensors/arrays in order."""
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().numpy() if hasattr(parameter, "detach") else np.asarray(parameter)
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def audit_partition(node_id: int, before: list[int], after: list[int], flip_map: dict[int, int], malicious: bool) -> dict[str, Any]:
    """Describe and validate a partition transformation without changing it."""
    changed = [index for index, pair in enumerate(zip(before, after, strict=True)) if pair[0] != pair[1]]
    expected = [index for index, value in enumerate(before) if value in flip_map and flip_map[value] != value] if malicious else []
    if changed != expected:
        raise AssertionError(f"node-{node_id} changed indices do not match the configured label map")
    return {
        "node_id": f"node-{node_id}",
        "malicious": malicious,
        "labels_examined": len(before) if malicious else 0,
        "labels_changed": len(changed),
        "changed_partition_indices": changed,
        "before_label_sha256": label_hash(before),
        "after_label_sha256": label_hash(after),
        "attack_application_count": 1 if malicious else 0,
    }


def write_evidence(output_dir: str | Path, evidence: dict[str, Any]) -> Path:
    """Persist validation evidence."""
    path = Path(output_dir) / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
