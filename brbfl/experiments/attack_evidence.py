"""Attack-aware, immutable evidence for controlled partition transformations."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
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


def _array(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def _canonical(value: Any) -> bytes:
    if hasattr(value, "mode") and hasattr(value, "size"):
        value = _array(value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return b"array\0" + str(array.dtype).encode() + b"\0" + repr(array.shape).encode() + b"\0" + array.tobytes()
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()


@dataclass(frozen=True)
class PartitionSnapshot:
    """A detached view of a training partition captured before poisoning."""

    rows: tuple[dict[str, Any], ...]
    sha256: str
    image_sha256: str
    label_sha256: str


def snapshot_partition(dataset: Any) -> PartitionSnapshot:
    """Capture a partition without retaining mutable dataset references."""
    rows = tuple(copy.deepcopy(dict(row)) for row in dataset._data[dataset._train_split_name])
    partition_digest = hashlib.sha256()
    image_digest = hashlib.sha256()
    labels_before: list[int] = []
    for row in rows:
        for key in sorted(row):
            encoded = _canonical(row[key])
            partition_digest.update(key.encode() + b"\0" + encoded)
            if key == "image":
                image_digest.update(encoded)
        labels_before.append(int(row["label"]))
    return PartitionSnapshot(rows, partition_digest.hexdigest(), image_digest.hexdigest(), label_hash(labels_before))


def parameter_hash(parameters: Any) -> str:
    """Hash serialized model tensors/arrays in order."""
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().numpy() if hasattr(parameter, "detach") else np.asarray(parameter)
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _partition_diff(before: PartitionSnapshot, after_dataset: Any) -> tuple[dict[str, Any], PartitionSnapshot]:
    """Compute generic evidence; attach no attack semantics to expected changes."""
    after = snapshot_partition(after_dataset)
    if len(before.rows) != len(after.rows):
        raise AssertionError("source and result partitions have different lengths")
    image_changed: list[int] = []
    label_changed: list[int] = []
    for index, (source, result) in enumerate(zip(before.rows, after.rows, strict=True)):
        if set(source) != set(result):
            raise AssertionError(f"partition fields changed at index {index}")
        if "image" in source and not np.array_equal(_array(source["image"]), _array(result["image"])):
            image_changed.append(index)
        if int(source["label"]) != int(result["label"]):
            label_changed.append(index)
        unrelated = set(source) - {"image", "label"}
        if any(_canonical(source[key]) != _canonical(result[key]) for key in unrelated):
            raise AssertionError(f"unrelated partition field changed at index {index}")
    changed = set(image_changed) | set(label_changed)
    evidence = {
        "samples_examined": len(before.rows),
        "source_partition_sha256": before.sha256,
        "result_partition_sha256": after.sha256,
        "before_image_sha256": before.image_sha256,
        "after_image_sha256": after.image_sha256,
        "before_label_sha256": before.label_sha256,
        "after_label_sha256": after.label_sha256,
        "image_changed_indices": image_changed,
        "label_changed_indices": label_changed,
        "labels_changed": len(label_changed),
        "changed_partition_indices": label_changed,
        "unchanged_indices": [i for i in range(len(before.rows)) if i not in changed],
    }
    return evidence, after


def _validate_label_flipping(
    node_id: int, before: PartitionSnapshot, after: PartitionSnapshot, evidence: dict[str, Any], flip_map: dict[int, int]
) -> None:
    if not flip_map:
        raise AssertionError(f"node-{node_id} label flipping requires a configured label map")
    expected = [i for i, row in enumerate(before.rows) if int(row["label"]) in flip_map and flip_map[int(row["label"])] != row["label"]]
    if evidence["label_changed_indices"] != expected:
        raise AssertionError(f"node-{node_id} changed indices do not match the configured label map")
    if evidence["image_changed_indices"]:
        raise AssertionError(f"node-{node_id} label flipping changed images")
    for index, (source, result) in enumerate(zip(before.rows, after.rows, strict=True)):
        expected_label = int(flip_map.get(int(source["label"]), int(source["label"])))
        if int(result["label"]) != expected_label:
            raise AssertionError(f"node-{node_id} label at index {index} does not equal its mapped label")


def _validate_backdoor(node_id: int, before: PartitionSnapshot, after: PartitionSnapshot, evidence: dict[str, Any], attack: Any) -> None:
    examined = len(before.rows)
    count = int(examined * attack.poison_rate)
    poisoned = sorted(np.random.default_rng(attack.seed).permutation(examined)[:count].tolist())
    if attack.poisoning_evidence is None or attack.poisoning_evidence.get("changed_image_indices") != poisoned:
        raise AssertionError(f"node-{node_id} backdoor selected indices do not match deterministic selection")
    expected_label_changes = [i for i in poisoned if int(before.rows[i]["label"]) != attack.target_class]
    if evidence["image_changed_indices"] != poisoned:
        raise AssertionError(f"node-{node_id} backdoor image changes do not match poisoned indices")
    if evidence["label_changed_indices"] != expected_label_changes:
        raise AssertionError(f"node-{node_id} backdoor label changes do not match all-to-one replacement")

    trigger_valid = True
    outside_valid = True
    for index, (source, result) in enumerate(zip(before.rows, after.rows, strict=True)):
        source_pixels = _array(source["image"])
        result_pixels = _array(result["image"])
        if index not in poisoned:
            if not np.array_equal(source_pixels, result_pixels) or int(source["label"]) != int(result["label"]):
                raise AssertionError(f"node-{node_id} non-poisoned sample {index} changed")
            continue
        if int(result["label"]) != attack.target_class:
            raise AssertionError(f"node-{node_id} poisoned label {index} does not equal target label")
        rows = slice(result_pixels.shape[-2] - attack.trigger_size, result_pixels.shape[-2])
        columns = slice(result_pixels.shape[-1] - attack.trigger_size, result_pixels.shape[-1])
        trigger_value = np.asarray(attack.trigger_value, dtype=result_pixels.dtype)
        if not np.all(result_pixels[..., rows, columns] == trigger_value):
            trigger_valid = False
        mask = np.ones(result_pixels.shape, dtype=bool)
        mask[..., rows, columns] = False
        if not np.array_equal(source_pixels[mask], result_pixels[mask]):
            outside_valid = False
    if not trigger_valid:
        raise AssertionError(f"node-{node_id} poisoned image has a missing or malformed trigger")
    if not outside_valid:
        raise AssertionError(f"node-{node_id} poisoned image changed outside the trigger")

    evidence.update(
        {
            "samples_poisoned": count,
            "poisoned_indices": poisoned,
            "original_labels_at_poisoned_indices": [int(before.rows[i]["label"]) for i in poisoned],
            "resulting_labels_at_poisoned_indices": [int(after.rows[i]["label"]) for i in poisoned],
            "target_label": attack.target_class,
            "trigger_coordinates": attack.trigger.coordinates(),
            "trigger_value": attack.trigger_value,
            "trigger_validation_passed": trigger_valid,
            "non_trigger_pixels_preserved": outside_valid,
        }
    )


def audit_partition(
    *, node_id: int, attack_type: str, before: PartitionSnapshot, after: Any, malicious: bool, attack: Any | None = None
) -> dict[str, Any]:
    """Dispatch generic partition evidence to exactly one attack-specific validator."""
    evidence, after_snapshot = _partition_diff(before, after)
    evidence.update({"attack_type": attack_type, "node_id": f"node-{node_id}", "malicious": malicious})
    if not malicious:
        if evidence["image_changed_indices"] or evidence["label_changed_indices"]:
            raise AssertionError(f"node-{node_id} benign partition changed")
        evidence.update({"samples_poisoned": 0, "source_partition_unchanged": True, "attack_application_count": 0})
        return evidence
    if attack_type == "label_flipping":
        _validate_label_flipping(node_id, before, after_snapshot, evidence, dict(attack.params.get("flip_map", {})))
        evidence.update(
            {
                "labels_examined": len(before.rows),
                "labels_changed": len(evidence["label_changed_indices"]),
                "changed_partition_indices": evidence["label_changed_indices"],
                "samples_poisoned": len(evidence["label_changed_indices"]),
            }
        )
    elif attack_type == "backdoor":
        _validate_backdoor(node_id, before, after_snapshot, evidence, attack)
    elif attack_type == "sign_flipping":
        if evidence["image_changed_indices"] or evidence["label_changed_indices"]:
            raise AssertionError(f"node-{node_id} sign flipping unexpectedly changed its dataset partition")
        evidence.update({"samples_poisoned": 0, "source_partition_unchanged": True, "attack_application_count": 0})
        return evidence
    else:
        raise AssertionError(f"node-{node_id} has no partition audit validator for attack type {attack_type!r}")
    evidence.update({"source_partition_unchanged": True, "attack_application_count": 1})
    return evidence


def write_evidence(output_dir: str | Path, evidence: dict[str, Any]) -> Path:
    """Persist validation evidence."""
    path = Path(output_dir) / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
