"""Compact, non-mutating evidence for model-update transformations."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    return np.asarray(value).copy()


def _hash(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def transformation_evidence(before: list[Any], after: list[Any], names: list[str], scale: float, tolerance: float = 1e-6) -> dict[str, Any]:
    """Verify ``after = scale * before`` and return textual tensor evidence."""
    original = [_array(value) for value in before]
    transformed = [_array(value) for value in after]
    if len(original) != len(transformed) or len(original) != len(names):
        raise AssertionError("parameter names and pre/post updates must have equal lengths")
    flat_before = np.concatenate([value.astype(np.float64, copy=False).ravel() for value in original])
    flat_after = np.concatenate([value.astype(np.float64, copy=False).ravel() for value in transformed])
    expected = flat_before * scale
    maximum_error = float(np.max(np.abs(flat_after - expected), initial=0.0))
    denominator = float(np.linalg.norm(flat_before) * np.linalg.norm(flat_after))
    cosine = float(np.dot(flat_before, flat_after) / denominator) if denominator else None
    if maximum_error > tolerance:
        raise AssertionError(f"sign-flipping formula error {maximum_error} exceeds tolerance {tolerance}")
    return {
        "formula": "attacked = scale * original",
        "scale": scale,
        "numerical_tolerance": tolerance,
        "pre_attack_sha256": _hash(original),
        "post_attack_sha256": _hash(transformed),
        "pre_attack_l2_norm": float(np.linalg.norm(flat_before)),
        "post_attack_l2_norm": float(np.linalg.norm(flat_after)),
        "cosine_similarity": cosine,
        "maximum_transformation_error": maximum_error,
        "parameters": [
            {
                "name": name,
                "shape": list(pre.shape),
                "pre_sample": pre.ravel()[:3].tolist(),
                "post_sample": post.ravel()[:3].tolist(),
            }
            for name, pre, post in zip(names, original, transformed, strict=True)
        ],
    }


class AuditedModelUpdateAttack:
    """Delegate an attack while retaining compact evidence for every invocation."""

    def __init__(self, attack: Any, parameter_names: list[str], tolerance: float = 1e-6):
        """Initialize a recorder around the preserved attack implementation."""
        self.attack = attack
        self.parameter_names = parameter_names
        self.tolerance = tolerance
        self.events: list[dict[str, Any]] = []

    def manipulate_update(self, parameters: list[Any]) -> list[Any]:
        """Transform one copied update and prove the caller's input was not mutated."""
        original = [_array(value) for value in parameters]
        original_hash = _hash(original)
        transformed = self.attack.manipulate_update([value.copy() for value in original])
        evidence = transformation_evidence(original, transformed, self.parameter_names, float(self.attack.params["scale"]), self.tolerance)
        if _hash(original) != original_hash or _hash([_array(value) for value in parameters]) != original_hash:
            raise AssertionError("evidence capture or attack mutated the original update")
        evidence["original_pre_attack_update_preserved"] = True
        self.events.append(evidence)
        return transformed

    def on_attach(self, node: Any) -> None:
        """Forward node attachment to the preserved implementation."""
        self.attack.on_attach(node)
