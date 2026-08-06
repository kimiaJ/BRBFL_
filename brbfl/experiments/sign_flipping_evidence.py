"""Compact, non-mutating evidence for model-update transformations."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    return np.asarray(value).copy()


def _hash(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
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
    pre_norm = float(np.linalg.norm(flat_before))
    post_norm = float(np.linalg.norm(flat_after))
    if not np.isclose(post_norm, abs(scale) * pre_norm, rtol=tolerance, atol=tolerance):
        raise AssertionError("sign-flipping norm does not match the configured scale")
    if pre_norm and scale < 0 and not np.isclose(cosine, -1.0, rtol=tolerance, atol=tolerance):
        raise AssertionError("sign-flipping cosine similarity is not -1")
    return {
        "formula": "attacked = scale * original",
        "scale": scale,
        "numerical_tolerance": tolerance,
        "pre_attack_sha256": _hash(original),
        "post_attack_sha256": _hash(transformed),
        "pre_attack_l2_norm": pre_norm,
        "post_attack_l2_norm": post_norm,
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
    """Apply an update attack once and audit its potentially many transmissions."""

    def __init__(
        self,
        attack: Any,
        parameter_names: list[str],
        tolerance: float = 1e-6,
        round_provider: Callable[[], Any] | None = None,
    ):
        """Initialize a recorder around the preserved attack implementation."""
        self.attack = attack
        self.parameter_names = parameter_names
        self.tolerance = tolerance
        self._round_provider = round_provider
        self._cache: dict[str, list[np.ndarray]] = {}
        self._post_hash_to_update_id: dict[str, str] = {}
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.transmissions: list[dict[str, Any]] = []

    def _round_id(self) -> str:
        value = self._round_provider() if self._round_provider is not None else None
        return "unassigned" if value is None else str(value)

    def manipulate_update(self, parameters: list[Any]) -> list[Any]:
        """Transform one copied update and prove the caller's input was not mutated."""
        original = [_array(value) for value in parameters]
        original_hash = _hash(original)
        round_id = self._round_id()
        update_id = f"round-{round_id}:{original_hash}"
        with self._lock:
            # Defensive handling for a caller that passes our output back through
            # the hook.  It is a transmission, never a new mathematical attack.
            previously_attacked_id = self._post_hash_to_update_id.get(original_hash)
            if previously_attacked_id is not None:
                update_id = previously_attacked_id
                transformed = original
            elif update_id in self._cache:
                transformed = [value.copy() for value in self._cache[update_id]]
            else:
                transformed = self.attack.manipulate_update([value.copy() for value in original])
                evidence = transformation_evidence(
                    original, transformed, self.parameter_names, float(self.attack.params["scale"]), self.tolerance
                )
                if _hash(original) != original_hash or _hash([_array(value) for value in parameters]) != original_hash:
                    raise AssertionError("evidence capture or attack mutated the original update")
                evidence.update({"original_pre_attack_update_preserved": True, "round_id": round_id, "update_id": update_id})
                self.events.append(evidence)
                self._cache[update_id] = [_array(value) for value in transformed]
                self._post_hash_to_update_id[evidence["post_attack_sha256"]] = update_id

            post_hash = _hash([_array(value) for value in transformed])
            expected_hash = _hash(self._cache[update_id])
            if post_hash != expected_hash:
                raise AssertionError("transmitted copies of an attacked update have different hashes")
            self.transmissions.append({"round_id": round_id, "update_id": update_id, "post_attack_sha256": post_hash})
            return [_array(value) for value in transformed]

    def on_attach(self, node: Any) -> None:
        """Forward node attachment to the preserved implementation."""
        self.attack.on_attach(node)
        if self._round_provider is None:
            self._round_provider = lambda: node.state.round
