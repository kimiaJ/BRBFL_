"""Compact, non-mutating evidence for model-update transformations."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    return np.asarray(value).copy()


def canonical_parameter_hash(values: Mapping[str, Any] | Sequence[Any], names: Sequence[str] | None = None) -> str:
    """Hash only canonical parameter content, independent of mapping/transport order."""
    if isinstance(values, Mapping):
        named_values = values.items()
    else:
        parameter_names = list(names) if names is not None else [str(index) for index in range(len(values))]
        if len(parameter_names) != len(values):
            raise ValueError("parameter names and values must have equal lengths")
        named_values = zip(parameter_names, values, strict=True)

    digest = hashlib.sha256()
    for name, value in sorted(named_values, key=lambda item: item[0]):
        # np.asarray(torch_tensor.detach().cpu()) supplies exact detached CPU
        # bytes for framework tensors while the copy isolates later mutation.
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        contiguous = np.ascontiguousarray(np.asarray(value).copy())
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
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
    if canonical_parameter_hash(original, names) == canonical_parameter_hash(transformed, names):
        raise AssertionError("sign-flipping pre/post hashes must differ")
    return {
        "formula": "attacked = scale * original",
        "scale": scale,
        "numerical_tolerance": tolerance,
        "pre_attack_sha256": canonical_parameter_hash(original, names),
        "post_attack_sha256": canonical_parameter_hash(transformed, names),
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
        self._original_snapshots: dict[str, list[np.ndarray]] = {}
        self._eligible: dict[str, dict[str, Any]] = {}
        # An attacked payload is only a retry within the round that produced it.
        # Scoping this index by round prevents a later locally trained update
        # with the same bytes from being mistaken for an earlier transmission.
        self._post_hash_to_update_id: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.hook_invocations: list[dict[str, Any]] = []
        self.transmissions: list[dict[str, Any]] = []
        self.event_trace: list[dict[str, Any]] = []
        self.node_id: str | None = None
        self._thread_state = threading.local()
        self._hook_sequence = 0
        self._transmission_sequence = 0

    def trace(self, event_type: str, **fields: Any) -> None:
        """Append one compact lifecycle record without producing console noise."""
        latest = next((event for event in reversed(self.events) if event["round_id"] == self._round_id()), None)
        transmissions = sum(item["round_id"] == self._round_id() for item in self.transmissions)
        self.event_trace.append(
            {
                "node_id": self.node_id,
                "framework_round": self._round_id(),
                "event_type": event_type,
                "update_id": fields.pop("update_id", latest["update_id"] if latest else None),
                "pre_attack_sha256": fields.pop("pre_attack_sha256", latest["pre_attack_sha256"] if latest else None),
                "post_attack_sha256": fields.pop("post_attack_sha256", latest["post_attack_sha256"] if latest else None),
                "recipients": fields.pop("recipients", []),
                "transmission_count": fields.pop("transmission_count", transmissions),
                **fields,
            }
        )

    def _round_id(self) -> str:
        value = self._round_provider() if self._round_provider is not None else None
        return "unassigned" if value is None else str(value)

    def manipulate_update(self, parameters: list[Any]) -> list[Any]:
        """Transform one copied update and prove the caller's input was not mutated."""
        original = [_array(value) for value in parameters]
        original_hash = canonical_parameter_hash(original, self.parameter_names)
        round_id = self._round_id()
        update_id = self._update_id(round_id, original_hash)
        with self._lock:
            logically_applied = False
            reused = False
            self._hook_sequence += 1
            hook_invocation_id = f"{self.node_id or 'unassigned'}:hook-{self._hook_sequence}"
            self.trace("sign_flipping_hook_entered", update_id=update_id, pre_attack_sha256=original_hash)
            # Defensive handling for a caller that passes our output back through
            # the hook.  It is a transmission, never a new mathematical attack.
            previously_attacked_id = self._post_hash_to_update_id.get((round_id, original_hash))
            if previously_attacked_id is not None:
                update_id = previously_attacked_id
                transformed = [value.copy() for value in self._cache[update_id]]
                reused = True
            elif update_id in self._cache:
                transformed = [value.copy() for value in self._cache[update_id]]
                reused = True
            elif update_id in self._eligible:
                snapshot = self._original_snapshots[update_id]
                transformed = self.attack.manipulate_update([value.copy() for value in snapshot])
                logically_applied = True
                evidence = transformation_evidence(
                    snapshot, transformed, self.parameter_names, float(self.attack.params["scale"]), self.tolerance
                )
                if (
                    canonical_parameter_hash(snapshot, self.parameter_names) != original_hash
                    or canonical_parameter_hash(parameters, self.parameter_names) != original_hash
                ):
                    raise AssertionError("evidence capture or attack mutated the original update")
                evidence.update(
                    {
                        "original_pre_attack_update_preserved": True,
                        "round_id": round_id,
                        "update_id": update_id,
                        "logical_application_count": 1,
                        "logical_application_id": f"{update_id}:application-1",
                    }
                )
                self.events.append(evidence)
                self._cache[update_id] = [_array(value) for value in transformed]
                self._post_hash_to_update_id[(round_id, evidence["post_attack_sha256"])] = update_id
                self.trace("sign_flipping_logically_applied", **evidence)
            else:
                # P2PFL serializes successive partial aggregates while gossiping.
                # They may contain this node's contribution, but are not new
                # locally trained updates and must not be scaled as a whole.
                transformed = original

            post_hash = canonical_parameter_hash(transformed, self.parameter_names)
            expected = self._cache.get(update_id)
            if expected is not None and post_hash != canonical_parameter_hash(expected, self.parameter_names):
                raise AssertionError("transmitted copies of an attacked update have different hashes")
            observation = {
                "round_id": round_id,
                "update_id": update_id if expected is not None else None,
                "pre_attack_sha256": self._eligible.get(update_id, {}).get("pre_attack_sha256"),
                "post_attack_sha256": post_hash,
                "transformed": logically_applied,
                "cached_result_reused": reused,
                "logical_application_id": (f"{update_id}:application-1" if expected is not None else None),
                "hook_invocation_id": hook_invocation_id,
            }
            self.hook_invocations.append(observation)
            self._thread_state.last_hook_observation = observation
            self.trace(
                "sign_flipping_hook_observed",
                update_id=observation["update_id"],
                pre_attack_sha256=observation["pre_attack_sha256"],
                post_attack_sha256=post_hash,
                transformed=logically_applied,
            )
            return [_array(value) for value in transformed]

    def record_update_created(self, parameters: list[Any]) -> None:
        """Register one locally trained update as eligible before serialization."""
        original = [_array(value) for value in parameters]
        original_hash = canonical_parameter_hash(original, self.parameter_names)
        update_id = self._update_id(self._round_id(), original_hash)
        self._eligible.setdefault(
            update_id,
            {
                "node_id": self.node_id,
                "round_id": self._round_id(),
                "update_id": update_id,
                "pre_attack_sha256": original_hash,
            },
        )
        self._original_snapshots.setdefault(update_id, [value.copy() for value in original])
        self.trace(
            "local_update_created",
            update_id=update_id,
            pre_attack_sha256=original_hash,
        )

    def record_transmission(self, recipient: str) -> None:
        """Associate a completed gossip serialization with its recipient."""
        observation = getattr(self._thread_state, "last_hook_observation", None)
        if observation is not None and observation["round_id"] == self._round_id():
            transmission_count = sum(item["round_id"] == self._round_id() for item in self.transmissions) + 1
            self._transmission_sequence += 1
            item = {
                **observation,
                "producer": self.node_id,
                "recipient": recipient,
                "transmission_sequence": self._transmission_sequence,
                "transmission_count": transmission_count,
            }
            self.transmissions.append(item)
            self.trace(
                "update_transmitted",
                update_id=item["update_id"],
                post_attack_sha256=item["post_attack_sha256"],
                recipients=[recipient],
                transmission_count=transmission_count,
            )
        else:
            # This is deliberately retained: it makes an eligible serialized
            # local update that bypassed the hook visible to the validator.
            self.transmissions.append(
                {
                    "round_id": self._round_id(),
                    "update_id": None,
                    "pre_attack_sha256": None,
                    "post_attack_sha256": None,
                    "recipient": recipient,
                    "transformed": False,
                    "transmission_count": len(self.transmissions) + 1,
                }
            )
            self.trace("update_transmitted", recipients=[recipient])

    def eligible_round_ids(self) -> set[str]:
        """Return rounds with local training and an attempted outbound update."""
        trained = {item["framework_round"] for item in self.event_trace if item["event_type"] == "local_training_completed"}
        transmitted = {item["framework_round"] for item in self.event_trace if item["event_type"] == "update_transmitted"}
        return trained & transmitted

    def eligible_update_ids(self) -> set[str]:
        """Return stable IDs for locally trained updates that were transmitted."""
        transmitted_rounds = {item["round_id"] for item in self.transmissions}
        return {update_id for update_id, item in self._eligible.items() if item["round_id"] in transmitted_rounds}

    def validate_eligible_updates(self) -> dict[str, int]:
        """Require one attack for every observed eligible local update."""
        eligible = self.eligible_update_ids()
        observed = {event["update_id"] for event in self.events}
        counts = {update_id: sum(event["update_id"] == update_id for event in self.events) for update_id in eligible | observed}
        invalid = {update_id: count for update_id, count in counts.items() if count != (1 if update_id in eligible else 0)}
        if invalid:
            raise AssertionError(f"eligible sign-flipping updates did not execute exactly once: {invalid}")
        for update_id in eligible:
            event = next(event for event in self.events if event["update_id"] == update_id)
            expected_hash = event["post_attack_sha256"]
            copies = [item for item in self.transmissions if item["update_id"] == update_id]
            transmitted_pre_hashes = {item["pre_attack_sha256"] for item in copies}
            if transmitted_pre_hashes != {event["pre_attack_sha256"]}:
                raise AssertionError(f"transmitted copies of eligible update {update_id} have inconsistent pre-attack hashes")
            transmitted_hashes = {item["post_attack_sha256"] for item in copies}
            if transmitted_hashes != {expected_hash}:
                raise AssertionError(f"transmitted copies of eligible update {update_id} have inconsistent post-attack hashes")
            application_id = event["logical_application_id"]
            if any(item["logical_application_id"] != application_id for item in copies):
                raise AssertionError(f"hook invocations for eligible update {update_id} refer to different applications")
        return {update_id: counts[update_id] for update_id in eligible}

    def _update_id(self, round_id: str, pre_attack_hash: str) -> str:
        """Build a semantic identity independent of objects and recipients."""
        return f"producer-{self.node_id or 'unassigned'}:round-{round_id}:{pre_attack_hash}"

    def participated_in_round(self, round_id: Any) -> bool:
        """Return the participant selection recorded for a framework round."""
        matches = [item for item in self.event_trace if item["event_type"] == "round_entered" and item["framework_round"] == str(round_id)]
        return bool(matches and matches[-1]["participating"])

    def evidence_for_round(self, round_id: Any) -> dict[str, Any]:
        """Return one logical update and its separate transmission count."""
        normalized_round = str(round_id)
        matching_events = [event for event in self.events if event["round_id"] == normalized_round]
        matching_transmissions = [item for item in self.transmissions if item["round_id"] == normalized_round]
        if len(matching_events) != 1:
            raise AssertionError(f"round {normalized_round} has {len(matching_events)} sign-flipping logical applications; expected one")
        evidence = dict(matching_events[0])
        evidence["transmission_count"] = len(matching_transmissions)
        return evidence

    def on_attach(self, node: Any) -> None:
        """Forward node attachment to the preserved implementation."""
        self.attack.on_attach(node)
        self.node_id = node.addr
        if self._round_provider is None:
            self._round_provider = lambda: node.state.round
