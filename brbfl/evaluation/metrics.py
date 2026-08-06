"""Clean and genuine trigger-based backdoor evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MNISTTrigger:
    """A square at the bottom-right of a 28x28 image (default rows/cols 25-27)."""

    size: int = 3
    value: float = 1.0
    normalization_mean: float | None = None
    normalization_std: float | None = None

    def coordinates(self, height: int = 28, width: int = 28) -> list[list[int]]:
        """Return the exact ``[row, column]`` trigger coordinates."""
        return [[row, column] for row in range(height - self.size, height) for column in range(width - self.size, width)]

    @property
    def tensor_value(self) -> float:
        """Convert the raw configured value if input tensors are normalized."""
        if self.normalization_mean is None and self.normalization_std is None:
            return self.value
        if self.normalization_mean is None or not self.normalization_std:
            raise ValueError("normalization_mean and non-zero normalization_std must be configured together")
        return (self.value - self.normalization_mean) / self.normalization_std


def apply_mnist_trigger(images: Any, trigger: MNISTTrigger) -> Any:
    """Clone images and place the fixed trigger on 2D, 3D, or 4D tensors."""
    if images.ndim not in (2, 3, 4):
        raise ValueError(f"expected MNIST tensor with 2, 3, or 4 dimensions, got {images.ndim}")
    result = images.clone()
    result[..., -trigger.size :, -trigger.size :] = trigger.tensor_value
    return result


def triggered_asr_counts(
    predictions: Any, original_labels: Any, target_label: int, source_labels: tuple[int, ...] | None = None
) -> dict[str, float | int]:
    """Calculate all-to-one ASR, excluding original target-label examples."""
    eligible = original_labels != target_label
    if source_labels is not None:
        source_mask = eligible.clone().fill_(False)
        for label in source_labels:
            source_mask |= original_labels == label
        eligible &= source_mask
    count = int(eligible.sum().item())
    target_count = int(((predictions == target_label) & eligible).sum().item())
    return {
        "triggered_test_target_prediction_count": target_count,
        "eligible_triggered_examples": count,
        "triggered_test_asr": target_count / count if count else 0.0,
    }


def evaluate_batch(logits: Any, labels: Any, target_label: int) -> dict[str, float | int]:
    """Evaluate controlled logits using the genuine ASR formula."""
    return triggered_asr_counts(logits.argmax(dim=1), labels, target_label)


def backdoor_evaluation_configured(attack: object | None) -> bool:
    """Return whether an attack carries explicit trigger evaluation semantics."""
    return bool(
        attack
        and hasattr(attack, "target_class")
        and (hasattr(attack, "trigger") or all(hasattr(attack, field) for field in ("trigger_size", "trigger_value")))
    )
