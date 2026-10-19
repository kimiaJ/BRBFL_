"""Observable reference construction for the process-local detector committee."""

from collections.abc import Mapping, Sequence

import numpy as np

from brbfl.canonical import canonical_hash
from brbfl.experiments.sign_flipping_evidence import canonical_parameter_hash


def median_reference(submissions: Mapping[str, Sequence[np.ndarray]]) -> tuple[tuple[np.ndarray, ...], str]:
    """
    Construct a coordinate median from the complete published candidate set.

    This policy assumes a strict honest majority among submitted contributors.
    It observes submitted model values, never pre-attack values or attack labels.
    The caller must independently enforce round membership and completeness.
    """
    if len(submissions) < 3:
        raise ValueError("median reference requires at least three submitted contributors")
    ordered = sorted(submissions)
    first = submissions[ordered[0]]
    if not first:
        raise ValueError("empty submitted model")
    for node in ordered:
        values = submissions[node]
        if len(values) != len(first):
            raise ValueError("submitted parameter counts differ")
        for left, right in zip(first, values, strict=True):
            if left.shape != right.shape or left.dtype != right.dtype or not np.isfinite(right).all():
                raise ValueError("submitted parameter schema differs or contains nonfinite values")
    reference = tuple(
        np.median(np.stack([submissions[node][index] for node in ordered]), axis=0).astype(value.dtype) for index, value in enumerate(first)
    )
    for value in reference:
        value.flags.writeable = False
    provenance = canonical_hash(
        "MedianModelReference/v1",
        {
            "policy": "coordinate_median_submitted_models",
            "submissions": {node: canonical_parameter_hash(submissions[node]) for node in ordered},
            "reference_hash": canonical_parameter_hash(reference),
            "assumption": "strict_honest_contributor_majority",
        },
    )
    return reference, provenance
