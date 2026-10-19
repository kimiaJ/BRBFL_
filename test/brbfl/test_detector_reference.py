"""Observable median reference tests, including attack and schema boundaries."""

import numpy as np
import pytest

from brbfl.ca import EvidenceCategory
from brbfl.detectors import SignFlippingDetector
from brbfl.detectors.reference import median_reference


def test_honest_majority_reference_detects_reversal_without_hidden_clean_update():
    """A reversed model does not dominate the observable coordinate median."""
    models = {"a": [np.array([1.0, 2.0])], "b": [np.array([1.1, 2.1])], "c": [np.array([-3.0, -6.0])]}
    reference, provenance = median_reference(models)
    other, other_provenance = median_reference(dict(reversed(tuple(models.items()))))
    assert provenance == other_provenance
    np.testing.assert_array_equal(reference[0], other[0])
    assert not reference[0].flags.writeable
    for node, expected in (("a", EvidenceCategory.POSITIVE), ("c", EvidenceCategory.SEVERE)):
        result = SignFlippingDetector("member").evaluate(
            experiment_id="exp",
            round_number=0,
            detector_assignment_hash="a" * 64,
            evaluated_participant_id=node,
            candidate_update=models[node],
            reference_update=reference,
        )
        assert result.evidence_category == expected
    assert models["c"][0][0] == -3


def test_reference_rejects_insufficient_nonfinite_and_incompatible_submissions():
    """Invalid reference input fails before committee evidence can be manufactured."""
    with pytest.raises(ValueError, match="three"):
        median_reference({"a": [np.ones(2)]})
    for invalid in (np.ones(3), np.array([np.nan, 1.0])):
        with pytest.raises(ValueError, match="schema"):
            median_reference({"a": [np.ones(2)], "b": [np.ones(2)], "c": [invalid]})
