# ruff: noqa: D103
"""Focused mathematical and deterministic trust tests."""

import pytest

from brbfl.trust import TrustRuntime


def evidence(round_id, validators, candidates, agreement=True):
    return [
        {
            "validator_id": validator,
            "candidate_id": candidate,
            "reported_decision": agreement,
            "reference_decision": True,
        }
        for validator in validators
        for candidate in candidates
    ]


def test_prior_and_agreement_disagreement_math():
    validators = ("honest", "byzantine")
    runtime = TrustRuntime("experiment", validators)
    assert runtime.states["honest"].score == 0.5
    rows = evidence(0, ("honest",), ("a", "b", "c")) + evidence(0, ("byzantine",), ("a", "b", "c"), False)
    runtime.finalize_round(0, validators, ("a", "b", "c"), rows)
    assert runtime.states["honest"].artifact() | {} == {
        "alpha": 4.0, "beta": 1.0, "score": 0.8, "agreement_count": 3,
        "disagreement_count": 0, "processed_vote_count": 3, "last_finalized_round": 0,
    }
    assert runtime.states["byzantine"].score == 0.2


def test_mixed_evidence_and_duplicate_is_idempotent():
    runtime = TrustRuntime("experiment", ("v",))
    rows = evidence(0, ("v",), ("a",))
    rows += rows
    rows += evidence(0, ("v",), ("b",), False)
    runtime.finalize_round(0, ("v",), ("a", "b"), rows)
    assert runtime.states["v"].score == 0.5
    assert runtime.states["v"].processed_vote_count == 2


def test_conflict_incomplete_unknown_and_malformed_fail_closed():
    runtime = TrustRuntime("experiment", ("v",))
    duplicate = evidence(0, ("v",), ("a",))[0]
    conflict = dict(duplicate, reported_decision=False)
    with pytest.raises(RuntimeError, match="conflicting"):
        runtime.finalize_round(0, ("v",), ("a",), (duplicate, conflict))
    with pytest.raises(RuntimeError, match="incomplete"):
        runtime.finalize_round(0, ("v",), ("a",), ())
    with pytest.raises(RuntimeError, match="unknown"):
        runtime.finalize_round(0, ("v",), ("a",), (dict(duplicate, validator_id="x"),))
    with pytest.raises(ValueError, match="booleans"):
        runtime.finalize_round(0, ("v",), ("a",), (dict(duplicate, reported_decision=1),))


def test_order_independent_hash_accumulation_and_round_guards():
    rows = evidence(0, ("a", "b"), ("x", "y"))
    left = TrustRuntime("same", ("a", "b"))
    right = TrustRuntime("same", ("a", "b"))
    one = left.finalize_round(0, ("a", "b"), ("x", "y"), rows)
    two = right.finalize_round(0, ("a", "b"), ("x", "y"), reversed(rows))
    assert one.snapshot_sha256 == two.snapshot_sha256
    pre = left.states["a"]
    left.finalize_round(1, ("a", "b"), ("x",), evidence(1, ("a", "b"), ("x",)))
    assert left.snapshots[1].pre_round["a"] == pre
    assert left.states["a"].processed_vote_count == 3
    with pytest.raises(RuntimeError, match="already"):
        left.finalize_round(1, ("a", "b"), ("x",), evidence(1, ("a", "b"), ("x",)))
    with pytest.raises(RuntimeError, match="consecutively"):
        TrustRuntime("gap", ("a",)).finalize_round(1, ("a",), ("x",), evidence(1, ("a",), ("x",)))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_priors(value):
    with pytest.raises(ValueError, match="priors"):
        TrustRuntime("experiment", ("v",), value, 1)


def test_experiment_isolation_and_immutable_views():
    left, right = TrustRuntime("left", ("v",)), TrustRuntime("right", ("v",))
    left.finalize_round(0, ("v",), ("a",), evidence(0, ("v",), ("a",)))
    assert right.states["v"].score == 0.5
    with pytest.raises(TypeError):
        left.states["v"] = right.states["v"]
