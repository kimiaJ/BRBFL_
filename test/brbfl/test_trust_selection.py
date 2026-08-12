# ruff: noqa: D103
"""Focused deterministic trust-ranked validator selection tests."""

import pytest

from brbfl.selection import SelectionContext, StaticRoundRoleSelector, TrustRankedValidatorSelector

CAPS = {f"node-{i}": frozenset({"contributor", "validator", "aggregator"}) for i in range(5)}


def selector():
    return TrustRankedValidatorSelector(
        StaticRoundRoleSelector(("node-0", "node-1", "node-2"), ("node-0", "node-3", "node-4"), ("node-0", "node-1", "node-2")),
        tuple(CAPS),
        3,
    )


def test_bootstrap_and_next_round_use_only_prior_finalized_trust():
    value = selector()
    first = value.select_roles(SelectionContext("x", 0, CAPS, trust_scores={n: 0.99 for n in CAPS}))
    assert first.selected_validators == ("node-0", "node-3", "node-4")
    scores = {"node-0": 0.8, "node-1": 0.5, "node-2": 0.5, "node-3": 0.2, "node-4": 0.2}
    second = value.select_roles(SelectionContext("x", 1, CAPS, "finalized-round-0", scores))
    assert second.selected_validators == ("node-0", "node-1", "node-2")
    assert second.selected_contributors == first.selected_contributors
    assert second.aggregation_eligible_nodes == first.aggregation_eligible_nodes


def test_ranking_is_order_independent_and_fail_closed():
    scores = dict(reversed(list({"node-0": 0.8, "node-1": 0.5, "node-2": 0.5, "node-3": 0.2, "node-4": 0.2}.items())))
    value = selector()
    value.select_roles(SelectionContext("x", 0, CAPS))
    assert value.select_roles(SelectionContext("x", 1, CAPS, "h", scores)).selected_validators == ("node-0", "node-1", "node-2")
    value = TrustRankedValidatorSelector(StaticRoundRoleSelector((), ("node-0",), ()), tuple(CAPS), 1, minimum_trust=0.9)
    value.select_roles(SelectionContext("x", 0, CAPS))
    with pytest.raises(RuntimeError, match="insufficient"):
        value.select_roles(SelectionContext("x", 1, CAPS, "h", {n: 0.5 for n in CAPS}))
