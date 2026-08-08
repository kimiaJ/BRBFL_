# ruff: noqa: D103
"""Focused round-role selection tests."""  # noqa: D103

import pytest

from brbfl.selection.roles import RoundRoleAssignment, SelectionContext, StaticRoundRoleSelector

CAPS = {
    "a": frozenset({"contributor", "validator", "aggregator"}),
    "b": frozenset({"contributor", "validator"}),
    "v": frozenset({"contributor", "validator"}),
}


def test_assignment_hash_and_set_order_are_deterministic():
    left = RoundRoleAssignment("x", 0, ("v", "a", "b"), ("b", "a"), ("v", "a"), aggregation_eligible_nodes=("a",))
    right = RoundRoleAssignment("x", 0, ("a", "b", "v"), ("a", "b"), ("a", "v"), aggregation_eligible_nodes=("a",))
    assert left == right
    assert left.assignment_hash == right.assignment_hash
    assert left.verify_hash()
    assert all(role.round_number == 0 for role in left.roles())


def test_static_selection_preserves_roles_and_allows_later_change():
    context = SelectionContext("x", 0, CAPS)
    selector = StaticRoundRoleSelector(("a", "b"), ("a", "v"), ("a",))
    assert selector.select_roles(context).selected_contributors == ("a", "b")
    later = StaticRoundRoleSelector(("v",), ("b",), ("a",)).select_roles(SelectionContext("x", 1, CAPS, "previous"))
    assert later.selected_contributors == ("v",)
    assert later.previous_state_hash == "previous"


def test_capability_and_membership_validation():
    with pytest.raises(ValueError, match="unknown"):
        StaticRoundRoleSelector(("missing",), ()).select_roles(SelectionContext("x", 0, CAPS))
    limited = {"a": frozenset({"validator"})}
    with pytest.raises(ValueError, match="exceed"):
        StaticRoundRoleSelector(("a",), ()).select_roles(SelectionContext("x", 0, limited))
