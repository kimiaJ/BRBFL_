"""Focused tests for canonical admission and cross-role round coordination."""

import threading
import time

import pytest

from brbfl.validation.byzantine_gate import (
    AdmissionPolicy,
    ValidatorSubgroupGate,
    clear_validator_gate,
    install_validator_gate,
    publish_admission_decision,
    wait_at_round_barrier,
    wait_for_admitted_contributors,
)


@pytest.fixture(autouse=True)
def gate_registry():
    """Install an isolated five-node policy for each test."""
    policy = AdmissionPolicy(
        contributors=("node-0", "node-1", "node-2"),
        validators=("node-0", "node-1", "node-2", "node-3", "node-4"),
    )
    install_validator_gate(ValidatorSubgroupGate(policy))
    yield
    clear_validator_gate()


def test_every_node_observes_exact_canonical_admitted_set():
    """All participants read node-1/node-2 and never await rejected node-0."""
    publish_admission_decision(1, "node-2", True)
    publish_admission_decision(1, "node-0", False)
    publish_admission_decision(1, "node-1", True)

    observations = [wait_for_admitted_contributors(1, 0.1) for _ in range(5)]
    assert observations == [("node-1", "node-2")] * 5


def test_final_rejection_wakes_admission_wait_without_model_timeout():
    """The last final decision releases readiness immediately."""
    result = []
    thread = threading.Thread(target=lambda: result.append(wait_for_admitted_contributors(0, 1)))
    thread.start()
    publish_admission_decision(0, "node-1", True)
    publish_admission_decision(0, "node-2", True)
    assert thread.is_alive()
    publish_admission_decision(0, "node-0", False)
    thread.join(0.2)

    assert not thread.is_alive()
    assert result == [("node-1", "node-2")]


def test_round_barrier_blocks_next_round_until_all_five_complete():
    """No participant crosses the round boundary before all five arrive."""
    released = []
    threads = [
        threading.Thread(target=lambda node=node: (wait_at_round_barrier(0, node, 1), released.append(node)))
        for node in ("node-0", "node-1", "node-2", "node-3")
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.02)
    assert released == []

    wait_at_round_barrier(0, "node-4", 1)
    for thread in threads:
        thread.join(0.2)
    assert sorted(released) == ["node-0", "node-1", "node-2", "node-3"]
