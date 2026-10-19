"""Concurrent model publication, committee decisions, and admission enforcement."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from brbfl.ca import ParticipantState
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig, clear_runtime_ledger, install_runtime_ledger
from brbfl.validation.byzantine_gate import AdmissionPolicy, ValidatorSubgroupGate, parameter_hash


def runtime():
    """Create a three-contributor runtime with two observing specialists."""
    ids = ("a", "b", "c")
    return RuntimeLedgerAdapter(
        RuntimeLedgerConfig(
            enabled=True,
            trust_enabled=True,
            ca_enabled=True,
            selection_strategy="ca_state",
            validator_eligible_participants=ids,
            detector_target_counts={"model_update_detector": 2},
            detector_capabilities={node: ("model_update_detector",) for node in ids},
        ),
        "exp",
        ids,
        ids,
        ("a", "b"),
    )


def test_concurrent_observable_models_reject_reversal_and_finalize_ca():
    """No candidate reaches admission before complete committee evaluation."""
    value = runtime()
    models = {"a": [np.array([1.0, 2.0])], "b": [np.array([1.1, 2.1])], "c": [np.array([-3.0, -6.0])]}
    for node, model in models.items():
        value.record_candidate(
            0,
            node,
            "parent",
            parameter_hash(model),
            [{"validator_node_id": validator, "reported_decision": True, "reference_decision": True} for validator in ("a", "b")],
        )
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            node: pool.submit(value.detect_model_submission, 0, node, model, publisher_id=node, timeout=2) for node, model in models.items()
        }
        decisions = {node: future.result() for node, future in futures.items()}
    assert decisions == {"a": True, "b": True, "c": False}
    value.finalize_admission(0, decisions)
    for node in models:
        value.confirm_installation(0, node, "aggregate")
    value.finalize_round(0)
    assert value.ca_snapshot(1).participant_states["c"].state == ParticipantState.SUSPICIOUS
    record = value.ledger.get_round_record("exp", 0)
    assert set(record["aggregate"]["contributor_hashes"]) == {"a", "b"}
    assert len(record["detector_committees"]) == 3
    assert value.ledger.verify_round("exp", 0)


def test_incomplete_publication_times_out_and_cannot_recover_silently():
    """Missing updates fail closed instead of selecting a partial reference."""
    value = runtime()
    model = [np.array([1.0, 2.0])]
    value.record_candidate(0, "a", "parent", parameter_hash(model), [])
    with pytest.raises(RuntimeError, match="timed out"):
        value.detect_model_submission(0, "a", model, publisher_id="a", timeout=0)
    with pytest.raises(RuntimeError, match="previously failed"):
        value.detect_model_submission(0, "a", model, publisher_id="a", timeout=0)


def test_production_gate_blocks_reversed_model_before_aggregation():
    """The actual gate consumes runtime committee decisions on candidate owners."""
    value = runtime()
    install_runtime_ledger(value)
    try:
        models = {"a": [np.array([1.0, 2.0])], "b": [np.array([1.1, 2.1])], "c": [np.array([-3.0, -6.0])]}
        gates = {
            node: ValidatorSubgroupGate(AdmissionPolicy(tuple(models), ("a", "b"), quorum=2, acceptance_threshold=2)) for node in models
        }
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                node: pool.submit(gates[node].submit_and_decide, 0, node, model, current_node=node, parent_global_model_sha256="parent")
                for node, model in models.items()
            }
            assert {node: future.result() for node, future in futures.items()} == {"a": True, "b": True, "c": False}
        with pytest.raises(RuntimeError, match="rejected candidate"):
            gates["c"].observe_aggregation_input(0, "c", models["c"])
    finally:
        clear_runtime_ledger()
