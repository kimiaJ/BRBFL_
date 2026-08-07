"""Focused free-rider lifecycle and evidence tests."""

import numpy as np
import pytest

from brbfl.attacks.free_rider import FreeRiderAttack
from brbfl.attacks.registry import clear_attacks, create_attack, get_attack, register_attack
from brbfl.experiments.config import load_experiment_config
from brbfl.experiments.free_rider_evidence import TrainingLifecycleAudit, parameter_delta


def test_construction_and_controlled_configs():
    """Construct only the explicit strategy and verify paired controls."""
    attack = create_attack("free_rider", {"strategy": "no_training_stale_current_model"})
    clean = load_experiment_config("configs/smoke/mnist_free_rider_clean.yaml")
    attacked = load_experiment_config("configs/smoke/mnist_free_rider.yaml")
    assert isinstance(attack, FreeRiderAttack)
    assert clean.attack.name == "none" and clean.attack.adversaries == ()
    assert attacked.attack.adversaries == (1,)
    assert clean.nodes == attacked.nodes == 3 and clean.rounds == attacked.rounds == 2
    assert clean.seed == attacked.seed and clean.dataset == attacked.dataset


def test_stale_current_model_is_skipped_and_reaches_aggregation():
    """Prove zero training, unchanged publication, and aggregation receipt."""
    audit = TrainingLifecycleAudit(FreeRiderAttack(), configured_epochs=3, configured_batch_count=4)
    start = [np.array([1.0, 2.0], dtype=np.float32)]
    audit.begin_local_training(0, start)
    assert audit.should_skip_local_training()
    audit.complete_local_training(start, skipped=True)
    submitted = audit.publish_update(start)
    audit.record_submission(submitted, 0)
    audit.observe_aggregation(submitted)
    audit.observe_global_model(submitted)
    row = audit.evidence_for_round(0)
    assert row["optimizer_step_count"] == row["local_epochs_actually_executed"] == 0
    assert row["free_rider_attack_application_count"] == 1
    assert row["pre_training_model_sha256"] == row["submitted_model_sha256"]
    assert row["pre_to_submission_delta_l2_norm"] == row["maximum_absolute_delta"] == 0.0
    assert row["all_submitted_parameters_equal_pre_training"]
    assert row["aggregation_matches_submitted_snapshot"]


def test_benign_audit_counts_training_without_attack_application():
    """Keep benign training active without applying free-rider control."""
    audit = TrainingLifecycleAudit(None, configured_epochs=1)
    values = [np.array([1.0])]
    audit.begin_local_training(0, values)
    assert not audit.should_skip_local_training()
    audit.record_optimizer_step()
    audit.complete_local_training(values, skipped=False)
    submitted = audit.publish_update(values)
    audit.record_submission(submitted, 0)
    audit.observe_aggregation(submitted)
    audit.observe_global_model(submitted)
    assert audit.evidence_for_round(0)["optimizer_step_count"] == 1
    assert audit.evidence_for_round(0)["free_rider_attack_application_count"] == 0


def test_round_keys_are_integer_and_evidence_requires_finalization():
    """Use one key representation and reject incomplete or absent evidence clearly."""
    audit = TrainingLifecycleAudit(None, configured_epochs=1)
    audit.node_id = "node-2"
    with pytest.raises(RuntimeError, match=r"node=node-2.*available_round_keys=\[\].*initialized=False"):
        audit.evidence_for_round("0")

    values = [np.array([1.0])]
    audit.begin_local_training("0", values)
    assert list(audit.rounds) == [0]
    with pytest.raises(RuntimeError, match=r"initialized=True, finalized=False"):
        audit.evidence_for_round(0)
    with pytest.raises(RuntimeError, match="already initialized"):
        audit.begin_local_training(0, values)

    assert not audit.should_skip_local_training()
    audit.record_optimizer_steps(1)
    audit.complete_local_training(values, skipped=False)
    submitted = audit.publish_update(values)
    audit.record_submission(submitted, 0)
    audit.observe_aggregation(submitted)
    audit.observe_global_model(submitted)
    assert audit.evidence_for_round("0")["record_finalized"]
    with pytest.raises(RuntimeError, match="already finalized"):
        audit.observe_global_model(submitted)


def test_all_clean_nodes_finalize_two_rounds_without_fabricating_absence():
    """Exercise the paired clean run's three-node, two-round lifecycle."""
    audits = [TrainingLifecycleAudit(None, 1, 1) for _ in range(3)]
    for node_index, audit in enumerate(audits):
        audit.node_id = f"node-{node_index}"
        for round_id in range(2):
            before = [np.array([float(round_id)])]
            after = [np.array([float(round_id + 1)])]
            audit.begin_local_training(round_id, before)
            assert not audit.should_skip_local_training()
            audit.record_optimizer_steps(1)
            audit.complete_local_training(after, skipped=False)
            submitted = audit.publish_update(after)
            audit.record_submission(submitted, round_id)
            audit.observe_aggregation(submitted)
            audit.observe_global_model(submitted)

    assert all(set(audit.rounds) == {0, 1} for audit in audits)
    assert all(audit.evidence_for_round(r)["optimizer_step_count"] == 1 for audit in audits for r in range(2))

    absent = TrainingLifecycleAudit(None, 1, 1)
    with pytest.raises(RuntimeError, match="initialized=False"):
        absent.evidence_for_round(0)


def test_falsey_clean_audit_is_the_same_instance_used_by_training_stage_registry():
    """Guard the clean-path registration condition that caused missing round zero."""
    clear_attacks()
    audit = TrainingLifecycleAudit(None, 1, 1)
    assert not audit
    register_attack("node-0", audit)
    assert get_attack("node-0") is audit
    clear_attacks()


def test_lifecycle_order_errors_are_diagnostic_and_duplicates_fail():
    """Reject reordered and duplicate callbacks with actionable node evidence."""
    audit = TrainingLifecycleAudit(None, 1, 1)
    audit.node_id = "node-2"
    values = [np.array([1.0])]
    audit.begin_local_training(0, values)
    assert not audit.should_skip_local_training()
    audit.complete_local_training(values, skipped=False)
    with pytest.raises(
        RuntimeError,
        match=r"aggregation observed before submission.*node=node-2.*round=0.*training_completed=True.*submission_recorded=False",
    ):
        audit.observe_aggregation(values)
    audit.record_submission(values, 0)
    with pytest.raises(RuntimeError, match=r"submission already recorded.*node=node-2"):
        audit.record_submission(values, 0)
    audit.observe_aggregation(values)
    with pytest.raises(RuntimeError, match=r"aggregation already observed.*node=node-2"):
        audit.observe_aggregation(values)


def test_submission_evidence_is_immune_to_later_in_place_mutation():
    """Keep publication and aggregation as distinct immutable observations."""
    audit = TrainingLifecycleAudit(None, 1, 1)
    live = [np.array([2.0], dtype=np.float32)]
    audit.begin_local_training(0, live)
    assert not audit.should_skip_local_training()
    audit.complete_local_training(live, skipped=False)
    submitted = audit.publish_update(live)
    audit.record_submission(submitted, 0)
    recorded_hash = audit.rounds[0]["submitted_model_sha256"]
    submitted[0][0] = 99.0
    assert audit.rounds[0]["submitted_model_sha256"] == recorded_hash
    audit.observe_aggregation([np.array([2.0], dtype=np.float32)])
    assert audit.rounds[0]["aggregation_matches_submitted_snapshot"]
    assert audit.rounds[0]["aggregation_input_numerically_equals_submission"]


def test_attacked_and_benign_nodes_finalize_both_rounds():
    """Exercise both controlled-run branches across every participant and round."""
    audits = [TrainingLifecycleAudit(FreeRiderAttack() if node == 1 else None, 1, 1) for node in range(3)]
    for node, audit in enumerate(audits):
        audit.node_id = f"node-{node}"
        for round_id in range(2):
            before = [np.array([float(round_id)])]
            audit.begin_local_training(round_id, before)
            skipped = audit.should_skip_local_training()
            after = before if skipped else [before[0] + 1]
            audit.record_optimizer_steps(0 if skipped else 1)
            audit.complete_local_training(after, skipped)
            submitted = audit.publish_update(after)
            audit.record_submission(submitted, round_id)
            audit.observe_aggregation(submitted)
            audit.observe_global_model(submitted)
    assert all(set(audit.rounds) == {0, 1} for audit in audits)
    assert all(audits[1].evidence_for_round(r)["attack_application_count"] == 1 for r in range(2))
    assert all(
        audits[1].evidence_for_round(r)["pre_training_model_sha256"] == audits[1].evidence_for_round(r)["submitted_model_sha256"]
        for r in range(2)
    )


def test_zero_delta_tolerance_behavior():
    """Make exact and strict-tolerance assertions explicit."""
    before = [np.array([1.0])]
    after = [np.array([1.0 + 1e-8])]
    assert not parameter_delta(before, after, tolerance=0.0)["zero_delta_within_tolerance"]
    assert parameter_delta(before, after, tolerance=1e-7)["zero_delta_within_tolerance"]


def test_unknown_free_rider_strategy_rejected():
    """Reject random and legacy false free-rider variants."""
    with pytest.raises(ValueError, match="unsupported"):
        FreeRiderAttack("random")
