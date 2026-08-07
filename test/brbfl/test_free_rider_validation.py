"""Focused free-rider lifecycle and evidence tests."""

import numpy as np
import pytest

from brbfl.attacks.free_rider import FreeRiderAttack
from brbfl.attacks.registry import create_attack
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
    audit.publish_update(values)
    assert audit.evidence_for_round(0)["optimizer_step_count"] == 1
    assert audit.evidence_for_round(0)["free_rider_attack_application_count"] == 0


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
