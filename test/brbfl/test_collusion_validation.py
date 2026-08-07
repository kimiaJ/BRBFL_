"""Focused coordinated-collusion tests."""
# ruff: noqa: D103

import numpy as np
import pytest

from brbfl.attacks.collusion import CollusionAttack
from brbfl.attacks.registry import create_attack
from brbfl.experiments.collusion_evidence import CollusionLifecycleAudit, cosine, delta
from brbfl.experiments.config import load_experiment_config


def attack():
    return CollusionAttack("g", [1, 2], 42, 1.0)


def test_configuration_and_registry():
    clean = load_experiment_config("configs/smoke/mnist_collusion_clean.yaml")
    attacked = load_experiment_config("configs/smoke/mnist_collusion.yaml")
    assert clean.attack.name == "none" and clean.attack.adversaries == ()
    assert attacked.attack.adversaries == (1, 2)
    assert isinstance(create_attack(attacked.attack.name, attacked.attack.parameters), CollusionAttack)
    assert clean.nodes == attacked.nodes == 3 and clean.rounds == attacked.rounds == 2
    assert clean.seed == attacked.seed and clean.dataset == attacked.dataset


def test_shared_direction_is_deterministic_shared_and_round_specific():
    values = [np.zeros((2, 3), dtype=np.float32)]
    a, b = attack(), attack()
    assert all(np.array_equal(x, y) for x, y in zip(a.shared_direction(values, 0), b.shared_direction(values, 0), strict=True))
    assert not np.array_equal(a.shared_direction(values, 0)[0], a.shared_direction(values, 1)[0])


def complete(audit, before, trained):
    audit.begin_local_training(0, before)
    audit.record_optimizer_steps(2)
    audit.complete_local_training(trained, False)
    submitted = audit.publish_update(trained)
    audit.record_submission(submitted, 0)
    audit.observe_aggregation(submitted)
    audit.observe_global_model(submitted)
    return submitted


def test_genuine_training_then_aligned_poisoning_and_receipt():
    before = [np.array([1.0, 2.0], dtype=np.float32)]
    left = CollusionLifecycleAudit(attack(), "node-1", 1, 2)
    right = CollusionLifecycleAudit(attack(), "node-2", 1, 2)
    ls = complete(left, before, [np.array([2.0, 2.5], dtype=np.float32)])
    rs = complete(right, before, [np.array([0.5, 4.0], dtype=np.float32)])
    assert cosine(delta(before, ls), delta(before, rs)) == pytest.approx(1.0, abs=1e-6)
    for audit in (left, right):
        row = audit.evidence_for_round(0)
        assert row["optimizer_step_count"] == 2 and row["attack_application_count"] == 1
        assert row["submitted_update_l2_norm"] == pytest.approx(row["genuine_update_l2_norm"], rel=1e-6)
        assert row["aggregation_receipt"] and row["submitted_shared_direction_cosine_similarity"] == pytest.approx(1, abs=1e-6)


def test_zero_update_cosine_and_immutable_snapshot():
    assert cosine([np.zeros(2)], [np.ones(2)]) == 0.0
    before = [np.ones(2, dtype=np.float32)]
    audit = CollusionLifecycleAudit(attack(), "node-1", 1, 1)
    submitted = complete(audit, before, before)
    digest = audit.evidence_for_round(0)["submitted_model_sha256"]
    submitted[0][0] = 99
    assert audit.evidence_for_round(0)["submitted_model_sha256"] == digest
    assert audit.evidence_for_round(0)["submitted_update_l2_norm"] == 0


@pytest.mark.parametrize("alpha", [-1, float("nan"), float("inf")])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CollusionAttack("g", [1, 2], 1, alpha)


def test_invalid_groups_rejected():
    with pytest.raises(ValueError, match="group_id"):
        CollusionAttack("", [1, 2], 1)
    with pytest.raises(ValueError, match="at least two"):
        CollusionAttack("g", [1], 1)


def test_benign_never_applies_attack():
    audit = CollusionLifecycleAudit(None, "node-0", 1, 1)
    complete(audit, [np.zeros(1)], [np.ones(1)])
    assert audit.evidence_for_round(0)["attack_application_count"] == 0
