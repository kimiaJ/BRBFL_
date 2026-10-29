"""Focused workflow failure and scheduled-publication checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from brbfl.ca.probe_workflow import recovery_probe_barrier
from brbfl.experiments.sign_flipping_evidence import AuditedModelUpdateAttack
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig


def test_probe_timeout_poisoned_phase_rejects_late_submission(monkeypatch):
    """A timeout cannot release another waiting publisher successfully."""
    runtime = RuntimeLedgerAdapter(RuntimeLedgerConfig(), 'timeout', ('a',), ('a',), ('a',))
    monkeypatch.setattr(runtime, 'recovery_probe_subjects', lambda r: ('a',))
    with pytest.raises(RuntimeError, match='phase failed'):
        runtime.wait_for_recovery_probes(0, 0)
    with pytest.raises(RuntimeError, match='phase failed'):
        runtime.submit_recovery_probe(0, 'a', 'parent', [np.array([1.])])
    with pytest.raises(RuntimeError, match='phase failed'):
        runtime.wait_for_recovery_probes(0, 0)


def test_failed_fit_restores_parent_and_signals_waiters(monkeypatch):
    """Training exceptions cannot leave a local probe model installed."""
    errors = []
    runtime = SimpleNamespace(config=SimpleNamespace(recovery_probes_enabled=True),
                              recovery_probe_subjects=lambda r: ('a',),
                              fail_recovery_probes=lambda r, error: errors.append(error))
    monkeypatch.setattr('brbfl.ca.probe_workflow.get_runtime_ledger', lambda: runtime)

    class Model:
        def __init__(self):
            self.values = [np.array([1.])]

        def get_parameters(self):
            return self.values

    class Learner:
        def __init__(self):
            self.model = Model()

        def get_model(self):
            return self.model

        def set_model(self, model):
            self.model = model

        def fit(self):
            self.model.values[0][0] = 99
            raise RuntimeError('fit failed')

    learner = Learner()
    state = SimpleNamespace(addr='a', round=0, trainer_role_evidence={0: {}})
    with pytest.raises(RuntimeError, match='fit failed'):
        recovery_probe_barrier(state, learner, 1)
    assert learner.get_model().get_parameters()[0][0] == 1
    assert len(errors) == 1


def test_attack_schedule_leaves_inactive_publication_benign():
    """Transient behavior changes only the declared attack rounds."""
    attack = SimpleNamespace(params={'scale': -3}, manipulate_update=lambda values: [v * -3 for v in values])
    current = [0]
    audit = AuditedModelUpdateAttack(attack, ['weight'], round_provider=lambda: current[0])
    audit.active_rounds = [0, 1]
    values = [np.array([1., 2.])]
    assert np.array_equal(audit.publish_update(values)[0], [-3., -6.])
    audit.observe_aggregation([-3 * values[0]])
    current[0] = 2
    assert np.array_equal(audit.publish_update(values)[0], values[0])
    assert len(audit.events) == 1
    audit.validate_eligible_updates()
