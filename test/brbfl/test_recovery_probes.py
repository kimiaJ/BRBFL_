"""Recovery through finalized audit evidence, without aggregate contamination."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from brbfl.ca import CATransitionPolicy, ParticipantState
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig, clear_runtime_ledger, install_runtime_ledger
from brbfl.validation.byzantine_gate import (
    AdmissionPolicy,
    ValidatorSubgroupGate,
    clear_validator_gate,
    get_validator_gate,
    install_validator_gate,
    publish_admission_decision,
    wait_for_admitted_contributors,
)


@pytest.mark.parametrize('behavior', ['recover', 'reattack', 'attack', 'missing'])
def test_probe_recovery_lifecycle(behavior):
    """Only fresh positive probes allow reentry, and only in a later round."""
    nodes = tuple('abcdvw')
    trainers, validators = nodes[:4], nodes[4:]
    capability = 'model_update_detector'
    runtime = RuntimeLedgerAdapter(RuntimeLedgerConfig(
        enabled=True, trust_enabled=True, trust_observation_only=False, ca_enabled=True,
        selection_strategy='ca_contributors', validator_eligible_participants=validators,
        ca_minimum_contributors=3, detector_target_counts={capability: 2},
        detector_capabilities=dict.fromkeys(validators, (capability,)),
        ca_policy=CATransitionPolicy(evidence_bootstrap_rounds=3, recovery_enabled=True),
        recovery_probes_enabled=True,
    ), 'probe-integration', nodes, trainers, validators)
    install_runtime_ledger(runtime)
    install_validator_gate(ValidatorSubgroupGate(AdmissionPolicy(trainers, validators, quorum=2, acceptance_threshold=2)))
    parent = 'initial'
    try:
        for number in range(11):
            runtime.open_round(number, parent)
            selected = runtime.role_assignment(number).selected_contributors
            if number in range(2, 9):
                assert 'b' not in selected
            if behavior in ('recover', 'reattack') and number == 9:
                assert 'b' in selected
            if behavior in ('attack', 'missing') and number >= 2:
                assert 'b' not in selected
            if runtime.recovery_probe_subjects(number):
                with pytest.raises(RuntimeError, match='finalized admission'):
                    runtime.submit_recovery_probe(number, 'b', parent, [np.array([1., 2.])])
            attacking = number < 2 or behavior == 'reattack' and number >= 9
            models = {node: [np.array([1., 2.]) * (-3 if node == 'b' and attacking else 1)] for node in selected}
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                futures = {node: pool.submit(get_validator_gate(node).submit_and_decide,
                           number, node, model, current_node=node, parent_global_model_sha256=parent)
                           for node, model in models.items()}
                decisions = {node: task.result(timeout=5) for node, task in futures.items()}
            for node, yes in decisions.items():
                publish_admission_decision(number, node, yes)
                if yes:
                    get_validator_gate(node).observe_aggregation_input(number, node, models[node])
            admitted = wait_for_admitted_contributors(number, timeout=0)
            before = runtime.ledger.get_round_record(runtime.experiment_id, number)
            scheduled = runtime.recovery_probe_subjects(number)
            if number < 5:
                assert scheduled == ()
            if scheduled and behavior != 'missing':
                with pytest.raises(RuntimeError, match='current parent'):
                    runtime.submit_recovery_probe(number, 'b', 'stale', [np.array([1., 2.])])
                with pytest.raises(ValueError, match='scheduled'):
                    runtime.submit_recovery_probe(number, 'a', parent, [np.array([1., 2.])])
                if number > 5:
                    from brbfl.ca import EvidenceCategory
                    from brbfl.detectors.evidence import DetectorEvidence

                    previous = runtime.ledger.get_round_record(runtime.experiment_id, number - 1)
                    stale = []
                    for row in previous['recovery_probes']['b']['evidence']:
                        fields = dict(row)
                        fields.pop('evidence_hash')
                        fields['round_number'] = fields.pop('round')
                        fields['evidence_category'] = EvidenceCategory(fields['evidence_category'])
                        stale.append(DetectorEvidence(**fields))
                    with pytest.raises(ValueError, match='committee context'):
                        runtime.ledger.commit_recovery_probe(runtime.experiment_id, number, 'b', parent,
                                                             stale[0].source_artifact_hash, tuple(stale))
                probe = [np.array([1., 2.]) * (-3 if behavior == 'attack' else 1)]
                receipt = runtime.submit_recovery_probe(number, 'b', parent, probe)
                assert receipt['decision']['category'] == ('severe' if behavior == 'attack' else 'positive')
                count = len(runtime.ledger.events)
                assert runtime.submit_recovery_probe(number, 'b', parent, probe) == receipt
                assert len(runtime.ledger.events) == count
                with pytest.raises(RuntimeError, match='conflicting'):
                    runtime.submit_recovery_probe(number, 'b', parent, [np.array([4., 8.])])
                after = runtime.ledger.get_round_record(runtime.experiment_id, number)
                assert before['candidates'] == after['candidates']
                assert before['admission'] == after['admission']
                assert 'b' not in admitted
                # Probe evidence cannot alter CA before finalization.
                assert runtime.ca_snapshot_for_round(number).participant_states['b'].state is ParticipantState.EXCLUDED
            for node in nodes:
                runtime.confirm_installation(number, node, str(number) * 64)
            runtime.finalize_round(number)
            assert runtime.validation_artifact()['round_verification'][str(number)]
            if scheduled and behavior != 'missing':
                with pytest.raises(RuntimeError, match='open round'):
                    runtime.submit_recovery_probe(number, 'b', parent, probe)
            parent = str(number) * 64
        final = runtime.ca_snapshot_for_round(11).participant_states['b'].state
        assert final is (ParticipantState.TRUSTED if behavior == 'recover' else ParticipantState.EXCLUDED)
    finally:
        clear_validator_gate()
        clear_runtime_ledger()
