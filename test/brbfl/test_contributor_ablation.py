"""Controlled trainer selection through the production detector and ledger boundaries."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from brbfl.ca import CATransitionPolicy, ParticipantState
from brbfl.ledger.runtime import RuntimeLedgerAdapter, RuntimeLedgerConfig, clear_runtime_ledger, install_runtime_ledger
from brbfl.selection import SelectionContext, StaticRoundRoleSelector, TrustRankedContributorSelector
from brbfl.validation.byzantine_gate import (
    AdmissionPolicy,
    ValidatorSubgroupGate,
    clear_validator_gate,
    get_validator_gate,
    install_validator_gate,
    publish_admission_decision,
    wait_for_admitted_contributors,
)

NODES = tuple("abcdvw")
TRAINERS, VALIDATORS = NODES[:4], NODES[4:]
CAPABILITY = "model_update_detector"


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("strategy", ["static", "trust_contributors", "ca_contributors"])
@pytest.mark.parametrize("attacked", [False, True])
def test_three_rounds_keep_sensitive_roles_and_detector_fixed(strategy, attacked, bootstrap):
    """All modes reject the same malicious model but may stop selecting its owner at different rounds."""
    runtime = RuntimeLedgerAdapter(
        RuntimeLedgerConfig(
            enabled=True,
            trust_enabled=True,
            trust_observation_only=False,
            selection_strategy=strategy,
            ca_enabled=strategy == "ca_contributors",
            ca_policy=CATransitionPolicy(evidence_bootstrap_rounds=3 if bootstrap else 0),
            validator_eligible_participants=VALIDATORS,
            contributor_minimum_count=3,
            ca_minimum_contributors=3,
            detector_target_counts={CAPABILITY: 2},
            detector_capabilities=dict.fromkeys(VALIDATORS, (CAPABILITY,)),
        ),
        "ablation",
        NODES,
        TRAINERS,
        VALIDATORS,
    )
    install_runtime_ledger(runtime)
    install_validator_gate(ValidatorSubgroupGate(AdmissionPolicy(TRAINERS, VALIDATORS, quorum=2, acceptance_threshold=2)))
    selections = []
    try:
        parent = "initial"
        for number in range(3):
            runtime.open_round(number, parent)
            assignment = runtime.role_assignment(number)
            selected = assignment.selected_contributors
            selections.append(selected)
            assert assignment.selected_validators == VALIDATORS
            assert assignment.detector_subgroups[CAPABILITY] == VALIDATORS
            if number and strategy == "trust_contributors":
                assert assignment.source_trust_round == number - 1
                assert assignment.source_trust_hash == runtime._trust.snapshots[number - 1].snapshot_sha256
            models = {node: [np.array([1.0, 2.0]) * (-3 if attacked and node == "b" else 1)] for node in selected}
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                futures = {
                    node: pool.submit(
                        get_validator_gate(node).submit_and_decide,
                        number,
                        node,
                        model,
                        current_node=node,
                        parent_global_model_sha256=parent,
                    )
                    for node, model in models.items()
                }
                decisions = {node: future.result(timeout=5) for node, future in futures.items()}
            for node, admitted in decisions.items():
                assert admitted == (not attacked or node != "b")
                publish_admission_decision(number, node, admitted)
                if admitted:
                    get_validator_gate(node).observe_aggregation_input(number, node, models[node])
            admitted = wait_for_admitted_contributors(number, timeout=0)
            assert set(admitted) == set(selected) - ({"b"} if attacked else set())
            parent = str(number) * 64
            for node in NODES:
                runtime.confirm_installation(number, node, parent)
            runtime.finalize_round(number)
            assert runtime.validation_artifact()["round_verification"][str(number)]
        if strategy == "ca_contributors":
            snapshot = runtime.ca_snapshot_for_round(3)
            for node in NODES:
                expected_state = (ParticipantState.EXCLUDED if attacked and node == "b" else
                                  ParticipantState.TRUSTED if bootstrap else ParticipantState.OBSERVATION)
                assert snapshot.participant_states[node].state is expected_state
            runtime.open_round(3, parent)
            assert runtime.role_assignment(3).source_ca_snapshot_hash == snapshot.snapshot_hash
        expected = [True, True, True]
        if attacked and strategy == "trust_contributors":
            expected = [True, False, False]
        elif attacked and strategy == "ca_contributors":
            expected = [True, True, False]
        assert [("b" in selected) for selected in selections] == expected
        assert (runtime.ca_artifact() is not None) == (strategy == "ca_contributors")
    finally:
        clear_validator_gate()
        clear_runtime_ledger()


def trust_context():
    """Provide finalized provenance without CA input."""
    caps = {node: frozenset({"contributor", "aggregator", "validator"}) for node in TRAINERS}
    return SelectionContext(
        "exp", 1, caps, "aggregate-0", {"a": 0.8, "b": 0.5, "c": 0.6, "d": 0.2}, source_trust_round=0, source_trust_hash="trust-0"
    )


def trust_selector():
    """Use a fixed four-trainer pool and two fixed validators."""
    return TrustRankedContributorSelector(StaticRoundRoleSelector(TRAINERS, ("a", "b"), TRAINERS))


def test_threshold_is_inclusive_and_selection_is_order_independent():
    """Boundary scores are eligible; dictionary iteration cannot alter the assignment hash."""
    ctx = trust_context()
    left = trust_selector().select_roles(ctx)
    right = trust_selector().select_roles(replace(ctx, trust_scores=dict(reversed(tuple(ctx.trust_scores.items())))))
    assert left.selected_contributors == ("a", "b", "c")
    assert left.selected_validators == ("a", "b")
    assert left.aggregation_eligible_nodes == ("a", "b", "c")
    assert left.assignment_hash == right.assignment_hash


@pytest.mark.parametrize(
    "change",
    [
        {"source_trust_round": 1},
        {"source_trust_hash": None},
        {"previous_state_hash": None},
        {"trust_scores": {"a": 0.9}},
        {"trust_scores": {"a": float("nan"), "b": 0.9, "c": 0.9, "d": 0.9}},
        {"trust_scores": {"a": True, "b": 0.9, "c": 0.9, "d": 0.9}},
    ],
)
def test_missing_or_stale_provenance_and_invalid_scores_fail_closed(change):
    """No selection can use incomplete or non-finalized trust input."""
    with pytest.raises(RuntimeError):
        trust_selector().select_roles(replace(trust_context(), **change))


def test_shortage_never_readmits_low_trust_trainers():
    """The minimum model count is enforced without silently relaxing the threshold."""
    ctx = replace(trust_context(), trust_scores={"a": 0.8, "b": 0.8, "c": 0.1, "d": 0.1})
    with pytest.raises(RuntimeError, match="insufficient"):
        trust_selector().select_roles(ctx)


def test_nonfinalized_runtime_round_cannot_supply_trust():
    """An opened but unfinished round is not a trust source for selection."""
    runtime = RuntimeLedgerAdapter(
        RuntimeLedgerConfig(
            enabled=True,
            trust_enabled=True,
            trust_observation_only=False,
            selection_strategy="trust_contributors",
            validator_eligible_participants=VALIDATORS,
        ),
        "unfinished",
        NODES,
        TRAINERS,
        VALIDATORS,
    )
    runtime.open_round(0, "initial")
    with pytest.raises(RuntimeError, match="provenance"):
        runtime.open_round(1, "not-finalized")


def test_fixed_ca_sensitive_roles_ignore_ranking_changes():
    """The ablation freezes committee identities even if another specialist ranks higher."""
    from brbfl.ca import CATransitionEngine, CATransitionInput, CATransitionPolicy, EvidenceCategory
    from brbfl.selection import CARoleSelectionPolicy, CAStateRoleSelector

    caps = {node: frozenset({"contributor", "aggregator", "validator", "detector", CAPABILITY}) for node in NODES}
    topology = {node: tuple(other for other in NODES if other != node) for node in NODES}
    policy = CATransitionPolicy()
    initial = CATransitionEngine.initialize("fixed-roles", NODES, policy, topology)
    scores = dict.fromkeys(NODES, 0.5)
    scores["a"] = 0.99
    snapshot = CATransitionEngine.transition(
        initial,
        CATransitionInput(
            0,
            scores,
            dict.fromkeys(NODES, EvidenceCategory.NEUTRAL),
            topology,
        ),
        policy,
    )
    selector = CAStateRoleSelector(
        StaticRoundRoleSelector(TRAINERS, VALIDATORS, TRAINERS, {CAPABILITY: VALIDATORS}),
        CARoleSelectionPolicy(4, 2, minimum_contributors=3, detector_target_counts={CAPABILITY: 2}, fixed_sensitive_roles=True),
    )
    selected = selector.select_roles(
        SelectionContext(
            "fixed-roles",
            1,
            caps,
            snapshot.snapshot_hash,
            scores,
            snapshot,
            0,
            "trust-0",
        )
    )
    assert selected.selected_validators == VALIDATORS
    assert selected.detector_subgroups[CAPABILITY] == VALIDATORS
    assert selected.selection_source == "ca_contributors"


def test_prepared_matrix_controls_inputs_and_uses_unique_output_paths(tmp_path):
    """All 18 runnable configurations differ only in the intended factors."""
    from copy import deepcopy

    import yaml

    from brbfl.experiments.config import load_experiment_config
    from brbfl.experiments.prepare_selection_ablation import prepare

    paths = prepare(tmp_path / "configs")
    assert len(paths) == 18
    configurations = [yaml.safe_load(path.read_text()) for path in paths]
    assert len({config["output_dir"] for config in configurations}) == 18
    for path in paths:
        assert load_experiment_config(path).model_detector.enabled
    controlled = []
    for config in configurations:
        row = deepcopy(config)
        for key in ("selection", "ca", "seed", "attack", "output_dir"):
            row.pop(key)
        controlled.append(row)
        assert config["model_detector"]["eligible_participants"] == config["validation"]["validators"]
        assert config["selection"]["minimum_trust"] == 0.5
    assert all(row == controlled[0] for row in controlled)
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "configs")


def test_bootstrap_configs_bind_policy_and_control_all_other_inputs():
    """The executable pair changes only its policy window and output path."""
    from pathlib import Path

    import yaml

    from brbfl.experiments.config import load_experiment_config

    root = Path(__file__).resolve().parents[2] / 'configs/ablation/ca-bootstrap'
    for scenario in ('clean', 'attacked'):
        baseline = root / f'baseline_{scenario}_666.yaml'
        evidence = root / f'evidence_{scenario}_666.yaml'
        assert load_experiment_config(baseline).ca.policy.evidence_bootstrap_rounds == 0
        assert load_experiment_config(evidence).ca.policy.evidence_bootstrap_rounds == 3
        left, right = (yaml.safe_load(p.read_text()) for p in (baseline, evidence))
        assert left.pop('output_dir') != right.pop('output_dir')
        left['ca']['policy'].pop('evidence_bootstrap_rounds')
        right['ca']['policy'].pop('evidence_bootstrap_rounds')
        assert left == right
        assert left['rounds'] == 4
