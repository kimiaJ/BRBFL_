"""Fail-closed verification of paired CA-5 real P2PFL runtime artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from brbfl.canonical import canonical_hash

EXECUTION_MODE = "real_p2pfl_runtime"
STATUS = "proven_runtime_ca_transitions_excluded_byzantine_participants"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_runtime_artifact(artifact: dict[str, Any]) -> None:
    """Validate recorded production objects without regenerating CA decisions."""
    _require(artifact.get("execution_mode") == EXECUTION_MODE, "artifact is not from the real P2PFL runtime")
    ledger, trust, ca = artifact.get("ledger"), artifact.get("trust"), artifact.get("ca")
    _require(isinstance(ledger, dict) and ledger.get("enabled") is True, "missing runtime ledger provenance")
    _require(isinstance(trust, dict) and isinstance(ca, dict) and ca.get("enabled") is True, "missing trust/CA provenance")
    _require(ledger.get("ledger_round_consensus") is True, "ledger rounds did not reach consensus")
    rounds = ledger.get("rounds", {})
    _require(len(rounds) == artifact.get("configured_rounds") == 6, "CA-5 requires six finalized runtime rounds")
    assignments = ledger.get("per_round_role_assignment", {})
    assignment_hashes = ledger.get("per_round_role_assignment_hash", {})
    generations, transitions = ca.get("generations", {}), ca.get("transitions", {})
    _require(len(transitions) == 6 and len(generations) == 7, "exactly one CA finalization is required per round")
    for number in range(6):
        key = str(number)
        row = rounds.get(key, {})
        _require(row.get("finalized") is True and ledger["round_verification"].get(key) is True, "unverified ledger round")
        assignment = assignments.get(key)
        _require(isinstance(assignment, dict), "missing immutable role assignment")
        _require(
            assignment_hashes.get(key) == canonical_hash("RoundRoleAssignment/v1", assignment),
            "forged role assignment hash",
        )
        expected_source = "static" if number == 0 else "ca_state"
        _require(assignment.get("selection_source") == expected_source, "selection did not use the configured lifecycle")
        source = generations.get(key, {})
        if number > 0:
            _require(assignment.get("source_ca_generation") == number, "selection consumed the wrong CA generation")
            _require(assignment.get("source_ca_snapshot_hash") == source.get("snapshot_hash"), "selection consumed a stale CA snapshot")
        transition = transitions.get(key, {})
        result = generations.get(str(number + 1), {})
        _require(transition.get("source_round") == number, "CA transition source round mismatch")
        _require(transition.get("source_ledger_hash") and transition.get("source_trust_snapshot_hash"), "missing runtime provenance")
        _require(transition.get("previous_ca_snapshot_hash") == source.get("snapshot_hash"), "broken CA hash continuity")
        _require(transition.get("resulting_ca_snapshot_hash") == result.get("snapshot_hash"), "forged resulting CA hash")
        _require(result.get("previous_snapshot_hash") == source.get("snapshot_hash"), "broken snapshot chain")
        for record in result.get("transition_records", []):
            expected = {
                ("observation", "suspicious", "severe"): "observation_severe",
                ("suspicious", "excluded", "severe"): "suspicious_repeated_negative",
                ("observation", "trusted", "positive"): "observation_promoted",
            }.get((record["previous_state"], record["next_state"], record["evidence_category"]))
            if record["previous_state"] == record["next_state"]:
                expected = "state_held"
            _require(expected == record.get("reason_code"), "altered or impossible CA transition reason")
    _require(artifact.get("final_model_consensus") is True, "final model lacks cross-node consensus")
    hashes = artifact.get("per_node_final_installed_model_hashes", {})
    _require(len(hashes) >= 2 and len(set(hashes.values())) == 1, "consensus cannot be claimed from one node")


def _controlled_configuration(value: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(value["configuration"])
    config.pop("output_dir", None)
    config.pop("attack", None)
    return config


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Make a causal claim only after independently validating both artifacts."""
    validate_runtime_artifact(clean)
    validate_runtime_artifact(attacked)
    _require(_controlled_configuration(clean) == _controlled_configuration(attacked), "controlled fields differ")
    adversaries = set(attacked.get("malicious_node_ids", ()))
    _require(adversaries, "attacked artifact lacks the validated Byzantine intervention")
    clean_states = clean["ca"]["generations"]
    attacked_states = attacked["ca"]["generations"]
    for node in adversaries:
        path = [attacked_states[str(g)]["participant_states"][node]["state"] for g in range(1, 7)]
        _require(path[0] == "suspicious" and path[1:] == ["excluded"] * 5, "Byzantine CA path was not proven")
        for round_number in range(1, 6):
            assignment = attacked["ledger"]["per_round_role_assignment"][str(round_number)]
            _require(node not in assignment["selected_validators"], "suspicious participant actually validated")
            if round_number >= 2:
                _require(node not in assignment["selected_contributors"], "excluded participant actually trained")
                aggregate = attacked["ledger"]["rounds"][str(round_number)]["aggregate"] or {}
                _require(node not in aggregate.get("contributor_hashes", {}), "excluded update was accepted for aggregation")
    _require(
        all(
            row["state"] not in {"suspicious", "excluded"}
            for generation in clean_states.values()
            for row in generation["participant_states"].values()
        ),
        "clean participant entered Byzantine escalation",
    )
    return {"verification_result": True, "execution_mode": EXECUTION_MODE, "causal_status": STATUS}


def main() -> None:
    """Run the paired artifact comparison command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--attacked", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_evidence(json.loads(args.clean.read_text()), json.loads(args.attacked.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
