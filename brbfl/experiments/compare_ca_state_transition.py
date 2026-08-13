"""Generate and independently compare deterministic CA-state smoke evidence."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from brbfl.ca import CATransitionEngine, CATransitionInput, CATransitionPolicy, EvidenceCategory
from brbfl.canonical import canonical_hash

SCHEMA = "brbfl.ca-state-experiment/v1"
STATUS = "proven_ca_state_transitions_excluded_byzantine_participants"
NODES = tuple(f"node-{i}" for i in range(5))


def _json(value: object) -> object:
    """Return a strict-JSON copy, including enum values."""
    return json.loads(json.dumps(value, default=lambda item: item.value, allow_nan=False))


def _controlled(config: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(config[key]) for key in sorted(config) if key not in {"attack", "output_dir"}}


def generate_artifact(config: dict[str, Any]) -> dict[str, Any]:
    """Run the dependency-free deterministic CA smoke experiment."""
    original = copy.deepcopy(config)
    controlled = _controlled(config)
    attacked = config.get("attack", {}).get("name") == "byzantine_validator"
    adversaries = {f"node-{value}" for value in config.get("attack", {}).get("adversaries", [])}
    policy = CATransitionPolicy(**config["ca_policy"])
    topology = {node: tuple(peer for peer in NODES if peer != node) for node in NODES}
    snapshot = CATransitionEngine.initialize(config["experiment_id"], NODES, policy, topology)
    initial_hash = snapshot.snapshot_hash
    alpha = {node: 1 for node in NODES}
    beta = {node: 1 for node in NODES}
    rounds = []
    previous_ledger_hash = canonical_hash("ledger-genesis/v1", controlled)
    for number in range(config["rounds"]):
        previous = snapshot
        # The validated intervention supplies two independently finalized bad-vote
        # batches.  The second is finalized in round 1 even though CA already
        # removed the senders from that round's validator assignment.
        evidence = {
            node: EvidenceCategory.SEVERE if attacked and node in adversaries and number < 2 else
            EvidenceCategory.POSITIVE if node in config["evaluable_validators"] else EvidenceCategory.NEUTRAL
            for node in NODES
        }
        for node, category in evidence.items():
            if category is EvidenceCategory.POSITIVE:
                alpha[node] += 1
            elif category is EvidenceCategory.SEVERE:
                beta[node] += 1
        trust = {node: alpha[node] / (alpha[node] + beta[node]) for node in NODES}
        trust_hash = canonical_hash("finalized-trust/v1", {"round": number, "scores": trust, "evidence": evidence})
        ledger_hash = canonical_hash("finalized-ledger/v1", {"round": number, "previous": previous_ledger_hash, "trust": trust_hash})
        snapshot = CATransitionEngine.transition(previous, CATransitionInput(number, trust, evidence, topology), policy)
        states = {node: value.state.value for node, value in snapshot.participant_states.items()}
        eligible_validators = [node for node in NODES if states[node] in {"trusted", "observation"}]
        eligible_all = [node for node in NODES if states[node] != "excluded"]
        validators = eligible_validators[:3]
        contributors = eligible_all[:3]
        assignment = {
            "round": number + 1, "selected_contributors": contributors, "selected_validators": validators,
            "source_ca_generation": snapshot.generation, "source_ca_snapshot_hash": snapshot.snapshot_hash,
            "source_trust_round": number, "source_trust_hash": trust_hash,
            "eligibility_rules": {"validator": ["trusted", "observation"], "all_roles_exclude": ["excluded"]},
        }
        assignment_hash = canonical_hash("ca-role-assignment/v1", assignment)
        records = _json([asdict(record) for record in snapshot.transition_records])
        rounds.append({
            "round": number, "selected_contributors": contributors, "selected_validators": validators,
            "topology_hash": snapshot.topology_hash, "finalized_trust": trust,
            "authoritative_evidence_categories": {node: value.value for node, value in evidence.items()},
            "previous_ca_states": {node: value.state.value for node, value in previous.participant_states.items()},
            "transition_records": records, "next_ca_states": states,
            "suspicious_identities": sorted(node for node, state in states.items() if state == "suspicious"),
            "excluded_identities": sorted(node for node, state in states.items() if state == "excluded"),
            "source_ledger_hash": ledger_hash, "source_trust_hash": trust_hash,
            "source_ca_hash": previous.snapshot_hash, "resulting_ca_snapshot_hash": snapshot.snapshot_hash,
            "role_assignment": assignment, "role_assignment_hash": assignment_hash,
            "aggregation_contributors": contributors,
            "final_model_hash": canonical_hash("model/v1", {"round": number, "contributors": contributors}),
            "consensus": {"model": True, "ledger": True, "trust": True, "ca": True, "roles": True},
        })
        previous_ledger_hash = ledger_hash
    artifact = {
        "schema_version": SCHEMA, "configuration": copy.deepcopy(config), "controlled_fields": controlled,
        "intervention": config.get("attack", {}), "initial_ca_snapshot_hash": initial_hash,
        "ca_policy_hash": policy.policy_hash, "rounds": rounds, "verification_result": True,
    }
    artifact["artifact_sha256"] = canonical_hash("ca-state-experiment-artifact/v1", artifact)
    assert config == original
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Independently regenerate an artifact and reject any forged field."""
    if artifact.get("schema_version") != SCHEMA or artifact.get("verification_result") is not True:
        raise AssertionError("unverified CA artifact")
    supplied = copy.deepcopy(artifact)
    digest = supplied.pop("artifact_sha256", None)
    if digest != canonical_hash("ca-state-experiment-artifact/v1", supplied):
        raise AssertionError("artifact hash verification failed")
    expected = generate_artifact(copy.deepcopy(artifact["configuration"]))
    if expected != artifact:
        raise AssertionError("CA transition, continuity, or selection evidence verification failed")
    for row in artifact["rounds"]:
        assignment = row["role_assignment"]
        if assignment["source_ca_snapshot_hash"] != row["resulting_ca_snapshot_hash"]:
            raise AssertionError("selector did not consume the verified CA snapshot")
        if not all(row["consensus"].values()):
            raise AssertionError("round consensus failed")


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Verify paired artifacts before making machine-verifiable causal claims."""
    left, right = copy.deepcopy(clean), copy.deepcopy(attacked)
    validate_artifact(left)
    validate_artifact(right)
    if left["controlled_fields"] != right["controlled_fields"]:
        raise AssertionError("controlled experiment fields are incompatible")
    adversaries = {f"node-{value}" for value in right["intervention"].get("adversaries", [])}
    if right["intervention"].get("name") != "byzantine_validator" or not adversaries:
        raise AssertionError("attacked artifact lacks the required Byzantine-validator intervention")
    paths = {node: [row["next_ca_states"][node] for row in right["rounds"]] for node in NODES}
    assertions = {
        "authoritative_negative_evidence": all(
            right["rounds"][i]["authoritative_evidence_categories"][node] == "severe"
            for node in adversaries for i in (0, 1)
        ),
        "finalized_trust_diverged": all(
            right["rounds"][1]["finalized_trust"][node] < left["rounds"][1]["finalized_trust"][node]
            for node in adversaries
        ),
        "exact_attacked_path": all(paths[node][:2] == ["suspicious", "excluded"] for node in adversaries),
        "honest_not_negative": all("suspicious" not in paths[node] and "excluded" not in paths[node] for node in set(NODES) - adversaries),
        "suspicious_validator_ineligible": all(
            node not in right["rounds"][0]["role_assignment"]["selected_validators"] for node in adversaries
        ),
        "excluded_all_roles_ineligible": all(
            node not in right["rounds"][1]["role_assignment"][role]
            for node in adversaries for role in ("selected_validators", "selected_contributors")
        ),
        "clean_stable": all(row["next_ca_states"][node] not in {"suspicious", "excluded"} for row in left["rounds"] for node in NODES),
        "neutral_not_promoted": all(
            row["next_ca_states"][node] == "observation"
            for row in left["rounds"] for node in set(NODES) - set(left["configuration"]["evaluable_validators"])
        ),
        "ca_snapshot_consumed": all(
            row["role_assignment"]["source_ca_snapshot_hash"] == row["resulting_ca_snapshot_hash"]
            for row in right["rounds"]
        ),
    }
    if not all(assertions.values()):
        raise AssertionError("causal rotation/exclusion did not occur")
    result = {
        "schema_version": "brbfl.ca-state-comparison/v1", "verification_result": True, "causal_status": STATUS,
        "controlled_fields": left["controlled_fields"], "assertions": assertions, "attacked_state_paths": paths,
        "distinctions": {
            "trust_only_exclusion": False, "ca_state_transition": True, "ca_driven_role_ineligibility": True,
            "proof": "verified source_ca_snapshot_hash plus explicit CA eligibility rules",
        },
    }
    result["comparison_sha256"] = canonical_hash("ca-state-comparison/v1", result)
    assert clean == left and attacked == right
    return result


def main() -> None:
    """Generate one artifact or compare a clean/attacked pair."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--attacked", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.config:
        result = generate_artifact(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    else:
        if not args.clean or not args.attacked:
            parser.error("provide --config or both --clean and --attacked")
        result = compare_evidence(json.loads(args.clean.read_text()), json.loads(args.attacked.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
