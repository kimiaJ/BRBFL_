"""Verify a paired three-round detector exclusion experiment from runtime receipts."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from brbfl.canonical import canonical_hash


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _verify_artifact(value: dict[str, Any]) -> None:
    _require(value.get("execution_mode") == "real_p2pfl_runtime", "real runtime evidence required")
    _require(value.get("configured_rounds") == 3 and len(value.get("rounds", [])) == 3, "three runtime rounds required")
    ledger, trust, ca = value["ledger"], value["trust"], value["ca"]
    _require(ledger.get("enabled") is True and ledger.get("ledger_round_consensus") is True, "ledger consensus missing")
    _require(trust.get("verification_result") is True and not trust.get("observation_only"), "enforced trust required")
    _require(ca.get("enabled") is True, "CA required")
    for number in range(3):
        key = str(number)
        record, assignment = ledger["rounds"][key], ledger["per_round_role_assignment"][key]
        assignment_hash = canonical_hash("RoundRoleAssignment/v1", assignment)
        _require(assignment_hash == ledger["per_round_role_assignment_hash"][key], "assignment hash mismatch")
        selected = set(assignment["selected_contributors"])
        _require(len(selected) >= 3, "median reference requires three contributors")
        _require(record.get("finalized") is True and ledger["round_verification"].get(key) is True, "round is not verified")
        _require(set(record["candidate_commitments"]) == selected == set(record["admission"]), "candidate membership mismatch")
        aggregate = record["aggregate"]
        admitted = {node for node, accepted in record["admission"].items() if accepted}
        _require(set(aggregate["contributor_hashes"]) == admitted, "aggregate includes a rejected or missing input")
        for node in admitted:
            _require(
                aggregate["contributor_hashes"][node] == record["candidate_commitments"][node]["candidate_hash"], "input hash mismatch"
            )
        installed = record["installation_confirmations"]
        _require(set(installed) == set(assignment["network_participants"]), "missing installation confirmation")
        _require(set(installed.values()) == {aggregate["aggregate_hash"]}, "conflicting aggregate installation")
        roles = value["rounds"][number]["trainer_roles"]
        _require(
            set(roles["actual_training_nodes"]) == selected == set(roles["submitted_candidates"]), "actual training differs from assignment"
        )
        for node, row in roles["per_node_selection"].items():
            _require(set(row["selected_trainers"]) == selected, "nodes disagree on trainer selection")
            _require(row["actually_trained"] == (node in selected) == row["submitted_candidate"], "per-node training contradicts selection")
        committees = record["detector_committees"]
        _require(
            Counter(row["decision"]["participant_id"] for row in committees) == Counter(selected),
            "missing or duplicate committee decisions",
        )
        for row in committees:
            decision, evidence = row["decision"], row["evidence"]
            payload = {k: v for k, v in decision.items() if k != "decision_hash"}
            _require(canonical_hash("DetectorCommitteeDecision/v1", payload) == decision["decision_hash"], "decision hash mismatch")
            candidate = decision["participant_id"]
            source = record["candidate_commitments"][candidate]["candidate_hash"]
            members = set(assignment["detector_subgroups"]["model_update_detector"])
            _require(
                decision["assignment_hash"] == assignment_hash and decision["source_artifact_hash"] == source, "stale committee decision"
            )
            _require(decision["capability"] == "model_update_detector", "wrong detector capability")
            seen, votes = set(), Counter()
            for item in evidence:
                payload = {k: v for k, v in item.items() if k != "evidence_hash"}
                _require(canonical_hash("DetectorEvidence/v1", payload) == item["evidence_hash"], "evidence hash mismatch")
                _require(item["detector_id"] in members and item["detector_id"] not in seen, "invalid detector membership")
                seen.add(item["detector_id"])
                _require(item["round"] == number and item["evaluated_participant_id"] == candidate, "wrong evidence identity")
                _require(
                    item["detector_assignment_hash"] == assignment_hash and item["source_artifact_hash"] == source, "wrong evidence context"
                )
                if item["verification_result"]:
                    votes[item["evidence_category"]] += 1
            _require(sorted(item["evidence_hash"] for item in evidence) == decision["evidence_hashes"], "evidence set mismatch")
            _require(
                decision["quorum"] == len(members) // 2 + 1 and votes[decision["category"]] >= decision["quorum"],
                "committee quorum missing",
            )
            updates = trust["rounds"][key]["detector_updates"]
            _require(decision in updates, "committee decision did not reach trust")
        transition = ca["transitions"][key]
        _require(transition["source_round"] == number, "wrong CA transition round")
        _require(transition["source_trust_snapshot_hash"] == trust["rounds"][key]["snapshot_sha256"], "CA did not consume finalized trust")
        _require(transition["previous_ca_snapshot_hash"] == ca["generations"][key]["snapshot_hash"], "broken CA source chain")
        _require(transition["resulting_ca_snapshot_hash"] == ca["generations"][str(number + 1)]["snapshot_hash"], "broken CA result chain")
        if number:
            _require(assignment["selection_source"] == "ca_state", "selection did not use CA")
            _require(assignment["source_ca_generation"] == number, "wrong source generation")
            _require(assignment["source_ca_snapshot_hash"] == ca["generations"][key]["snapshot_hash"], "stale CA snapshot")
            _require(assignment["source_trust_hash"] == trust["rounds"][str(number - 1)]["snapshot_sha256"], "stale trust snapshot")
    hashes = value.get("per_node_final_installed_model_hashes", {})
    _require(
        value.get("final_model_consensus") is True and set(hashes) == set(installed) and len(set(hashes.values())) == 1,
        "final consensus missing",
    )


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Check the bounded three-round intervention, without a general robustness claim."""
    for value in (clean, attacked):
        _verify_artifact(value)
    configs = []
    for value in (clean, attacked):
        config = copy.deepcopy(value["configuration"])
        config.pop("attack", None)
        config.pop("output_dir", None)
        configs.append(config)
    _require(configs[0] == configs[1], "controlled configurations differ")
    for field in ("producing_commit", "partition_manifest_sha256"):
        _require(
            clean["provenance"].get(field) and clean["provenance"][field] == attacked["provenance"].get(field), "run provenance differs"
        )
    _require(clean["attack_type"] == "none" and not clean["malicious_node_ids"], "clean control has an attack")
    _require(attacked["attack_type"] == "sign_flipping" and attacked["malicious_node_ids"], "sign-flipping intervention required")
    _require(
        all(
            all(record["admission"].values()) and all(row["decision"]["category"] == "positive" for row in record["detector_committees"])
            for record in clean["ledger"]["rounds"].values()
        ),
        "clean input was rejected or flagged",
    )
    for node in attacked["malicious_node_ids"]:
        for number in (0, 1):
            key = str(number)
            decision = next(
                row["decision"]
                for row in attacked["ledger"]["rounds"][key]["detector_committees"]
                if row["decision"]["participant_id"] == node
            )
            _require(
                decision["category"] == "severe" and attacked["ledger"]["rounds"][key]["admission"][node] is False,
                "attack was not detected and blocked",
            )
        scores = [attacked["trust"]["rounds"][str(n)]["post_round"][node]["score"] for n in (0, 1)]
        prior = attacked["trust"]["rounds"]["0"]["pre_round"][node]["score"]
        _require(prior > scores[0] > scores[1], "severe evidence did not reduce contributor trust")
        path = [attacked["ca"]["generations"][str(g)]["participant_states"][node]["state"] for g in (1, 2)]
        _require(path == ["suspicious", "excluded"], "CA escalation missing")
        third = attacked["ledger"]["per_round_role_assignment"]["2"]
        _require(node not in third["selected_contributors"] + third["selected_validators"], "excluded node still selected")
        _require(all(node not in members for members in third["detector_subgroups"].values()), "excluded node still detects")
        for number in range(3):
            key = str(number)
            _require(clean["ledger"]["rounds"][key]["admission"].get(node) is True, "clean counterpart was rejected")
    _require(
        all(
            row["state"] not in {"suspicious", "excluded"}
            for gen in clean["ca"]["generations"].values()
            for row in gen["participant_states"].values()
        ),
        "false clean escalation",
    )
    return {
        "verification_result": True,
        "execution_mode": "real_p2pfl_runtime",
        "status": "verified_three_round_detector_exclusion",
        "rounds_verified": 3,
        "excluded_participants": attacked["malicious_node_ids"],
        "scope": "paired smoke test; no general robustness or accuracy claim",
    }


def main() -> None:
    """Compare user-produced clean and attacked validation artifacts."""
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
