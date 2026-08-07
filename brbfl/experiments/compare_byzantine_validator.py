"""Compare controlled clean and Byzantine-validator admission evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONTROLLED_FIELDS = (
    "nodes",
    "rounds",
    "epochs",
    "seed",
    "protocol",
    "framework",
    "aggregator",
    "topology",
    "batch_size",
    "dataset",
    "validation",
)


def _candidate_map(evidence: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(row["round"], row["candidate_node_id"]): row for row in evidence["validator_admission"]}


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless falsified votes alter the real admission path."""
    for field in CONTROLLED_FIELDS:
        if clean["configuration"].get(field) != attacked["configuration"].get(field):
            raise AssertionError(f"controlled configuration differs: {field}")
    clean_rows, attacked_rows = _candidate_map(clean), _candidate_map(attacked)
    if clean_rows.keys() != attacked_rows.keys():
        raise AssertionError("candidate participation differs")
    changed = []
    comparisons = []
    for key in clean_rows:
        baseline, attack = clean_rows[key], attacked_rows[key]
        if baseline["submitted_model_sha256"] != attack["submitted_model_sha256"]:
            raise AssertionError(f"submitted candidate differs: {key}")
        clean_votes = {vote["validator_node_id"]: vote for vote in baseline["votes"]}
        attack_votes = {vote["validator_node_id"]: vote for vote in attack["votes"]}
        if clean_votes.keys() != attack_votes.keys():
            raise AssertionError(f"validator membership differs: {key}")
        for validator, vote in attack_votes.items():
            reference = clean_votes[validator]["reference_decision"]
            if vote["reference_decision"] != reference:
                raise AssertionError(f"reference decision differs: {key}, {validator}")
            if vote["byzantine"]:
                if vote["reported_decision"] == reference or vote["attack_application_count"] != 1:
                    raise AssertionError(f"Byzantine validator did not invert exactly once: {key}, {validator}")
            elif vote["reported_decision"] != reference or vote["attack_application_count"]:
                raise AssertionError(f"honest validator falsified a vote: {key}, {validator}")
        if baseline["admitted"] != attack["admitted"]:
            changed.append(
                {
                    "round": key[0],
                    "candidate_node_id": key[1],
                    "clean_admitted": baseline["admitted"],
                    "attacked_admitted": attack["admitted"],
                }
            )
        for row in (baseline, attack):
            if row["admitted"]:
                if not row["reached_aggregator_add_model"] or not row["aggregation_matches_submitted_snapshot"]:
                    raise AssertionError(f"admitted snapshot did not reach aggregation unchanged: {key}")
            elif row["reached_aggregator_add_model"] or not row["rejection_receipt"]:
                raise AssertionError(f"rejected snapshot reached aggregation: {key}")
        comparisons.append({"round": key[0], "candidate_node_id": key[1], "clean": baseline, "attacked": attack})
    if not changed:
        raise AssertionError("Byzantine votes did not change a real admission result")
    clean_last, attack_last = clean["rounds"][-1], attacked["rounds"][-1]
    return {
        "controlled_configuration_equal": True,
        "dataset_preserved": clean["dataset_preservation"]["all_partitions_unchanged"]
        and attacked["dataset_preservation"]["all_partitions_unchanged"],
        "candidates": comparisons,
        "changed_admission_results": changed,
        "clean_aggregation_contributors": [row["candidate_node_id"] for row in clean_rows.values() if row["admitted"]],
        "attacked_aggregation_contributors": [row["candidate_node_id"] for row in attacked_rows.values() if row["admitted"]],
        "clean_metrics_by_round": [
            {"round": row["round"], "loss": row["clean_test_loss"], "accuracy": row["clean_test_accuracy"]} for row in clean["rounds"]
        ],
        "attacked_metrics_by_round": [
            {"round": row["round"], "loss": row["clean_test_loss"], "accuracy": row["clean_test_accuracy"]} for row in attacked["rounds"]
        ],
        "final_loss_difference": attack_last["clean_test_loss"] - clean_last["clean_test_loss"],
        "final_accuracy_difference": attack_last["clean_test_accuracy"] - clean_last["clean_test_accuracy"],
        "clean_final_model_sha256": clean["final_model_sha256"],
        "attacked_final_model_sha256": attacked["final_model_sha256"],
        "clean_per_node_final_hashes": clean["per_node_final_installed_model_hashes"],
        "attacked_per_node_final_hashes": attacked["per_node_final_installed_model_hashes"],
        "canonical_final_hash_source": "node-0 final installed global model",
    }


def compare(clean_path: Path, attacked_path: Path, output_path: Path) -> dict[str, Any]:
    """Load, validate, and persist the comparison."""
    result = compare_evidence(json.loads(clean_path.read_text()), json.loads(attacked_path.read_text()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    """Run the command-line comparator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=Path("results/byzantine-validator-validation/clean/validation.json"))
    parser.add_argument("--attacked", type=Path, default=Path("results/byzantine-validator-validation/attacked/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/byzantine-validator-validation/comparison.json"))
    args = parser.parse_args()
    compare(args.clean, args.attacked, args.output)


if __name__ == "__main__":
    main()
