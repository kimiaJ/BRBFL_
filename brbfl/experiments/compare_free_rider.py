"""Compare controlled clean and genuine free-rider evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Validate lifecycle invariants and construct the comparison payload."""
    rows = []
    participant_equality = []
    for clean_round, attacked_round in zip(clean["rounds"], attacked["rounds"], strict=True):
        number = attacked_round["round"]
        equal = clean_round["participating_node_ids"] == attacked_round["participating_node_ids"]
        participant_equality.append(equal)
        if not equal:
            raise AssertionError(f"participants differ in round {number}")
        if clean_round["malicious_participant_ids"] != [] or attacked_round["malicious_participant_ids"] != ["node-1"]:
            raise AssertionError("malicious participant intersection is incorrect")
        clean_training = clean_round["per_node_training_evidence"]
        attacked_training = attacked_round["per_node_training_evidence"]
        free_rider = attacked_training["node-1"]
        unchanged = (
            free_rider["optimizer_step_count"] == 0
            and free_rider["local_epochs_actually_executed"] == 0
            and free_rider["pre_training_model_sha256"] == free_rider["submitted_model_sha256"]
            and free_rider["zero_delta_within_tolerance"]
            and free_rider["aggregation_matches_submitted_snapshot"]
            and free_rider["free_rider_attack_application_count"] == 1
        )
        benign = all(
            attacked_training[node]["optimizer_step_count"] > 0 and attacked_training[node]["free_rider_attack_application_count"] == 0
            for node in ("node-0", "node-2")
        )
        if not unchanged or not benign:
            raise AssertionError(f"free-rider lifecycle validation failed in round {number}")
        rows.append(
            {
                "round": number,
                "participants_equal": equal,
                "malicious_participant_ids": attacked_round["malicious_participant_ids"],
                "clean_loss_difference": attacked_round["clean_test_loss"] - clean_round["clean_test_loss"],
                "clean_accuracy_difference": attacked_round["clean_test_accuracy"] - clean_round["clean_test_accuracy"],
                "free_rider_application_counts": {
                    node: evidence["free_rider_attack_application_count"] for node, evidence in attacked_training.items()
                },
                "optimizer_step_counts": {node: evidence["optimizer_step_count"] for node, evidence in attacked_training.items()},
                "node_1_clean_update_l2_norm": clean_training["node-1"]["pre_to_submission_delta_l2_norm"],
                "node_1_attacked_update_l2_norm": free_rider["pre_to_submission_delta_l2_norm"],
                "unchanged_submission_assertion": unchanged,
                "benign_node_training_assertion": benign,
            }
        )
    return {
        "clean_configuration_path": clean["configuration_path"],
        "attacked_configuration_path": attacked["configuration_path"],
        "clean_final_model_sha256": clean["final_model_sha256"],
        "attacked_final_model_sha256": attacked["final_model_sha256"],
        "final_models_differ": clean["final_model_sha256"] != attacked["final_model_sha256"],
        "participant_equality_by_round": participant_equality,
        "participants_equal_overall": all(participant_equality),
        "dataset_preservation_result": clean["dataset_preservation"]["all_partitions_unchanged"]
        and attacked["dataset_preservation"]["all_partitions_unchanged"],
        "rounds": rows,
    }


def compare(clean_path: Path, attacked_path: Path, output_path: Path) -> dict[str, Any]:
    """Load, compare, and persist evidence."""
    result = compare_evidence(_load(clean_path), _load(attacked_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the default comparison or paths supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=Path("results/free-rider-validation/clean/validation.json"))
    parser.add_argument("--attacked", type=Path, default=Path("results/free-rider-validation/attacked/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/free-rider-validation/comparison.json"))
    args = parser.parse_args()
    compare(args.clean, args.attacked, args.output)


if __name__ == "__main__":
    main()
