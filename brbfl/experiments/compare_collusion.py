"""Validate the controlled clean-versus-coordinated-collusion experiment."""
# ruff: noqa: D103

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOLERANCE = 1e-6
CONTROLLED_FIELDS = ("nodes", "rounds", "epochs", "seed", "protocol", "framework", "aggregator", "topology", "batch_size", "dataset")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless participation, training, coordination, and receipt are proven."""
    for field in CONTROLLED_FIELDS:
        if clean["configuration"][field] != attacked["configuration"][field]:
            raise AssertionError(f"controlled configuration differs: {field}")
    if attacked["malicious_node_ids"] != ["node-1", "node-2"]:
        raise AssertionError("configured colluders are incorrect")
    rounds = []
    for clean_round, attack_round in zip(clean["rounds"], attacked["rounds"], strict=True):
        number = attack_round["round"]
        if clean_round["participating_node_ids"] != attack_round["participating_node_ids"]:
            raise AssertionError(f"participants differ in round {number}")
        group = attack_round["collusion_group_evidence"]
        if group["participating_colluders"] != ["node-1", "node-2"] or group["missing_configured_colluders"]:
            raise AssertionError(f"colluder participation invalid in round {number}")
        if not group["identical_shared_direction"]:
            raise AssertionError(f"shared directions differ in round {number}")
        evidence = attack_round["per_node_training_evidence"]
        for node in ("node-1", "node-2"):
            row = evidence[node]
            expected = row["configured_alpha"] * row["genuine_update_l2_norm"]
            if row["optimizer_step_count"] <= 0 or row["effective_local_epochs"] <= 0 or row["attack_application_count"] != 1:
                raise AssertionError(f"genuine colluder training/application invalid: {node}, round {number}")
            if abs(row["submitted_update_l2_norm"] - expected) > TOLERANCE * max(1.0, expected):
                raise AssertionError(f"collusive norm formula failed: {node}, round {number}")
            if not row["aggregation_receipt"] or row["aggregation_input_sha256"] != row["submitted_model_sha256"]:
                raise AssertionError(f"submission did not reach aggregation: {node}, round {number}")
        if evidence["node-0"]["attack_application_count"] != 0:
            raise AssertionError("benign node received collusion attack")
        pair = group["pairwise_updates"][0]
        if abs(pair["submitted_cosine_similarity"] - 1.0) > TOLERANCE:
            raise AssertionError(f"collusive submissions are not aligned in round {number}")
        rounds.append(
            {
                "round": number,
                "participants_equal": True,
                "optimizer_step_counts": {n: evidence[n]["optimizer_step_count"] for n in ("node-1", "node-2")},
                "genuine_update_norms": {n: evidence[n]["genuine_update_l2_norm"] for n in ("node-1", "node-2")},
                "submitted_update_norms": {n: evidence[n]["submitted_update_l2_norm"] for n in ("node-1", "node-2")},
                "shared_direction_hashes": group["shared_direction_hashes_by_colluder"],
                **pair,
                "clean_loss": clean_round["clean_test_loss"],
                "attacked_loss": attack_round["clean_test_loss"],
                "clean_accuracy": clean_round["clean_test_accuracy"],
                "attacked_accuracy": attack_round["clean_test_accuracy"],
            }
        )
    clean_last, attacked_last = clean["rounds"][-1], attacked["rounds"][-1]
    return {
        "tolerance": TOLERANCE,
        "participants_equal_overall": True,
        "dataset_preservation_result": clean["dataset_preservation"]["all_partitions_unchanged"]
        and attacked["dataset_preservation"]["all_partitions_unchanged"],
        "rounds": rounds,
        "final_loss_difference": attacked_last["clean_test_loss"] - clean_last["clean_test_loss"],
        "final_accuracy_difference": attacked_last["clean_test_accuracy"] - clean_last["clean_test_accuracy"],
        "clean_final_model_sha256": clean["final_model_sha256"],
        "attacked_final_model_sha256": attacked["final_model_sha256"],
        "clean_final_hash_source": clean["canonical_final_hash_source"],
        "attacked_final_hash_source": attacked["canonical_final_hash_source"],
    }


def compare(clean_path: Path, attacked_path: Path, output_path: Path) -> dict[str, Any]:
    result = compare_evidence(_load(clean_path), _load(attacked_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=Path("results/collusion-validation/clean/validation.json"))
    parser.add_argument("--attacked", type=Path, default=Path("results/collusion-validation/attacked/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/collusion-validation/comparison.json"))
    args = parser.parse_args()
    compare(args.clean, args.attacked, args.output)


if __name__ == "__main__":
    main()
