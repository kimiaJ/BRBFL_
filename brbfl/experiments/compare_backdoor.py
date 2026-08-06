"""Compare controlled clean and genuine MNIST-backdoor evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Validate paired-run invariants and return a round-wise comparison."""
    clean_rounds = clean["rounds"]
    attacked_rounds = attacked["rounds"]
    if len(clean_rounds) != len(attacked_rounds):
        raise AssertionError("controlled runs have different round counts")
    participant_equality = [
        a["participating_node_ids"] == b["participating_node_ids"] for a, b in zip(clean_rounds, attacked_rounds, strict=True)
    ]
    result = {
        "clean_final_model_sha256": clean["final_model_sha256"],
        "attacked_final_model_sha256": attacked["final_model_sha256"],
        "final_models_differ": clean["final_model_sha256"] != attacked["final_model_sha256"],
        "clean_accuracy_differences_by_round": [
            b.get("clean_test_accuracy", 0.0) - a.get("clean_test_accuracy", 0.0)
            for a, b in zip(clean_rounds, attacked_rounds, strict=True)
        ],
        "genuine_triggered_asr_differences_by_round": [
            b.get("triggered_test_asr", 0.0) - a.get("triggered_test_asr", 0.0) for a, b in zip(clean_rounds, attacked_rounds, strict=True)
        ],
        "participant_equality_by_round": participant_equality,
        "participants_equal": all(participant_equality),
        "poisoning_application_counts": {
            item["node_id"]: item["attack_application_count"] for item in attacked["per_node_poisoning_evidence"]
        },
        "malicious_participant_ids": attacked["malicious_node_ids"],
    }
    if result["poisoning_application_counts"].get("node-1") != 1:
        raise AssertionError("node-1 must be poisoned exactly once")
    if any(count for node, count in result["poisoning_application_counts"].items() if node != "node-1"):
        raise AssertionError("benign nodes must not be poisoned")
    return result


def write_comparison(clean_path: Path, attacked_path: Path, output: Path) -> dict[str, Any]:
    """Write comparison.json for two completed runs."""
    result = compare_evidence(_load(clean_path), _load(attacked_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the comparison command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=Path("results/backdoor-validation/clean/validation.json"))
    parser.add_argument("--attacked", type=Path, default=Path("results/backdoor-validation/attacked/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/backdoor-validation/comparison.json"))
    args = parser.parse_args()
    write_comparison(args.clean, args.attacked, args.output)


if __name__ == "__main__":
    main()
