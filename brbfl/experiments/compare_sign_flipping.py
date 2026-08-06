"""Compare controlled clean and sign-flipping P2PFL evidence."""

import json
from pathlib import Path

from brbfl.experiments.compare_label_flipping import compare as compare_runs


def compare(clean_path: Path, attacked_path: Path, output_path: Path) -> dict:
    """Assert attack/control evidence, then write the metric comparison."""
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    attacked = json.loads(attacked_path.read_text(encoding="utf-8"))
    if clean["malicious_node_ids"] or any(clean_round["model_update_transformations"] for clean_round in clean["rounds"]):
        raise AssertionError("clean run contains model-update attack evidence")
    for clean_round, attacked_round in zip(clean["rounds"], attacked["rounds"], strict=True):
        if clean_round["participating_node_ids"] != attacked_round["participating_node_ids"]:
            raise AssertionError("clean and attacked participants differ")
        if attacked_round["malicious_participant_ids"] != ["node-1"]:
            raise AssertionError("node-1 must be the sole malicious participant")
        if attacked_round["attack_application_counts"] != {"node-0": 0, "node-1": 1, "node-2": 0}:
            raise AssertionError("sign flipping must be applied exactly once to node-1")
        if set(attacked_round["model_update_transformations"]) != {"node-1"}:
            raise AssertionError("only node-1 may have a transformed update")
    result = compare_runs(clean_path, attacked_path, output_path)
    if not result["final_models_differ"]:
        raise AssertionError("final models unexpectedly match")
    return result


def main() -> None:
    """Write the default sign-flipping comparison JSON."""
    compare(
        Path("results/sign-flipping-validation/clean/validation.json"),
        Path("results/sign-flipping-validation/attacked/validation.json"),
        Path("results/sign-flipping-validation/comparison.json"),
    )


if __name__ == "__main__":
    main()
