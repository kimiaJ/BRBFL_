"""Compare completed clean and label-flipping P2PFL validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric(evidence: dict, round_number: int, names: tuple[str, ...]) -> float | None:
    """Average an available P2PFL metric across reporting nodes."""
    values = [row["value"] for row in evidence["rounds"][round_number]["per_node_metrics"] if row["metric"] in names]
    return sum(values) / len(values) if values else None


def compare(clean_path: Path, attacked_path: Path, output_path: Path) -> dict:
    """Write a round-wise machine-readable clean/attacked comparison."""
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    attacked = json.loads(attacked_path.read_text(encoding="utf-8"))
    rounds = []
    for number in range(min(len(clean["rounds"]), len(attacked["rounds"]))):
        clean_loss = metric(clean, number, ("test_loss", "loss"))
        attacked_loss = metric(attacked, number, ("test_loss", "loss"))
        clean_accuracy = metric(clean, number, ("test_metric", "accuracy"))
        attacked_accuracy = metric(attacked, number, ("test_metric", "accuracy"))
        rounds.append(
            {
                "round": number,
                "clean_global_loss": clean_loss,
                "attacked_global_loss": attacked_loss,
                "loss_difference": None if None in (clean_loss, attacked_loss) else attacked_loss - clean_loss,
                "clean_global_accuracy": clean_accuracy,
                "attacked_global_accuracy": attacked_accuracy,
                "accuracy_difference": None if None in (clean_accuracy, attacked_accuracy) else attacked_accuracy - clean_accuracy,
                "clean_participating_node_ids": clean["rounds"][number]["participating_node_ids"],
                "attacked_participating_node_ids": attacked["rounds"][number]["participating_node_ids"],
                "malicious_participant_ids": attacked["rounds"][number]["malicious_participant_ids"],
                "attack_application_counts": attacked["rounds"][number]["attack_application_counts"],
            }
        )
    result = {
        "rounds": rounds,
        "clean_final_model_sha256": clean["final_model_sha256"],
        "attacked_final_model_sha256": attacked["final_model_sha256"],
        "final_models_differ": clean["final_model_sha256"] != attacked["final_model_sha256"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the comparison CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=Path("results/label-flipping-validation/clean/validation.json"))
    parser.add_argument("--attacked", type=Path, default=Path("results/label-flipping-validation/attacked/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/label-flipping-validation/comparison.json"))
    arguments = parser.parse_args()
    compare(arguments.clean, arguments.attacked, arguments.output)


if __name__ == "__main__":
    main()
