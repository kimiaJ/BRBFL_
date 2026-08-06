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
    trace = attacked.get("model_update_event_trace", {}).get("node-1", [])
    created = {event["update_id"]: event["framework_round"] for event in trace if event["event_type"] == "local_update_created"}
    transmitted = {event["update_id"] for event in trace if event["event_type"] == "update_transmitted"}
    eligible_ids = set(created) & transmitted
    applied_ids = {event["update_id"] for event in trace if event["event_type"] == "sign_flipping_logically_applied"}
    if eligible_ids != applied_ids:
        raise AssertionError(
            f"eligible malicious updates and logical sign-flipping applications differ: "
            f"eligible={sorted(eligible_ids)}, applied={sorted(applied_ids)}"
        )
    eligible_rounds = {created[update_id] for update_id in eligible_ids}
    for clean_round, attacked_round in zip(clean["rounds"], attacked["rounds"], strict=True):
        if clean_round["participating_node_ids"] != attacked_round["participating_node_ids"]:
            raise AssertionError("clean and attacked participants differ")
        if attacked_round["malicious_participant_ids"] != ["node-1"]:
            raise AssertionError("node-1 must be the sole malicious participant")
        expected = 1 if str(attacked_round["round"]) in eligible_rounds else 0
        if attacked_round["attack_application_counts"] != {"node-0": 0, "node-1": expected, "node-2": 0}:
            raise AssertionError("sign flipping must be applied exactly once to node-1")
        if set(attacked_round["model_update_transformations"]) != ({"node-1"} if expected else set()):
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
