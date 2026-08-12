"""Verify the causal chain in dynamic trust-ranked selection artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brbfl.experiments.compare_byzantine_validator_trust import (
    _digest,
    canonicalize_json,
    controlled_fields,
    validate_trust_integrity,
)

_BOOTSTRAP_VALIDATORS = ["node-0", "node-3", "node-4"]
_ROTATED_VALIDATORS = ["node-0", "node-1", "node-2"]
_REMOVED_VALIDATORS = ["node-3", "node-4"]


def _selection(run: dict[str, Any], name: str) -> dict[str, Any]:
    selection = run.get("selection") or run.get("ledger", {}).get("selection")
    if not isinstance(selection, dict) or not selection.get("verification_result"):
        raise AssertionError(f"{name} selection evidence is missing or unverified")
    return selection


def _state_matches(state: dict[str, Any], *, score: float, votes: int | None = None) -> bool:
    return abs(state.get("score", -1) - score) <= 1e-12 and (votes is None or state.get("processed_vote_count") == votes)


def compare_dynamic(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Validate integrity and prove the dynamic trust-selection causal chain."""
    clean_trust = validate_trust_integrity(clean, "clean")
    attacked_trust = validate_trust_integrity(attacked, "attacked")
    controlled = controlled_fields(clean)
    if controlled != controlled_fields(attacked):
        raise AssertionError("controlled experiment fields are incompatible")
    for field in ("method", "prior"):
        if clean_trust.get(field) != attacked_trust.get(field):
            raise AssertionError(f"controlled trust field differs: {field}")
    if clean.get("final_model_consensus") is not True or attacked.get("final_model_consensus") is not True:
        raise AssertionError("final-model consensus failed")

    clean_selection, attacked_selection = _selection(clean, "clean"), _selection(attacked, "attacked")
    clean_rounds, attacked_rounds = clean_selection.get("rounds", {}), attacked_selection.get("rounds", {})
    if clean_rounds.get("0", {}).get("selected_validators") != _BOOTSTRAP_VALIDATORS:
        raise AssertionError("invalid clean bootstrap assignment")
    if attacked_rounds.get("0", {}).get("selected_validators") != _BOOTSTRAP_VALIDATORS:
        raise AssertionError("invalid attacked bootstrap assignment")
    if clean_rounds.get("1", {}).get("selected_validators") != _BOOTSTRAP_VALIDATORS:
        raise AssertionError("clean validator assignment did not remain stable")
    attacked_round_one = attacked_rounds.get("1", {})
    if attacked_round_one.get("selected_validators") != _ROTATED_VALIDATORS:
        raise AssertionError("low-trust Byzantine validators were not removed")
    excluded = attacked_round_one.get("excluded_validators", attacked_round_one.get("excluded_participants"))
    if excluded != _REMOVED_VALIDATORS or attacked_round_one.get("trust_source_round") != 0:
        raise AssertionError("attacked rotation lacks canonical exclusion or trust source evidence")

    clean_final, attacked_final = clean_trust["final_states"], attacked_trust["final_states"]
    expected_clean = {"node-0": 0.875, "node-1": 0.5, "node-2": 0.5, "node-3": 0.875, "node-4": 0.875}
    expected_attacked = {"node-0": (0.875, 6), "node-1": (0.8, 3), "node-2": (0.8, 3), "node-3": (0.2, 3), "node-4": (0.2, 3)}
    assertions = {
        "clean_selected_validators_gain_trust": all(
            _state_matches(clean_final[node], score=score) for node, score in expected_clean.items()
        ),
        "round_0_byzantine_validators_lose_trust": all(
            _state_matches(attacked_final[node], score=0.2, votes=3) for node in _REMOVED_VALIDATORS
        ),
        "excluded_validators_receive_no_round_1_updates": all(
            attacked_final[node].get("last_finalized_round") == 0 for node in _REMOVED_VALIDATORS
        ),
        "replacement_validators_receive_honest_updates": all(
            _state_matches(attacked_final[node], score=expected_attacked[node][0], votes=expected_attacked[node][1])
            for node in ("node-1", "node-2")
        ),
        "attacked_final_trust_matches_causal_path": all(
            _state_matches(attacked_final[node], score=score, votes=votes) for node, (score, votes) in expected_attacked.items()
        ),
        "candidate_assignments_unchanged": controlled["candidate_assignments"] == controlled_fields(attacked)["candidate_assignments"],
    }
    if not all(assertions.values()):
        raise AssertionError("dynamic trust-selection causal assertions failed")
    result = {
        "schema_version": "brbfl.dynamic-trust-comparison/v1",
        "comparison_type": "dynamic_trust_selection",
        "controlled_fields": controlled,
        "clean_final_states": clean_final,
        "attacked_final_states": attacked_final,
        "selection": attacked_selection,
        "selection_comparison": {"clean": clean_selection, "attacked": attacked_selection},
        "role_assignment_comparison": "attacked_round_1_rotated_by_finalized_round_0_trust",
        "assertions": assertions,
        "causal_status": "proven_trust_based_selection_removed_low_trust_byzantine_validators",
        "verification_result": True,
        "verification_reason": "verified",
    }
    result = canonicalize_json(result)
    result["comparison_sha256"] = _digest("DynamicTrustComparison/v1", result)
    return result


def main() -> None:
    """Run the explicit-path dynamic comparison CLI."""
    p = argparse.ArgumentParser()
    p.add_argument("--clean", type=Path, required=True)
    p.add_argument("--attacked", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare_dynamic(json.loads(a.clean.read_text()), json.loads(a.attacked.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(canonicalize_json(result), indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
