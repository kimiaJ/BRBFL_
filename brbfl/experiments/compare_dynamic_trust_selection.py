"""Verify the causal chain in dynamic trust-ranked selection artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brbfl.experiments.compare_byzantine_validator_trust import _digest, compare_evidence


def compare_dynamic(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Validate trust divergence and the attacked next-round assignment."""
    base = compare_evidence(clean, attacked)
    selection = attacked.get("selection") or attacked.get("ledger", {}).get("selection")
    if not selection or not selection.get("verification_result"):
        raise AssertionError("attacked selection evidence is missing or unverified")
    rounds = selection.get("rounds", {})
    if rounds.get("0", {}).get("selected_validators") != ["node-0", "node-3", "node-4"]:
        raise AssertionError("invalid bootstrap assignment")
    if rounds.get("1", {}).get("selected_validators") != ["node-0", "node-1", "node-2"]:
        raise AssertionError("low-trust Byzantine validators were not removed")
    result = {
        "schema_version": "brbfl.dynamic-trust-comparison/v1",
        "comparison_type": "dynamic_trust_selection",
        "trust_comparison_sha256": base["comparison_sha256"],
        "selection": selection,
        "causal_status": "proven_trust_based_selection_removed_low_trust_byzantine_validators",
        "verification_result": True,
        "verification_reason": "verified",
    }
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
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
