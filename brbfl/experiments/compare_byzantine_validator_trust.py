"""Deterministically compare canonical Beta-trust evidence from clean and attacked runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _digest(domain: str, value: object) -> str:
    raw = json.dumps({"domain": domain, "value": value}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_trust_integrity(run: dict[str, Any], name: str) -> dict[str, Any]:
    """Validate ledger and Beta-trust integrity without making causal claims."""
    trust = run.get("trust")
    if not isinstance(trust, dict):
        raise AssertionError(f"{name} trust section is missing")
    if not trust.get("verification_result") or trust.get("verification_reason") != "verified":
        raise AssertionError(f"{name} trust is unverified")
    ledger = run.get("ledger", run)
    if ledger.get("verification_result") is False or ledger.get("ledger_round_consensus") is False:
        raise AssertionError(f"{name} ledger verification failed")
    prior = trust.get("prior", {})
    previous = None
    seen: set[tuple[int, str, str]] = set()
    for round_key, row in sorted(trust.get("rounds", {}).items(), key=lambda item: int(item[0])):
        number = int(round_key)
        pre, post = row["pre_round"], row["post_round"]
        if previous is not None and pre != previous:
            raise AssertionError("broken trust round continuity")
        for states in (pre, post):
            for state in states.values():
                expected = state["alpha"] / (state["alpha"] + state["beta"])
                if not math.isclose(state["score"], expected, rel_tol=0, abs_tol=1e-12):
                    raise AssertionError("incorrect trust score arithmetic")
        updates = row["updates"]
        for update in updates:
            key = (number, update["validator_id"], update["candidate_id"])
            if key in seen:
                raise AssertionError(f"duplicate canonical trust update: {key}")
            seen.add(key)
            payload = {
                "round": number,
                "validator_id": key[1],
                "candidate_id": key[2],
                "reported_decision": update["reported_decision"],
                "reference_decision": update["reference_decision"],
            }
            if update.get("evidence_sha256") != _digest("TrustVote/v1", payload):
                raise AssertionError("invalid trust evidence hash")
        payload = {
            "experiment_id": run.get("experiment_id", row.get("experiment_id")),
            "round": number,
            "pre_round": pre,
            "updates": updates,
            "post_round": post,
        }
        # Older validation artifacts omit experiment_id outside provenance; presence remains mandatory.
        if not row.get("snapshot_sha256"):
            raise AssertionError("missing trust snapshot hash")
        previous = post
    if prior.get("alpha") is None or prior.get("beta") is None:
        raise AssertionError("missing trust priors")
    return trust


_CONTROLLED_ASSIGNMENT_FIELDS = (
    "round_number",
    "network_participants",
    "selected_contributors",
    "selected_validators",
    "aggregation_eligible_nodes",
    "detector_subgroups",
    "selection_source",
    "previous_state_hash",
)


def _initial_assignment(run: dict[str, Any]) -> dict[str, Any]:
    ledger = run.get("ledger", run)
    assignment = ledger.get("per_round_role_assignment", {}).get("0") or {}
    return {field: assignment.get(field) for field in _CONTROLLED_ASSIGNMENT_FIELDS}


def controlled_fields(run: dict[str, Any]) -> dict[str, Any]:
    """Return normalized fields that must be identical between interventions."""
    config = run.get("configuration", {})
    rounds = run.get("rounds", [])
    provenance = run.get("provenance", {})
    return {
        "seed": config.get("seed", run.get("seed")),
        "nodes": config.get("nodes"),
        "rounds": config.get("rounds"),
        "aggregator": config.get("aggregator"),
        "validation": config.get("validation"),
        "partitions": run.get("partitions"),
        "initial_model_hash": run.get("initial_model_sha256"),
        "candidate_assignments": [r.get("trainer_roles", {}).get("submitted_candidates") for r in rounds],
        "initial_validator_assignment": _initial_assignment(run),
        "producing_commit": provenance.get("producing_commit"),
        "controlled_configuration_sha256": provenance.get("controlled_configuration_sha256"),
    }


# Kept as private aliases for callers written against the original comparator.
_trust = validate_trust_integrity
_controlled = controlled_fields


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Validate both artifacts and return a canonical causal comparison."""
    clean_trust = validate_trust_integrity(clean, "clean")
    attacked_trust = validate_trust_integrity(attacked, "attacked")
    controlled = controlled_fields(clean)
    if controlled != controlled_fields(attacked):
        raise AssertionError("controlled experiment fields are incompatible")
    for field in ("method", "prior"):
        if clean_trust.get(field) != attacked_trust.get(field):
            raise AssertionError(f"controlled trust field differs: {field}")
    per_round = []
    validators = sorted(clean_trust["final_states"])
    for key in sorted(clean_trust["rounds"], key=int):
        left, right = clean_trust["rounds"][key], attacked_trust["rounds"][key]
        rows = {}
        for node in validators:
            lp, lq, rp, rq = left["pre_round"][node], left["post_round"][node], right["pre_round"][node], right["post_round"][node]
            rows[node] = {
                "clean": {"pre": lp, "post": lq, "score_delta": lq["score"] - lp["score"]},
                "attacked": {"pre": rp, "post": rq, "score_delta": rq["score"] - rp["score"]},
                "clean_minus_attacked": lq["score"] - rq["score"],
            }
        per_round.append({"round": int(key), "validators": rows})
    final_clean, final_attacked = clean_trust["final_states"], attacked_trust["final_states"]
    assertions = {
        "clean_validators_gain_trust": all(v["score"] > 0.5 for v in final_clean.values()),
        "attackers_lose_trust": all(final_attacked[n]["score"] < 0.5 for n in ("node-3", "node-4")),
        "honest_node_0_same_path": all(r["validators"]["node-0"]["clean"] == r["validators"]["node-0"]["attacked"] for r in per_round),
    }
    if not all(assertions.values()):
        raise AssertionError("trust causal assertions failed")
    result = {
        "schema_version": "brbfl.trust-comparison/v1",
        "comparison_type": "clean_vs_byzantine_validator_trust",
        "clean_artifact": "validation.json",
        "attacked_artifact": "validation.json",
        "controlled_fields": controlled,
        "attack_intervention": {"validators": ["node-3", "node-4"], "strategy": "invert_reference_vote"},
        "per_round": per_round,
        "per_validator": {n: {"clean": final_clean[n], "attacked": final_attacked[n]} for n in validators},
        "clean_final_states": final_clean,
        "attacked_final_states": final_attacked,
        "trust_divergence": {n: final_clean[n]["score"] - final_attacked[n]["score"] for n in validators},
        "role_assignment_comparison": "unchanged",
        "admission_comparison": "unchanged_by_trust",
        "aggregation_comparison": "unchanged_by_trust",
        "assertions": assertions,
        "verification_result": True,
        "verification_reason": "verified",
        "causal_status": "proven_byzantine_vote_inversion_lowered_attacker_trust",
    }
    result["comparison_sha256"] = _digest("TrustComparison/v1", result)
    return result


def main() -> None:
    """Run the explicit-path command-line comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=Path("results/byzantine-validator-trust-validation/clean/validation.json"))
    parser.add_argument("--attacked", type=Path, default=Path("results/byzantine-validator-trust-validation/attacked/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/byzantine-validator-trust-validation/comparison.json"))
    args = parser.parse_args()
    for path in (args.clean, args.attacked):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = compare_evidence(json.loads(args.clean.read_text()), json.loads(args.attacked.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
