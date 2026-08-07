"""Compare clean and Byzantine-validator evidence using causal model lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brbfl.experiments.partition_evidence import EVIDENCE_SCHEMA_VERSION, canonical_partition_manifest

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
    "eligible_trainers",
    "dataset",
    "validation",
)

# ``trainer_roles`` is intentionally a mixed evidence object.  Keep this
# allowlist limited to facts fixed before the validator intervention; admission
# and installation evidence is checked later through candidate and lineage
# assertions.
CONTROLLED_TRAINER_ROLE_FIELDS = (
    "all_network_nodes",
    "configured_contributors",
    "configured_validators",
    "validator_only_nodes",
    "eligible_trainers",
    "selected_trainers",
    "actual_training_nodes",
    "submitted_candidates",
)
CONTROLLED_PER_NODE_SELECTION_FIELDS = (
    "actually_trained",
    "selected_trainers",
    "protocol_selected_trainers",
    "submitted_candidate",
    "eligible_trainers",
    "network_participants",
)


def _fail(message: str, key: tuple[int, str] | None = None) -> None:
    suffix = f": {key}" if key else ""
    raise AssertionError(message + suffix)


def _candidate_map(evidence: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    rows = evidence["validator_admission"]
    owners = [row for row in rows if row.get("current_node", row["candidate_node_id"]) == row["candidate_node_id"]]
    selected = owners or rows
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in selected:
        key = (row["round"], row["candidate_node_id"])
        if key in result and result[key] != row:
            _fail("ambiguous canonical evidence pairing", key)
        result[key] = row
    return result


def _round_map(evidence: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {row["round"]: row for row in evidence["rounds"]}


def _controlled_trainer_roles(round_row: dict[str, Any], round_number: int) -> dict[str, Any]:
    """Extract only pre-intervention trainer role and election evidence."""
    try:
        roles = round_row["trainer_roles"]
        controlled = {field: roles[field] for field in CONTROLLED_TRAINER_ROLE_FIELDS}
        per_node = roles["per_node_selection"]
        controlled["per_node_selection"] = {
            node_id: {field: selection[field] for field in CONTROLLED_PER_NODE_SELECTION_FIELDS} for node_id, selection in per_node.items()
        }
        return controlled
    except (KeyError, AttributeError) as exc:
        raise AssertionError(f"round {round_number} lacks controlled trainer-role evidence") from exc


def _lineage(round_row: dict[str, Any], round_number: int) -> dict[str, Any]:
    try:
        value = round_row["aggregation_lineage"]
        required = ("contributors", "input_hashes", "installed_global_model_sha256", "canonical_hash_source")
        if any(value.get(field) is None for field in required):
            raise KeyError
        return value
    except KeyError as exc:
        raise AssertionError(
            f"round {round_number} lacks required aggregation/model-lineage evidence; rerun clean and attacked artifacts"
        ) from exc


def _validate_run(name: str, evidence: dict[str, Any], rows: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    round_rows = _round_map(evidence)
    participants = [f"node-{index}" for index in range(evidence["configuration"]["nodes"])]
    final_hashes = evidence.get("per_node_final_installed_model_hashes", {})
    if sorted(final_hashes) != participants or not evidence.get("final_model_consensus"):
        raise AssertionError(f"{name} lacks complete network-wide final model consensus")
    integrity = []
    for key, row in sorted(rows.items()):
        parent = row.get("parent_global_model_sha256")
        if not parent:
            _fail(f"{name} candidate lacks required parent-model lineage", key)
        submitted = row["submitted_model_sha256"]
        if row["admitted"]:
            if not row.get("reached_aggregator_add_model"):
                _fail(f"{name} admitted candidate did not reach aggregation", key)
            if row.get("aggregation_input_sha256") != submitted or not row.get("aggregation_matches_submitted_snapshot"):
                _fail(f"{name} admitted aggregation-input hash differs from submitted hash", key)
        elif row.get("reached_aggregator_add_model") or row.get("aggregation_input_sha256") is not None:
            _fail(f"{name} rejected candidate reached aggregation", key)
        lineage = _lineage(round_rows[key[0]], key[0])
        in_inputs = key[1] in lineage["input_hashes"]
        in_contributors = key[1] in lineage["contributors"]
        if row["admitted"] != in_inputs or row["admitted"] != in_contributors:
            _fail(f"{name} admission disagrees with round aggregation contributors/inputs", key)
        if row["admitted"] and lineage["input_hashes"][key[1]] != submitted:
            _fail(f"{name} round aggregation input differs from submission", key)
        integrity.append({"round": key[0], "candidate_node_id": key[1], "status": "valid"})
    for round_number in sorted(round_rows):
        lineage = _lineage(round_rows[round_number], round_number)
        installation = round_rows[round_number].get("round_installation", {})
        installed = installation.get("installed_model_hashes", {})
        if sorted(installation.get("installation_nodes", ())) != participants or sorted(installed) != participants:
            raise AssertionError(f"{name} round {round_number} lacks every installation node")
        if any(digest != lineage["installed_global_model_sha256"] for digest in installed.values()):
            raise AssertionError(f"{name} round {round_number} has a stale or mismatched installed model")
        configured = set(evidence["configuration"]["validation"]["contributors"])
        if not set(lineage["contributors"]) <= configured:
            raise AssertionError(f"{name} round {round_number} aggregation includes a non-contributor")
        if round_number == 0:
            continue
        previous = _lineage(round_rows[round_number - 1], round_number - 1)["installed_global_model_sha256"]
        for key, row in rows.items():
            if key[0] == round_number and row["parent_global_model_sha256"] != previous:
                _fail(f"{name} candidate parent is not the prior installed global model", key)
    return {"status": "valid", "candidates": integrity}


def compare_evidence(clean: dict[str, Any], attacked: dict[str, Any]) -> dict[str, Any]:
    """Prove vote inversion is the first cause and classify downstream differences."""
    for name, evidence in (("clean", clean), ("attacked", attacked)):
        provenance = evidence.get("provenance")
        if not isinstance(provenance, dict):
            raise AssertionError(f"{name} artifact lacks provenance/evidence schema; rerun both artifacts from the same commit")
        schema = provenance.get("evidence_schema_version")
        if schema != EVIDENCE_SCHEMA_VERSION:
            raise AssertionError(
                f"{name} artifact evidence schema is incompatible: expected={EVIDENCE_SCHEMA_VERSION!r}, actual={schema!r}, "
                f"commit={provenance.get('producing_commit')!r}"
            )
    controlled_provenance = (
        "producing_commit",
        "controlled_configuration_sha256",
        "dataset_identity",
        "partitioning_strategy",
        "configured_seeds",
    )
    for field in controlled_provenance:
        if clean["provenance"].get(field) != attacked["provenance"].get(field):
            raise AssertionError(f"controlled provenance differs: {field}")
    for field in CONTROLLED_FIELDS:
        if clean["configuration"].get(field) != attacked["configuration"].get(field):
            raise AssertionError(f"controlled configuration differs: {field}")
    if clean.get("seeds") != attacked.get("seeds"):
        raise AssertionError("controlled seeds differ")
    clean_partitions = canonical_partition_manifest(clean.get("partitions"))
    attack_partitions = canonical_partition_manifest(attacked.get("partitions"))
    if clean_partitions != attack_partitions:
        clean_rows = {(row["node_id"], row["split"]): row for row in clean_partitions["entries"]}
        attack_rows = {(row["node_id"], row["split"]): row for row in attack_partitions["entries"]}
        keys = sorted(clean_rows.keys() | attack_rows.keys())
        first = next(key for key in keys if clean_rows.get(key) != attack_rows.get(key))
        raise AssertionError(
            f"controlled partitions differ at node/split={first}: clean={clean_rows.get(first)!r}, attacked={attack_rows.get(first)!r}"
        )
    for name, evidence, canonical in (
        ("clean", clean, clean_partitions),
        ("attacked", attacked, attack_partitions),
    ):
        recorded = evidence["provenance"].get("partition_manifest_sha256")
        if recorded != canonical["sha256"]:
            raise AssertionError(
                f"{name} partition manifest provenance is stale or corrupt: recorded={recorded!r}, actual={canonical['sha256']!r}"
            )
    clean_rounds, attack_rounds = _round_map(clean), _round_map(attacked)
    if clean_rounds.keys() != attack_rounds.keys():
        raise AssertionError("completed round identities differ")
    for round_number in clean_rounds:
        clean_roles = _controlled_trainer_roles(clean_rounds[round_number], round_number)
        attack_roles = _controlled_trainer_roles(attack_rounds[round_number], round_number)
        if clean_roles != attack_roles:
            raise AssertionError(f"controlled trainer_roles differ: round={round_number}")

    clean_rows, attack_rows = _candidate_map(clean), _candidate_map(attacked)
    if clean_rows.keys() != attack_rows.keys():
        raise AssertionError("candidate participation differs")
    clean_integrity = _validate_run("clean", clean, clean_rows)
    attack_integrity = _validate_run("attacked", attacked, attack_rows)

    round_zero_parents = {row["parent_global_model_sha256"] for key, row in clean_rows.items() if key[0] == 0}
    attacked_round_zero_parents = {row["parent_global_model_sha256"] for key, row in attack_rows.items() if key[0] == 0}
    if len(round_zero_parents) != 1 or round_zero_parents != attacked_round_zero_parents:
        raise AssertionError("initial model hashes differ")

    byzantine = set(attacked.get("malicious_node_ids", ())) & set(attacked["configuration"]["validation"]["validators"])
    first_vote = first_admission = first_aggregation = first_global = first_downstream = None
    downstream = []
    comparisons = []
    causal_model_diverged = False

    for key in sorted(clean_rows):
        baseline, attack = clean_rows[key], attack_rows[key]
        candidate_equal = baseline["submitted_model_sha256"] == attack["submitted_model_sha256"]
        parent_equal = baseline["parent_global_model_sha256"] == attack["parent_global_model_sha256"]
        if not causal_model_diverged and (not candidate_equal or not parent_equal):
            _fail("pre-intervention candidate/parent mismatch", key)
        if causal_model_diverged and not candidate_equal:
            item = {
                "round": key[0],
                "candidate_node_id": key[1],
                "clean_sha256": baseline["submitted_model_sha256"],
                "attacked_sha256": attack["submitted_model_sha256"],
                "clean_parent_sha256": baseline["parent_global_model_sha256"],
                "attacked_parent_sha256": attack["parent_global_model_sha256"],
                "expected_downstream_effect": True,
            }
            downstream.append(item)
            first_downstream = first_downstream or item

        clean_votes = {vote["validator_node_id"]: vote for vote in baseline["votes"]}
        attack_votes = {vote["validator_node_id"]: vote for vote in attack["votes"]}
        if clean_votes.keys() != attack_votes.keys():
            _fail("validator membership differs", key)
        changed_validators = []
        for validator in clean_votes:
            left, right = clean_votes[validator], attack_votes[validator]
            if left["reference_decision"] != right["reference_decision"]:
                _fail("reference decision differs", (*key, validator))
            reference = left["reference_decision"]
            if left["reported_decision"] != reference or left.get("attack_application_count", 0):
                _fail("clean vote is not an honest reference vote", (*key, validator))
            if validator in byzantine:
                if right["reported_decision"] == reference or right.get("attack_application_count") != 1:
                    _fail("Byzantine validator did not invert exactly once", (*key, validator))
                changed_validators.append(validator)
            elif right["reported_decision"] != reference or right.get("attack_application_count", 0):
                _fail("honest validator falsified a vote", (*key, validator))
        if changed_validators:
            vote_item = {"round": key[0], "candidate_node_id": key[1], "validators": changed_validators}
            first_vote = first_vote or vote_item
        if baseline["admitted"] != attack["admitted"]:
            if first_admission is None and (not candidate_equal or not changed_validators):
                _fail("changed admission is not explained by controlled vote inversion", key)
            admission_item = {
                "round": key[0],
                "candidate_node_id": key[1],
                "clean_admitted": baseline["admitted"],
                "attacked_admitted": attack["admitted"],
            }
            first_admission = first_admission or admission_item
        comparisons.append({"round": key[0], "candidate_node_id": key[1], "clean": baseline, "attacked": attack})

        last_in_round = key == max(item for item in clean_rows if item[0] == key[0])
        if last_in_round:
            clean_lineage = _lineage(clean_rounds[key[0]], key[0])
            attack_lineage = _lineage(attack_rounds[key[0]], key[0])
            if clean_lineage["contributors"] != attack_lineage["contributors"]:
                item = {"round": key[0], "clean": clean_lineage["contributors"], "attacked": attack_lineage["contributors"]}
                first_aggregation = first_aggregation or item
            if clean_lineage["installed_global_model_sha256"] != attack_lineage["installed_global_model_sha256"]:
                item = {
                    "round": key[0],
                    "clean": clean_lineage["installed_global_model_sha256"],
                    "attacked": attack_lineage["installed_global_model_sha256"],
                }
                first_global = first_global or item
                causal_model_diverged = True

    if first_admission is None:
        raise AssertionError("Byzantine votes did not change a real admission result")
    if first_aggregation is None or first_global is None:
        raise AssertionError("changed admission did not produce aggregation and global-model divergence")
    if not (first_vote["round"] == first_admission["round"] == first_aggregation["round"] == first_global["round"]):
        raise AssertionError("causal ordering is not proven in the first divergent round")

    return {
        "causal_status": "proven_byzantine_vote_inversion_changed_model_path",
        "controlled_pre_intervention_invariants": "valid",
        "canonical_partition_manifest_sha256": clean_partitions["sha256"],
        "initial_model_sha256": next(iter(round_zero_parents)),
        "first_vote_difference": first_vote,
        "first_changed_admission": first_admission,
        "first_aggregation_contributor_difference": first_aggregation,
        "first_global_model_hash_difference": first_global,
        "first_downstream_candidate_difference": first_downstream,
        "downstream_candidate_differences": downstream,
        "per_run_snapshot_integrity": {"clean": clean_integrity, "attacked": attack_integrity},
        "candidates": comparisons,
        "clean_final_model_sha256": clean["final_model_sha256"],
        "attacked_final_model_sha256": attacked["final_model_sha256"],
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
