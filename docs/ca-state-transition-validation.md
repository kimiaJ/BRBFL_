# CA-4 deterministic causal validation

CA-4 uses two six-round, five-node smoke configurations. They control the seed,
producing commit, full topology, MNIST partition description, initial model,
FedAvg policy, reference-vote validation, Beta trust, CA transition policy,
observation bootstrap snapshot, and candidate assignments. Only the attacked
configuration enables the already validated `byzantine_validator` reference-vote
inversion intervention.

The expected attacked paths for `node-3` and `node-4` are suspicious after source
round 0, excluded after source round 1, and excluded through rounds 2--5. Honest
`node-0` remains under observation for rounds 0--1 and becomes trusted in round 2;
unevaluated `node-1` and `node-2` remain under observation. In the clean run,
evaluated nodes 0, 3, and 4 become trusted in round 2 while unevaluated nodes are
never promoted.

The artifact generator records finalized trust/evidence, all transition records,
snapshot provenance, eligibility rules, role assignments, aggregation identities,
and consensus hashes. The comparator regenerates each input artifact before it
compares controlled fields. It separately reports trust divergence, CA state
transition, and CA-driven role ineligibility; selection causality requires the role
assignment's source hash to equal the immediately resulting verified CA snapshot.

## Windows reproduction (attacked, clean, comparison)

Run these commands from the repository root in PowerShell:

```powershell
py -m brbfl.experiments.compare_ca_state_transition --config configs\smoke\mnist_byzantine_validator_ca_attacked.yaml --output results\byzantine-validator-ca\attacked\validation.json
py -m brbfl.experiments.compare_ca_state_transition --config configs\smoke\mnist_byzantine_validator_ca_clean.yaml --output results\byzantine-validator-ca\clean\validation.json
py -m brbfl.experiments.compare_ca_state_transition --attacked results\byzantine-validator-ca\attacked\validation.json --clean results\byzantine-validator-ca\clean\validation.json --output results\byzantine-validator-ca\comparison.json
```

Success is machine-readable as `verification_result: true` and
`causal_status: proven_ca_state_transitions_excluded_byzantine_participants`.
