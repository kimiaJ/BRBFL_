# CA-5 real-runtime validation

CA-5 is deliberately separate from the dependency-light CA-4 generator. The two
six-round configurations are `configs/smoke/mnist_ca_runtime_attacked.yaml` and
`configs/smoke/mnist_ca_runtime_clean.yaml`. They use the same five-node memory
topology, MNIST partitions, seed, model, FedAvg policy, bootstrap roles, trust
policy, and CA policy; only the already-supported Byzantine validator intervention
differs.

## Capture points

The MNIST entry point installs one `RuntimeLedgerAdapter` before learning. Candidate
owner callbacks commit candidate/vote facts, admission coordination commits the
decision map, verified installation callbacks commit the aggregate and installation,
and the network-wide round barrier finalizes the ledger. That same finalization
creates one trust snapshot, maps its authoritative vote agreements to CA evidence,
executes the synchronous transition, and commits its provenance. Opening the next
round passes that exact snapshot to `RoundRoleSelector.select_roles(context)` and
commits the immutable assignment before training and validation use it.

The output is marked `execution_mode: real_p2pfl_runtime` and contains the serialized
ledger, trust, CA, assignment, aggregation, installation, and cross-node model facts.
The CA-5 comparator rejects any other execution mode, missing production provenance,
non-final rounds, broken snapshot/assignment hashes, altered transition reasons, or a
single-node consensus claim. It validates each input before comparing controlled
configuration fields or making a causal claim.

The attacked validators follow `observation -> suspicious` after round 0 and
`suspicious -> excluded` after round 1, then remain excluded through rounds 2-5.
They are ineligible as validators beginning with round 1 and totally ineligible
beginning with round 2. Round 1 evidence is neutral because of that quarantine;
the configured one-round probation escalates the unresolved round 0 severe finding
without fabricating another validator event. Repeatedly evaluated honest validators can become trusted
after round 2; neutral participants cannot accumulate the positive counter.

## Windows execution order

Run from the repository root in PowerShell, with the project environment installed:

```powershell
Remove-Item -Force results\mnist-ca-runtime\attacked\validation.json -ErrorAction SilentlyContinue
Remove-Item -Force results\mnist-ca-runtime\clean\validation.json -ErrorAction SilentlyContinue
Remove-Item -Force results\mnist-ca-runtime\comparison.json -ErrorAction SilentlyContinue
uv run python -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_ca_runtime_attacked.yaml
uv run python -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_ca_runtime_clean.yaml
uv run python -m brbfl.experiments.compare_ca_runtime --clean results\mnist-ca-runtime\clean\validation.json --attacked results\mnist-ca-runtime\attacked\validation.json --output results\mnist-ca-runtime\comparison.json
```

Only a final comparison containing `verification_result = true`,
`execution_mode = real_p2pfl_runtime`, and
`causal_status = proven_runtime_ca_transitions_excluded_byzantine_participants`
establishes the real-runtime causal claim.
