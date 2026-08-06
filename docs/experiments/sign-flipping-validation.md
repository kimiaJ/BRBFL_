# Controlled P2PFL MNIST sign-flipping validation

## Inspected behavior and scope

The existing `SignFlippingAttack` transforms every serialized model parameter
(weights and biases, in `named_parameters()` order) using
`attacked = scale * original`; this smoke configuration uses `scale = -3.0`.
It runs at update transmission, after local training and before FedAvg
aggregation. It does not modify the dataset or run before training. The attack
algorithm is preserved; a decorator copies the update and records evidence
around its existing `manipulate_update()` method.

The clean and attacked runs use seed 666, three nodes, two rounds, one epoch,
the same deterministic 1/50 MNIST partitions and initial model construction,
memory transport, full topology, participants, and FedAvg. Only node-1 is
malicious. Outputs are isolated under `results/sign-flipping-validation`, so
the label-flipping artifacts remain untouched.

## Reproduction

POSIX shell from the repository root:

```bash
uv sync --extra torch
rm -rf results/sign-flipping-validation
.venv/bin/python -m p2pfl.examples.mnist.mnist --config configs/smoke/mnist_sign_flipping_clean.yaml
.venv/bin/python -m p2pfl.examples.mnist.mnist --config configs/smoke/mnist_sign_flipping.yaml
.venv/bin/python -m brbfl.experiments.compare_sign_flipping
```

Exact Windows CMD commands from the repository root:

```bat
uv sync --extra torch
if exist results\sign-flipping-validation rmdir /s /q results\sign-flipping-validation
.venv\Scripts\python.exe -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_sign_flipping_clean.yaml
.venv\Scripts\python.exe -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_sign_flipping.yaml
.venv\Scripts\python.exe -m brbfl.experiments.compare_sign_flipping
```

Each `validation.json` records participants, malicious participants, available
per-node loss/accuracy, application counts, and the final model hash. Each
malicious event records parameter names and shapes, pre/post hashes, L2 norms,
cosine similarity, scale, tolerance, maximum formula error, and three compact
sampled values per tensor. No parameter archive is stored. The comparison
utility rejects a clean attack, a benign-node transformation, mismatched
participants, an eligible node-1 update that bypasses the hook, or equal final
hashes. Eligibility is observed rather than inferred from the configured
round count: the node must complete local training and attempt to serialize an
outbound update in that framework round. A terminal round with evaluation but
no outbound update and a round in which node-1 was not selected are therefore
not fabricated as attack applications.

The lifecycle trace also documents a P2PFL detail that caused the previous
round-1 failure. FedAvg replaces the learner model after round 0 by calling
`build_copy()`. The poisoned wrapper previously lost its `node_addr` registry
key in that copy, so round 1 trained a distinct local model but its parameter
serialization could no longer find the attack hook (case A, not terminal-round
behavior or deduplication). The wrapper now carries that key into each
aggregated replacement. Framework training rounds are zero-based; P2PFL
increments the state after gossip and performs terminal evaluation at the
incremented value. Thus `rounds: 2` normally trains rounds 0 and 1 and may also
produce an evaluation event at framework value 2; these are separate from
logical attack applications and per-recipient network transmissions.

## Interpretation and limitation

Passing validation proves that sign flipping executed exactly as configured
and that its input remained available unchanged. Differences in per-round loss,
accuracy, or final hashes describe the observed effect only. A two-round,
three-node, reduced-data smoke run has low statistical power and cannot by
itself establish meaningful attack effectiveness. Do not claim degradation
unless the generated comparison actually shows it, and use repeated longer
runs with uncertainty estimates for an effectiveness study.

The former unconditional `backdoor_asr` was only the fraction of triggered
predictions equal to class 2, even when no backdoor was configured. ASR is now
suppressed unless an attached attack explicitly supplies trigger size/value,
target class, and batch-poisoning semantics. Historical JSON is not rewritten.
