# Controlled P2PFL MNIST label-flipping validation

## Scope and controls

This validation is deliberately separate from the deterministic NumPy check in
`docs/reproducibility.md`. It runs the real P2PFL memory transport, PyTorch
learner, and `p2pfl/MNIST` dataset. The clean and attacked YAML files both use
seed 666, three nodes, two rounds, one epoch, the same 1/50-size deterministic
IID partitions, a full topology, and FedAvg. Outputs are isolated under
`results/label-flipping-validation/clean` and `attacked`.

The attacked run assigns node 1 as malicious and maps source label 1 to target
label 7. Poisoning occurs exactly once during dataset preparation: after the
deterministic partition is made and before the node is created. The existing
`LabelFlippingAttack.poison_data` algorithm is unchanged.

## Reproduction

Install the PyTorch extra and run from the repository root:

```bash
uv sync --extra torch
rm -rf results/label-flipping-validation
.venv/bin/python -m p2pfl.examples.mnist.mnist --config configs/smoke/mnist_clean.yaml
.venv/bin/python -m p2pfl.examples.mnist.mnist --config configs/smoke/mnist_label_flipping.yaml
.venv/bin/python -m brbfl.experiments.compare_label_flipping
```

Each run writes its manifest, raw P2PFL metric CSVs, and `validation.json`.
Validation records label counts, changed indices, before/after partition hashes,
lifecycle stage, participants, malicious participants, attack counts, available
per-node metrics, and the final model hash. `comparison.json` normalizes P2PFL's
test loss and metric names into per-round clean/attacked loss and accuracy.

## Attack-execution proof

The focused tests establish that a clean configuration changes zero labels,
only the malicious partition changes, source label 1 becomes label 7, the
benign partition remains unchanged, one offline application is recorded, and
the source dataset is not modified in place. A real run repeats these checks
against every complete partition and aborts rather than writing misleading
evidence if an unexpected label changes.

## Observed performance effect

No real-run performance result is committed from this environment. The required
PyTorch dependency could not be downloaded because the package tunnel rejected
the `nvidia-nccl-cu12` wheel, and the initially supplied environment does not
contain Torch. Consequently this report **does not claim that label flipping is
effective**. Run the commands above in an environment with the locked PyTorch
extra and consult `comparison.json`; a measurable non-zero loss or accuracy
difference is evidence of an observed effect for this workload, not a general
effectiveness result.

## Smoke-workload limitations

Two rounds, three nodes, one malicious partition, and reduced local data have
low statistical power. PyTorch and asynchronous P2P message ordering are not
claimed to be bitwise deterministic. The partition seed and experimental
controls make the inputs comparable, while the recorded participants, metrics,
and hashes make deviations visible. A larger repeated study with uncertainty
estimates would be needed to support an effectiveness claim.
