# Three-round model detector exclusion smoke test

These paired runs use six network nodes, four bootstrap contributors and two
validators. One contributor (node-1) reverses its entire published model with
scale -3. After two severe decisions it becomes Excluded; round 2 must select
only node-0, node-2 and node-3. The clean counterpart keeps all four contributors.

Four initial contributors are intentional: the observable median reference still
requires at least three submitted models after exclusion. No hidden clean update
or attack identity is used to construct that reference. Its honest-majority
assumption remains necessary. Suspicious contributors may train in round 1;
their candidate still passes through the detector admission gate.

Run each command to completion in the existing environment:

```powershell
python -m pytest -o addopts="" -q test/brbfl/test_model_detector_exclusion.py
python -m p2pfl.examples.mnist.mnist --config configs/smoke/mnist_model_detector_exclusion_clean.yaml
python -m p2pfl.examples.mnist.mnist --config configs/smoke/mnist_model_detector_exclusion_attacked.yaml
python -m brbfl.experiments.compare_model_detector_runtime --clean results/mnist-model-detector-exclusion/clean/validation.json --attacked results/mnist-model-detector-exclusion/attacked/validation.json --output results/mnist-model-detector-exclusion/comparison.json
```

The comparison requires three finalized runtime rounds, matching controlled
configuration/commit/partitions, committee majority evidence tied to committed
candidates, exact admitted aggregate membership, network-wide installation,
contributor trust reduction, CA escalation and actual round-2 nonparticipation.
It rejects inconsistent evidence even if success flags are present. The result
is a bounded artifact consistency check, not independent replay of training or
proof of general Byzantine robustness. The ledger backend is process-local
memory, not an external blockchain deployment.

Expected comparison status: `verified_three_round_detector_exclusion`, with
`verification_result: true`. The nine new unit/integration tests use numeric
fixtures and mocked workflow boundaries; passing them is not an MNIST run.

The original two-round smoke configurations remain available. Output paths for
these new experiments are separate, preserving earlier results.
