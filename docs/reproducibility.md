# Clean smoke experiment reproducibility

## Scope

This report covers two executions of `configs/smoke/mnist_numpy_clean.yaml` at seed
`666`. The smoke configuration deliberately uses the repository's small,
synthetic MNIST-shaped data set and the NumPy reference trainer. This makes the
check independent of network downloads, GPU kernels, worker scheduling, and
optional deep-learning packages. It tests the experiment controls rather than
claiming that every P2PFL backend is bitwise reproducible.

The command removes `results/reproducibility/`, then starts each run in a new
Python process with the same configuration and `PYTHONHASHSEED`. Each run
records full partition indices, selected node IDs, round metrics, parameter
arrays, and SHA-256 parameter fingerprints. The comparison uses exact equality
first and also reports a maximum absolute parameter difference with an absolute
tolerance of `1e-12`.

## Command

```bash
.venv/bin/python -m brbfl.experiments.smoke_reproducibility
```

## Observed result

Both executions completed. The committed machine-readable result is
`results/reproducibility/comparison.json`; the compact table is
`results/reproducibility/comparison.md`.

| Output | Result | Evidence |
|---|---|---|
| Data partitions | **Identical** | All three complete index lists and their SHA-256 hashes match. |
| Initial model parameters | **Identical** | Hash `86799c85bde7b5fdaae485a0ddbc66b2870ec72019c07d782edeb9ab568e9aab`; maximum absolute difference `0.0`. |
| Participating nodes | **Identical** | Rounds 0 and 1 both selected `node-1`, `node-2`. |
| Per-round loss | **Identical** | `2.161180815557979`, then `2.0143113735839764`. |
| Per-round accuracy | **Identical** | `0.24166666666666667`, then `0.4666666666666667`. |
| Final model parameters | **Identical** | Hash `074df04019764a4904c8d78433f9812f286fa5ed311bdb7e01c088b53a127b07`; maximum absolute difference `0.0`. |

No differing requested output was observed, so there was no nondeterministic
source to repair in this reference workload. Random data creation, IID
partitioning, parameter initialization, and participant selection all consume
one explicitly seeded NumPy generator. Separate processes prevent random state
from one run leaking into the next.

## Limits of the claim

The result establishes **exact reproducibility for this clean NumPy smoke
configuration on the recorded environment** (Python 3.12.13, NumPy 1.26.4,
Linux x86-64). It does not establish bitwise reproducibility for PyTorch,
TensorFlow, CUDA, distributed gRPC, or other CPU/BLAS implementations. Those
systems can use nondeterministic kernels, asynchronous message order, parallel
reductions, or different floating-point instruction orders. Results from such
environments should be called approximately equal only after recording explicit
absolute and relative tolerances; they must not be relabeled identical merely
because rounded metrics look the same.

## Re-running and interpreting output

Run the command above from the repository root. A successful report sets
`all_requested_outputs_identical` to `true`. Every requested field has one of
three classifications:

- `identical`: exact list/scalar/array equality;
- `approximately_equal`: parameter arrays are not exact but their maximum
  absolute difference is at most the recorded tolerance;
- `different`: neither exact nor within tolerance.

The two committed `run.json` files retain human-readable inputs, choices,
metrics, and hashes. During a local run, adjacent `parameters.npz` files retain
the actual initial and final arrays so the comparison is not based on hashes or
rounded text alone. Those generated binary archives are intentionally not
version-controlled: this keeps pull requests compatible with review systems
that accept text artifacts only. The committed hashes and zero maximum
differences preserve the result of the parameter comparison.
