# Controlled MNIST backdoor validation

The genuine trigger is a 3 by 3 square in the bottom-right corner. On MNIST's
28 by 28 grid its exact zero-based coordinates are every `(row, column)` pair
where both row and column are 25, 26, or 27. The controlled configurations use
the white raw-MNIST pixel value `255.0`. If inputs are normalized, the implementation writes
`(trigger_value - mean) / std`, not the raw value.

Node 1 is the only malicious node. Once, after deterministic partitioning and
before node creation, 30 percent of its local training examples are copied,
triggered, and relabelled to target digit 2. The source and benign partitions
are not modified.

Genuine all-to-one ASR is evaluated on clean test examples whose original
label is not 2. Every eligible image receives the same trigger while its
original label is retained separately:

`ASR = triggered eligible examples predicted as 2 / eligible triggered examples`.

Clean accuracy and loss always use the unmodified test images. A zero-sized
eligible set has count zero and ASR `0.0`.

Windows CMD reproduction from the repository root:

```bat
.venv\Scripts\python.exe -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_backdoor_clean.yaml
.venv\Scripts\python.exe -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_backdoor.yaml
.venv\Scripts\python.exe -m brbfl.experiments.compare_backdoor
```
