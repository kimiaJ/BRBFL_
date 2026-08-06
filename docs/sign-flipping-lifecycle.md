# Sign-flipping update lifecycle

`TrainStage.execute` is the confirmed P2PFL local-training completion boundary: `learner.fit()` returns before the local model is submitted to `Aggregator.add_model`. Sign flipping formerly ran in `PoisonedLightningModel.get_parameters`, a read/serialization method called repeatedly by partial gossip and aggregation while those model values evolve. Consequently, producer/round transport observations were hashes of different partial aggregate states, not copies of one local update.

The corrected lifecycle calls `PoisonedLightningModel.publish_local_update` immediately after `learner.fit()`. It detaches the benign state-dict payload, hashes the sorted named parameters (name, dtype, shape, exact CPU bytes), applies the configured attack once, caches an independent attacked snapshot, and installs that snapshot before `Aggregator.add_model`. Aggregation observation verifies exact canonical equality with the cached publication.

Partial-model gossip now only observes `model.get_parameters()` after publication. A full payload equal to the cached attacked publication is identified as a copy; an evolving full partial aggregate is comparable but is not assigned that update identity. A send without a complete parameter payload is explicitly non-comparable. Transport recipients, timestamps, message identifiers, and object identities never participate in logical update identity, which is producer ID + training round + canonical benign snapshot hash.

## Windows CMD smoke reproduction

```cmd
cd C:\path\to\BRBFL_
uv sync --all-extras
uv run python -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_sign_flipping.yaml
uv run pytest test\brbfl\test_sign_flipping_validation.py
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall brbfl p2pfl test
```
