# MNIST attack lifecycle

`brbfl.attacks` is the canonical import location for every attack implementation and for the lifecycle API. The central `ATTACK_REGISTRY` maps configuration names to constructors; experiment code must use `create_attack()` rather than branching on attack-name strings.

For every MNIST node, the lifecycle is:

1. **Configuration:** `none` creates no attack. An enabled name creates exactly one attack object for an adversarial node.
2. **Dataset preparation:** `prepare_dataset()` invokes `poison_data()` once, after partitioning and before the dataset is passed to the node. Attacks without that hook leave the partition unchanged.
3. **Local training:** legacy online backdoor implementations retain their existing `poison_batch()` algorithm. `poison_training_batch()` is their single call site and runs once per training batch, immediately before the forward pass. This compatibility hook does not also call `poison_data()`.
4. **Update snapshot:** after training, `PoisonedLightningModel.record_local_update()` captures one detached, owning snapshot. The sign-flipping audit derives its logical ID from the producer, round, and canonical hash of that snapshot, transforms a deep copy exactly once, and retains both immutable snapshots.
5. **Update transmission:** each recognized local-update serialization receives an independent deep copy of the cached transformed snapshot. Per-recipient serialization never recaptures or transforms the live model, and a consumed hook observation cannot be reused by an unrelated transmission.
6. **Aggregation observation:** changing partial aggregates are classified separately. They can invoke the hook for observation, but are not grouped with or transformed as locally trained updates merely because producer and round match.
7. **Isolation:** the node-address registry associates that one object with its node and is cleared before each experiment, so clean and malicious runs cannot leak state into one another.

Attack algorithms intentionally remain unchanged. New integrations should add their constructor to `ATTACK_REGISTRY` and implement only the hooks appropriate to their lifecycle stages.
