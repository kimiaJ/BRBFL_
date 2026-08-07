# Byzantine validator milestone

## Architecture and admission boundary

The pre-existing `VoteTrainSetStage` votes only on *which nodes train*. It is
not an update-validation committee. Before this milestone, a locally trained
model flowed directly from `TrainStage` to `aggregator.add_model()`, while a
received partial model flowed through `PartialModelCommand.execute()` directly
to the same method. No hook could reject a submitted update; existing attack
audits observed aggregation but did not control admission. Validators and
contributors execute as node threads in the same experiment process for the
memory smoke configuration (Ray may still be used by learners).

The smoke manifests set the generic `eligible_trainers` policy to the
configured contributors. `VoteTrainSetStage` intersects its normal reachable
candidate pool with that allowlist before nominations are sampled; with a
train-set size of five and three eligible nodes, every node selects exactly
`node-0`, `node-1`, and `node-2`. Without the optional policy, the historical
all-neighbor election is unchanged. `TrainStage` independently fails before
`learner.fit()` if corrupt state routes a selected ineligible node into
training. Validator-only nodes stay connected and follow the non-trainer
workflow.

`ValidatorSubgroupGate` is the new reusable admission boundary at both real
call sites. It snapshots the submitted parameters, calculates each reference
decision, publishes votes, and returns the admission result. An accepted model
is hash-checked immediately before the real `add_model()` call. A rejected
model is sent to `Aggregator.reject_model()` so it is counted as handled for
round completion but is never included in aggregation.

## Controlled rule and voting semantics

The smoke configurations separate contributor (`node-0`, `node-1`, `node-2`)
and validator (`node-0`, `node-3`, `node-4`) roles. The attacked validators are
`node-3` and `node-4`, group `mnist-smoke-byzantine-validators`, using
`invert_reference_vote` exactly once per candidate. All contributors remain
benign.

The deterministic reference rule requires finite parameters, an L2 norm no
larger than the configured bound, and absence from the configured synthetic
rejection-case list. The synthetic case is applied at the real gate and never
changes model parameters. All three eligible validators vote; quorum is three
and two accept votes are required. Thus ties fail the threshold, missing votes
fail quorum, and both conditions fail closed. Ineligible and duplicate votes
raise descriptive errors. A rejected contribution is explicitly receipted and
cannot reach `add_model()`.

## Reproduction (Windows CMD)

```bat
.venv\Scripts\python.exe -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_byzantine_validator_clean.yaml
.venv\Scripts\python.exe -m p2pfl.examples.mnist.mnist --config configs\smoke\mnist_byzantine_validator.yaml
.venv\Scripts\python.exe -m brbfl.experiments.compare_byzantine_validator
```
