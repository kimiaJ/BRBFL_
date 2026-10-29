# Contributor selection ablation

This experiment compares trainer selection with the **same** sign-flipping
model detector, coordinate-median reference, validation quorum, trust accounting,
and fixed honest validators and specialist committee (`node-4`, `node-5`).
It does not compare dynamic validator selection against dynamic trainer selection.

| Display name | Selection strategy | Trainer eligibility after bootstrap |
| --- | --- | --- |
| Static | `static` | All four configured trainers |
| Trust-only | `trust_contributors` | Prior finalized Beta score >= 0.5 |
| CA-aware | `ca_contributors` | Existing CA eligibility and state priority, then prior trust |

The existing `trust_ranked` validator selector and `ca_state` multi-role selector
remain available. `ca_contributors` freezes validators and detector committees
while preserving the current trainer state rules. This is an explicit ablation,
not a proposed general defense for malicious fixed committee members.

## Policy fixed before MNIST execution

The first round uses the same four trainers. Trust starts at alpha=beta=1.
Trust-only uses threshold **0.5**, inclusive, chosen as the neutral Beta prior,
not fitted to an accuracy result. It filters the configured trainer pool without
consulting CA, attack identities, current-round evidence, or hidden clean models.
All eligible trainers are retained; canonical role assignments sort their IDs.
There is no replacement trainer pool in this six-node smoke experiment.

Both adaptive variants must retain at least three trainers for the unchanged
median reference. Insufficient membership raises; low-trust or excluded nodes
are never silently readmitted. CA permits suspicious trainers, uses the existing
severe probation rule, and disables recovery. These different response policies
are part of the experiment, **not evidence that one method is superior**.

Numeric integration tests exercise these expected selections for an always
sign-flipping trainer: Static selects it in all three rounds; Trust-only stops
selecting it in the second round after score 1/3; CA-aware stops in the third
round after the suspicious -> excluded transition. These are **test fixtures**,
not new MNIST results. In the clean fixture all four trainers remain eligible.

The detector now also operates with static and trust-ranked selection. Its
classification thresholds, reference construction and committee quorum are
unchanged. Fixed committees are chosen deterministically from registered
specialist capabilities. Legacy CA selection continues rotating its committees.

## Prepared matrix and execution

`configs/ablation/selection/` contains 18 configs: three modes, clean/attacked,
and seeds 666, 667 and 668. Each uses MNIST IID reduced data, six nodes, three
rounds, one local epoch, FedAvg and a process-local memory ledger. The attack is
full-model sign flipping with scale -3 on node-1. Outputs go into distinct
`results/selection-ablation/<seed>/<strategy>/<scenario>/` directories.

Use the existing project environment with PyTorch and P2PFL dependencies. First
run the six seed-666 configs individually, starting with:

```powershell
python -m p2pfl.examples.mnist.mnist --config configs/ablation/selection/static_clean_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/selection/static_attacked_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/selection/trust_contributors_clean_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/selection/trust_contributors_attacked_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/selection/ca_contributors_clean_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/selection/ca_contributors_attacked_666.yaml
```

Advance to seeds 667/668 only after inspecting the pilot receipts. Training is
not launched by the config generator. To prepare another matrix into a **new**
folder (existing files are never overwritten):

```powershell
python -m brbfl.experiments.prepare_selection_ablation --output configs/ablation/selection-next --seeds 666 667 668
```

## Evidence before conclusions

Check commit and partition provenance across each paired seed; candidate,
committee, admission and aggregate hashes; exact selected/actual training and
submitted membership; and installation by every node. Round assignment receipts
include policy and prior trust provenance. CA-aware additionally records its CA
snapshot provenance. No next-round trust selection is allowed before prior trust
finalization.

Count rejection and false rejection **per submitted candidate**. Count attacker
training/selection rounds separately; nonselection is not another detector true
positive. Report trainer counts and admitted aggregate membership alongside any
quality or runtime metrics: removing a trainer also changes training workload.
Validate the timing and evaluation population of accuracy/loss before calling
an existing logged metric global post-aggregation accuracy. Three rounds and
three seeds are an initial pilot, not the final statistical evaluation.

Do not use `compare_model_detector_runtime` as the three-method comparator: that
older checker intentionally expects CA-driven exclusion in its attacked input.
A dedicated cross-method results checker and metric extraction remain the next
step after the pilot artifacts are available.
