# MNIST recovery-probe workflow pilot

## Execution boundary

With `ca.recovery_probes_enabled: true`, an excluded scheduled trainer fits a
local model using its current parent and local data in WaitAggregatedModelsStage.
It restores a deep copy of the parent model, waits for ordinary admission, and
submits the detached probe to the runtime API. Selected trainers wait at the
same probe barrier after admission, before publishing aggregate inputs. Thus no
new aggregate can overwrite a probe learner during fitting. Receiver installation
and the existing all-node round barrier then proceed normally.

Fit failure restores the parent and signals waiting nodes. Missing completion
or a timeout aborts the phase; it does not fabricate neutral/positive evidence or
allow late probes. This first workflow uses the existing aggregation timeout and
memory transport. A timed-out fit may still finish computing but cannot commit a
late probe or release aggregate publishers. The pure runtime's omitted-probe
fixture remains a separate model of neutral evidence, not a network timeout policy.

Probe training is recorded separately in each node's `trainer_role_evidence`
under `recovery_probe`, including parent/restored hashes, actual training,
submission, applied attack, decision hash and elapsed seconds. Existing
`actually_trained` continues to mean ordinary contributor training; it must not
be interpreted as all compute when probes are enabled. Probe evidence is present
in `validation.json` under both per-node selection and ledger `recovery_probes`.
These costs must be counted separately in comparisons.

## Predeclared behavior and prepared runs

The audited sign-flipping attacker accepts optional zero-based `active_rounds`.
When absent it attacks every ordinary/probe publication. `[0, 1]` means it attacks
only the first two rounds, including any probe scheduled in those rounds. It
then trains and publishes normally from its own data. The schedule is used only
by the attack producer; committee/selection/CA do not read it. Ordinary attack
counters exclude probe transformations, which have their own records.

Four seed-666, 12-round configs are prepared in `configs/ablation/ca-recovery/`:

| Config | Behavior | Recovery/probes |
| --- | --- | --- |
| transient_disabled_666 | attack rounds 0,1 | disabled |
| transient_enabled_666 | attack rounds 0,1 | enabled |
| persistent_enabled_666 | attack every publication | enabled |
| clean_enabled_666 | no attack | enabled |

Bootstrap is enabled in every configuration. Thresholds, committee and data
settings follow the previous pilot. The transient pair differs only in recovery,
probe activation and output location. No MNIST results are included in this PR.

Run from the repository root in the existing full P2PFL environment:

```powershell
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-recovery/transient_disabled_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-recovery/transient_enabled_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-recovery/persistent_enabled_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-recovery/clean_enabled_666.yaml
```

Archive `results/ca-recovery/666/` including each scenario's validation, manifest,
global_metrics, messages and system_resources files. Inspect actual detector
categories, probe/candidate separation, parent restoration and next-round state
before concluding recovery occurred. Real local MNIST updates may not follow the
all-positive numeric fixture; do not tune thresholds to force the expected result.
A single seed is a workflow pilot, not a statistical superiority comparison.
