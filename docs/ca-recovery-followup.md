# Recovery followup protocol

## Scope fixed before new runs

Defense logic, topology, committee, data size, epochs, trust priors, bootstrap,
recovery thresholds and cooldown remain as in the twelve-round recovery pilot.
No threshold is adjusted in response to the seed-666 accuracy decrease.

Eleven new configs complete a five-scenario, three-seed matrix. Four seed-666
scenarios already exist in `results/ca-recovery/666/` and must not be repeated.
New outputs use `results/ca-recovery-followup/<seed>/<scenario>/`.

| Seed | New scenarios |
| --- | --- |
| 666 | relapse_enabled |
| 667 | relapse_enabled, transient_disabled, transient_enabled, persistent_enabled, clean_enabled |
| 668 | relapse_enabled, transient_disabled, transient_enabled, persistent_enabled, clean_enabled |

Relapse uses zero-based attack rounds `[0, 1, 9, 10, 11]`: the first two rounds,
then the tenth through twelfth. This schedule is chosen to exercise reentry seen
in the seed-666 pilot. It is **not** an independent prediction or a timing rule
that reads CA state. On other seeds the second episode may occur during probing
if reentry is later. Classify outcomes by actual state/selection at that time;
never claim an attack after reentry if the node had not yet reentered.

## Execution

From the existing full P2PFL environment and repository root:

```powershell
python -m brbfl.experiments.run_recovery_followup
```

This runs all eleven jobs sequentially, seed-666 relapse first, and stops if a
process fails or output receipts are missing/incomplete. It checks 12 rounds,
final consensus, ledger verification, configuration and producing-commit
consistency; it does not replace semantic evidence analysis. All requested output
folders are checked before training to prevent accidental overwrites.

To run just the first new pilot:

```powershell
python -m brbfl.experiments.run_recovery_followup --pilot-only
```

To continue after successful jobs already exist:

```powershell
python -m brbfl.experiments.run_recovery_followup --skip-completed
```

Existing completed outputs must match the expected effective configuration.
Incomplete folders cause an error and are never deleted automatically. Preserve
failed-run output for diagnosis; do not combine it with a later run in place.

Archive `results/ca-recovery-followup/` with all five files per run: validation,
manifest, global_metrics, messages and system_resources. The prior seed-666
archive is already available separately.

## Analysis to perform

- Verify partitions within seed and producer code provenance. Original seed-666
  outputs use the previous workflow commit; this change only adds preparation,
  execution helpers, configs and tests, with the training/defense implementation
  unchanged. Record both commits explicitly.
- Reconstruct finalized trust and CA; link prior snapshot to next-round roles.
  Count model probes separately from ordinary selected training and aggregation.
- For relapse, report actual reentry, second attack submission, rejection before
  aggregation, subsequent role restriction and probe behavior. Count no selection
  as nonselection, not another detected attack.
- Compare the transient pair by return time, continued exclusion, probe cost,
  admitted membership and local-test quality. Include every seed, including runs
  that do not recover in the horizon; report those as right-censored rather than
  removing or retuning them.
- Three paired seeds support descriptive mean, sample SD and individual results;
  they do not establish broad statistical superiority. The observed seed-666
  accuracy decline remains part of the results. Report current checks as pilot
  evidence under fixed honest committee assumptions, not general Byzantine safety.

Prepared configs and numeric tests are not new MNIST results. No training is
launched by the preparation module. Re-running preparation refuses existing paths.
