# Audit-only recovery evidence boundary

## Implemented scope

This change implements a process-local runtime API and ledger/trust integration.
The MNIST integration and YAML switch are now described in
[ca-recovery-workflow.md](ca-recovery-workflow.md). Tests exercise real detector,
gate, ledger, trust and CA code with small numeric models; they do not constitute
new MNIST results. The workflow barrier completes probes before aggregate publication.

`RuntimeLedgerConfig.recovery_probes_enabled` defaults to false. Enabling it
requires CA contributor selection with fixed sensitive roles, enabled detector,
recovery and suspicious trainer eligibility. The latter allows a recovered
excluded -> suspicious participant to provide ordinary evidence next round.

`recovery_probe_subjects(round)` uses prior finalized CA state and selects at
most one excluded trainer whose cooldown has elapsed. Selection rotates over
sorted eligible IDs using round modulo eligible count. This deterministic rule
has no bounded-wait guarantee when the eligible set changes; the current pilot
has one excluded trainer. Multiple-exclusion fairness needs separate evaluation.

`submit_recovery_probe(round, participant, parent, parameters)` runs after
ordinary admission, before round finalization. It requires the current parent,
a scheduled participant and at least three admitted model snapshots. It builds
an independent median reference from those admitted models; the probe never
influences that reference. The fixed committee evaluates the probe using the
existing sign-flipping detector and quorum. Assessment provenance binds the
reference, parent, round, assignment and submitted model hash.

The ledger stores probes in a separate collection/event. They never enter the
candidate, validator-admission or aggregate maps. Repeating an identical probe
within a round is idempotent; conflicting probes fail. Prior-round committee
evidence and late submissions fail. An omitted probe creates no positive evidence.
The trust runtime consumes committed probe decisions only at round finalization,
and CA updates affect the next assignment. Severe provenance, cooldown, trust
threshold and positive-history requirements still govern recovery.

## Verification and expected numeric fixture

Four numeric scenarios cover healthy recovery, continuing malicious probes,
missing probes, and an attack repeated after reentry. A six-node fixture starts
with attacks in rounds 1 and 2. With default cooldown and trust thresholds, healthy
probes occur in rounds 6–9: trust rises from 1/4 to 5/8. After the fourth positive
probe the participant becomes suspicious and can train in round 10. It becomes
observation after another positive round, then trusted after round 11. Continuing
attacks and missing probes leave it excluded. An attack after reentry excludes it
again. These are fixture expectations, not experimental MNIST measurements.

The tests also check stale parent, unauthorized subject, premature/late probe,
replayed committee evidence, conflicting/duplicate submission, immutable candidate
and admission membership, and unchanged aggregate eligibility during probe rounds.

## Workflow and remaining evaluation

See [ca-recovery-workflow.md](ca-recovery-workflow.md) for the integrated training
barrier, timeout behavior, separate probe accounting and runnable configurations.
Real MNIST recovery and persistent-attack behavior must be evaluated from the
uploaded receipts before drawing conclusions. Multi-exclusion scheduling and
fairness remain outside this one-excluded-trainer pilot.
