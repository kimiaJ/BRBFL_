# Audit-only recovery evidence boundary

## Implemented scope

This change implements a process-local runtime API and ledger/trust integration.
It does not yet schedule or train probes in P2PFL's MNIST workflow, expose a YAML
switch, or produce new MNIST results. Tests exercise real detector, gate, ledger,
trust and CA code with small numeric models. The next integration step is a
workflow barrier that completes scheduled probes before round finalization.

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

## Remaining workflow work

The training workflow must provide a bounded probe-training phase after ordinary
admission and before installation finalizes the round, with timeout/missing-probe
semantics and separate actual-training/probe counters. Probe data must come from
the participant's local data and current model. No clean hidden model, attacker
identity, or forged favorable evidence may be used. Transient and intermittent
attack schedules and paired recovery-enabled/disabled MNIST configs follow that
integration. Do not enable this API by editing YAML alone.
