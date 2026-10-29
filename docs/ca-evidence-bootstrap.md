# Evidence-based CA bootstrap experiment

## Contract and scope

The pilot starts every participant in observation. The ordinary promotion rule
requires trusted neighbors, so it cannot produce the first trusted participant.
This opt-in policy adds an explicitly bounded initial exception to that rule:

- `ca.policy.evidence_bootstrap_rounds: 3` enables it only for finalized source
  rounds 0, 1, 2. Zero disables it (the original pilot).
- An observation participant still needs current positive evidence, three
  consecutive positive rounds and trust >= 0.70 (existing promotion thresholds).
- Only the neighbor requirement is waived in that window. Normal supported
  promotion takes precedence; otherwise the reason is
  `observation_evidence_bootstrap`.
- Suspicious/excluded participants cannot use this exception to recover. Negative
  and severe evidence retain their existing precedence. Neutral evidence breaks
  the consecutive positive history. No evidence means no promotion.
- From source round 3 onward, promotion requires the original neighbor support.
  If no participant qualified during the window, the system can remain without
  trusted participants; no automatic fallback or extension is performed.
- Runtime evidence is obtained from finalized trust/detector records. Resulting
  CA state first affects selection in the following round. All transitions read
  the same prior snapshot, so simultaneous promotions cannot support one another.

This is a new research policy, not a correction to the reported experiment.
The initial exception relies on the existing honest fixed committee and detector
assumptions. A participant that behaves well during bootstrap can still attack
later. Trusted status is behavioral evidence, not proof of honesty or Sybil
resistance. No attacker identity or hidden clean model is an input to the rule.
An enabled policy uses the v2 hash domain and includes the window. Disabled
policies preserve the exact v1 policy and snapshot hashing.

## Controlled pilot (prepared, not executed on MNIST)

Four seed-666 configurations in `configs/ablation/ca-bootstrap/` compare the
existing policy with the new policy in clean and persistent sign-flipping cases.
Both use four rounds so the first selection after promotion can be inspected.
All other settings match the previous contributor-selection pilot, including
fixed validators/committee and disabled recovery. Outputs have new paths.

```powershell
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-bootstrap/baseline_clean_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-bootstrap/evidence_clean_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-bootstrap/baseline_attacked_666.yaml
python -m p2pfl.examples.mnist.mnist --config configs/ablation/ca-bootstrap/evidence_attacked_666.yaml
```

Check finalized evidence, trust, promotion reason, previous-generation neighbors,
and next-round assignment provenance. In the controlled numeric integration
fixture, clean participants promote after source round 2; the persistent attacker
is excluded and never promoted. MNIST detector behavior must be inspected from
its actual receipts. No accuracy benefit is predicted: selecting every eligible
trainer can leave aggregate membership identical. This experiment establishes
bootstrap operation, not the benefit of neighborhood or recovery.

## Recovery contract still to implement

Existing unit tests cover numeric recovery but excluded trainers currently have
no fresh model-evidence route. A separate integration change needs bounded,
audit-only probation submissions evaluated against the current admitted cohort's
reference. Such submissions must never enter that round's aggregation or count
as selected trainers. Evidence must be bound to participant, current parent model
and round, finalized once through the same committee/quorum, then applied to
trust and CA for later rounds. Cooldown and positive-history thresholds still
apply. Missing evidence remains neutral; replayed evidence cannot advance history.

Before enabling recovery, test rejected probes, missing/replayed probes,
insufficient reference membership, repeated attacks after reentry, deterministic
probe scheduling and separate probe cost accounting. This PR does not implement
that route or claim end-to-end recovery. Follow with transient and intermittent
attack schedules only after the probe boundary is verified.
