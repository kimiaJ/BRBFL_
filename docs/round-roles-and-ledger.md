# Round-scoped roles and lifecycle ledger

## Ownership and terminology

A registered **capability** says what a participant is permitted to do; a selected
**round role** says what it is responsible for in one numbered round. A participant
may have several capabilities and responsibilities. In particular, being
validator-only in one round does not permanently prevent contribution in a later
round. The five-node Byzantine-validator assignments are controlled experiment
fixtures, not framework identities.

`RoundRoleSelector` is the selection boundary. `StaticRoundRoleSelector` bootstraps
round 0 and preserves existing YAML membership (and can express per-round fixtures).
A future CA selector will consume the same immutable `SelectionContext` and return a
`RoundRoleAssignment`; this milestone deliberately contains no trust score, CA state,
or dynamic policy.

The distributed workflow remains authoritative: it produces candidates, detector
results, canonical admission, the off-chain aggregate, and verified installations.
When enabled, the ledger validates those exact results and freezes their lifecycle;
contradictions raise errors rather than becoming another source of truth.

## Event boundary

The integration order is participant registration, role commitment, round opening,
candidate hash commitment, exact-candidate validator decisions, admission
finalization, exact-input aggregate commitment, installation confirmation, and round
finalization. The corresponding stable event names are `ExperimentStarted`,
`ParticipantRegistered`, `RoundRolesCommitted`, `RoundOpened`, `CandidateCommitted`,
`ValidatorDecisionCommitted`, `AdmissionFinalized`, `AggregateCommitted`,
`ModelInstallationConfirmed`, and `RoundFinalized`.

Canonical records use UTF-8 JSON, sorted dictionary keys, compact separators, explicit
JSON booleans/null, finite JSON numbers, and sorted set-like collections. SHA-256 is
domain-separated by record/event type. Every event contains a payload hash, logical
sequence number, previous-event hash, backend reference, and complete event hash.
Existing model-artifact hashes are referenced unchanged; models, tensors, gradients,
and datasets stay off-chain.

`InMemoryLedger` deterministically enforces the future backend contract and supports
tests, but it is neither distributed nor a real blockchain. Ledger consensus means
all peers verify the same committed event history; admission consensus means
validators finalized the candidate set; model consensus means required nodes
installed the same aggregate. These are separate assertions.

When disabled, new validation evidence reports `enabled: false` with no fabricated
transactions, blocks, receipts, or chain hash. When enabled, `validation_artifact()`
provides registrations, per-round assignments and hashes, event names and receipts,
the final chain hash, verification results, and finalized-round consensus.

## Next milestone

Implement a permissioned local Ethereum backend against the `BlockchainLedger`
contract: define and test a Solidity lifecycle contract, deploy it to an ephemeral
Anvil network, map transaction receipts without fallback, validate chain/network and
contract identity, reproduce idempotency and fail-closed semantics, compare emitted
events with runtime evidence, and add restart/replay and multi-client consensus tests.
No model or dataset content should be placed on chain.
