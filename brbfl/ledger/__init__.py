"""Backend-independent ledger API and deterministic memory backend."""

from brbfl.canonical import canonical_bytes, canonical_hash
from brbfl.ledger.base import BlockchainLedger, EventType, LedgerEvent, LedgerReceipt
from brbfl.ledger.memory import InMemoryLedger


def create_ledger(*, enabled: bool, backend: str, ledger_identifier: str = "memory") -> BlockchainLedger | None:
    """Create the explicitly configured backend without fallback."""
    if not enabled:
        return None
    if backend == "memory":
        return InMemoryLedger(ledger_identifier)
    raise ValueError(f"unsupported blockchain ledger backend: {backend}")


def disabled_ledger_artifact() -> dict[str, object]:
    """Describe disabled recording without fabricating receipts or chain data."""
    return {"enabled": False, "backend": None, "ledger_identifier": None, "event_receipts": [], "final_event_chain_hash": None}


__all__ = [
    "BlockchainLedger",
    "EventType",
    "InMemoryLedger",
    "LedgerEvent",
    "LedgerReceipt",
    "canonical_bytes",
    "canonical_hash",
    "create_ledger",
    "disabled_ledger_artifact",
]
