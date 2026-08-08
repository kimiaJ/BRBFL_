"""Canonical JSON encoding and domain-separated SHA-256 hashing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


def _normalise(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalise(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_normalise(item) for item in value), key=lambda item: canonical_bytes(item))
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical numbers must be finite")
        return int(value) if value == value.to_integral() else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical numbers must be finite")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON as UTF-8 with sorted keys and no insignificant whitespace."""
    return json.dumps(_normalise(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_hash(domain: str, value: Any) -> str:
    """Hash a canonical value with an unambiguous UTF-8 domain prefix."""
    if not domain:
        raise ValueError("hash domain must not be empty")
    return hashlib.sha256(b"BRBFL\x00" + domain.encode("utf-8") + b"\x00" + canonical_bytes(value)).hexdigest()
