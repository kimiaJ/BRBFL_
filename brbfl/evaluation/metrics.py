"""Metric-reporting guards shared by experiment frameworks."""


def backdoor_evaluation_configured(attack: object | None) -> bool:
    """Return whether an attack defines complete trigger-evaluation semantics."""
    return bool(attack and all(hasattr(attack, field) for field in ("trigger_size", "trigger_value", "target_class", "poison_batch")))
