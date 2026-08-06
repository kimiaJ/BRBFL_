"""Central attack construction and node-association registry."""

from collections.abc import Callable, Mapping
from threading import Lock
from typing import Any

Attack = Any
AttackFactory = Callable[[Mapping[str, Any]], Attack]


def _label(p):
    from .label_flipping import LabelFlippingAttack
    return LabelFlippingAttack(flip_map=dict(p.get("flip_map", {})))


def _sign(p):
    from .sign_flipping import SignFlippingAttack
    return SignFlippingAttack(scale=float(p.get("scale", -3.0)))


def _scale(p):
    from .scale import ScaleAttack
    return ScaleAttack(factor=float(p.get("scale_factor", 3.0)), apply_on=p.get("scale_on", "delta"))


def _construct(module: str, class_name: str, **defaults):
    def factory(parameters):
        from importlib import import_module
        cls = getattr(import_module(f"brbfl.attacks.{module}"), class_name)
        return cls(**defaults)
    return factory


ATTACK_REGISTRY: dict[str, AttackFactory] = {
    "label_flipping": _label,
    "sign_flipping": _sign,
    "scale": _scale,
    "backdoor": _construct("backdoor", "BackdoorAttack", trigger_size=4, target_class=2, poison_rate=0.3),
    "model_replacement": _construct(
        "model_replacement", "ModelReplacementAttack", scaling_factor=3.0, trigger_size=16, target_class=2, poison_rate=1
    ),
    "sybil_backdoor": _construct("sybil_backdoor", "SybilBackdoorAttack", trigger_size=16, target_class=2, poison_rate=1.0),
    "free_rider": _construct("free_rider", "FreeRiderAttack", mode="scale", scale=0.01),
    "delay_drop": _construct("delay_drop", "DelayDropAttack", mode="drop", drop_rate=0.8),
    "colluding_backdoor": _construct("colluding_backdoor", "ColludingBackdoorAttack", scale_factor=20, poison_rate=1.0, trigger_size=48),
}
_node_attacks: dict[str, Attack] = {}
_lock = Lock()


def create_attack(name: str, parameters: Mapping[str, Any] | None = None) -> Attack | None:
    """Build an attack; ``none`` is the clean configuration."""
    if name == "none":
        return None
    try:
        return ATTACK_REGISTRY[name](parameters or {})
    except KeyError as exc:
        raise ValueError(f"Unknown attack: {name}") from exc


def register_attack(addr: str, attack: Attack) -> None:
    """Associate an attack instance with one node address."""
    with _lock:
        _node_attacks[addr] = attack


def get_attack(addr: str | None) -> Attack | None:
    """Find the attack associated with a node address."""
    with _lock:
        return _node_attacks.get(addr) if addr is not None else None


def clear_attacks() -> None:
    """Clear associations before starting another experiment."""
    with _lock:
        _node_attacks.clear()
