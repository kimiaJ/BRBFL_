"""Canonical public API for BRBFL attacks and their lifecycle."""
from .base import BaseAttack
from .lifecycle import poison_model_update, poison_training_batch, prepare_dataset
from .registry import ATTACK_REGISTRY, clear_attacks, create_attack, get_attack, register_attack

__all__ = [
    "ATTACK_REGISTRY",
    "BaseAttack",
    "clear_attacks",
    "create_attack",
    "get_attack",
    "poison_model_update",
    "poison_training_batch",
    "prepare_dataset",
    "register_attack",
]
