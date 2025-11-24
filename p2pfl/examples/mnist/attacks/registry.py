# attacks/registry.py
from typing import Dict, Any
import threading

_registry: Dict[str, Any] = {}
_lock = threading.Lock()

def register_attack(addr: str, attack) -> None:
    with _lock:
        _registry[addr] = attack

def get_attack(addr: str):
    with _lock:
        return _registry.get(addr)

def clear_attacks():
    with _lock:
        _registry.clear()