# attacks/model_wrapper.py
from typing import Optional, List
import numpy as np
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from brbfl.attacks.base import BaseAttack



class AttackableLightningModel(LightningModel):
    """
    A proper LightningModel subclass that wraps the base model and applies attacks.
    Must inherit from LightningModel to satisfy p2pfl's type checks and lifecycle.
    """
    def __init__(self, base_model: LightningModel, attack: Optional[BaseAttack] = None):
        """
        Args:
            base_model: The original LightningModel from model_build_fn()
            attack: Optional attack to apply on get_parameters()
        """
        # Copy essential attributes from base_model
        self.__dict__.update(base_model.__dict__)
        self._attack = attack
        # Do NOT call super().__init__ — we'll delegate

    def get_parameters(self) -> List[np.ndarray]:
        """Override to apply model poisoning."""
        params = super().get_parameters()  # This calls base_model's get_parameters
        if self._attack:
            params = self._attack.manipulate_update(params)
        return params

    # Explicitly delegate required methods
    def get_info(self):
        return super().get_info()

    def set_parameters(self, params):
        return super().set_parameters(params)

    def encode_parameters(self):
        return super().encode_parameters()

    def decode_parameters(self, encoded):
        return super().decode_parameters(encoded)

    # Optional: forward any missing attributes
    def __getattr__(self, name):
        return getattr(self.__dict__.get('_original_model', self), name)