# attacks/poisoned_model.py
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from typing import List, Any, Optional
import numpy as np
from .registry import get_attack
import copy


class PoisonedLightningModel(LightningModel):
    """
    Fully compatible drop-in replacement for LightningModel.
    Supports all p2pfl constructor args (params, num_samples, etc.)
    """
    def __init__(
        self,
        model: Any,
        params: Optional[List[np.ndarray] | bytes] = None,
        num_samples: Optional[int] = None,
        contributors: Optional[list[str]] = None,
        additional_info: Optional[dict[str, Any]] = None,
        compression: Optional[dict[str, dict[str, Any]]] = None,
        node_addr: Optional[str] = None,  # our custom arg
    ):
        # Call the real LightningModel.__init__ with all standard args
        super().__init__(
            model=model,
            params=params,
            num_samples=num_samples,
            contributors=contributors,
            additional_info=additional_info,
            compression=compression,
        )

        # Store only the address (serialization-safe!)
        self.node_addr = node_addr

    def get_parameters(self) -> List[np.ndarray]:
        params = super().get_parameters()
        attack = get_attack(self.node_addr) if self.node_addr else None
        if attack:
            params = attack.manipulate_update(params)
        return params

    # Optional: make copying work cleanly
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "node_addr":  # don't deepcopy node_addr (it's just a string)
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, v)
        return result