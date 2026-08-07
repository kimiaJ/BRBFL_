"""P2PFL model wrapper that applies the attack registered for its node."""

import copy
from typing import Any

import numpy as np

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel

from .lifecycle import poison_model_update
from .registry import get_attack


class PoisonedLightningModel(LightningModel):
    """
    Provide a fully compatible drop-in replacement for LightningModel.

    Supports all p2pfl constructor args (params, num_samples, etc.)
    """

    def __init__(
        self,
        model: Any,
        params: list[np.ndarray] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
        node_addr: str | None = None,
    ) -> None:
        """Initialize the wrapper and propagate its registry key to Lightning."""
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
        # Lightning invokes training_step on the wrapped module, so give the
        # online data-poisoning hook the same registry key.
        self.model.node_addr = node_addr

    def get_parameters(self) -> list[np.ndarray]:
        """Return the current published model without transport-time mutation."""
        return super().get_parameters()

    def publish_local_update(self) -> list[np.ndarray]:
        """Create and install the one attacked snapshot produced by local training."""
        benign = super().get_parameters()
        attack = get_attack(self.node_addr) if self.node_addr else None
        publisher = getattr(attack, "publish_update", None)
        attacked = publisher(benign) if publisher is not None else poison_model_update(benign, attack)
        detached = [np.asarray(value).copy() for value in attacked]
        super().set_parameters(detached)
        return [value.copy() for value in detached]

    def reset_optimizer_step_count(self) -> None:
        """Reset the counter that travels with actor-backed fitted models."""
        self.model._brbfl_optimizer_step_count = 0

    def optimizer_step_count(self) -> int:
        """Return the observed optimizer steps from local or actor-backed fit."""
        return int(getattr(self.model, "_brbfl_optimizer_step_count", 0))

    def build_copy(self, **kwargs):
        """
        Preserve the attack registry key when aggregation replaces the model.

        P2PFL installs the result of ``Aggregator.aggregate`` as the learner's
        model after every round.  The inherited factory did not forward our
        extension field, silently turning the malicious node into a clean node
        after round zero.
        """
        contributors = kwargs.get("contributors")
        # Received peers' models are built through this same factory and must
        # remain clean.  FedAvg's replacement contains the local contributor,
        # whereas a newly decoded remote partial model does not.
        keep_registry_key = contributors is None or self.node_addr in contributors
        node_addr = self.node_addr if keep_registry_key else None
        return self.__class__(copy.deepcopy(self.model), node_addr=node_addr, **kwargs)

    # Optional: make copying work cleanly
    def __deepcopy__(self, memo):
        """Copy the model while retaining its immutable node address."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "node_addr":  # don't deepcopy node_addr (it's just a string)
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, v)
        return result
