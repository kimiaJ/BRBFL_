from typing import TYPE_CHECKING, Any, Dict, Optional

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

if TYPE_CHECKING:
    from p2pfl.node import Node


class BaseAttack:

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.node: Optional["Node"] = None

    def on_attach(self, node: "Node") -> None:
        """Called when attack is attached to a node."""
        self.node = node

    def poison_data(self, dataset: P2PFLDataset) -> P2PFLDataset:
        """Modify dataset before training (data poisoning)."""
        return dataset

    def manipulate_update(self, state_dict: Dict[str, Any],model=None) -> Dict[str, Any]:
        """Modify model parameters before sending (model poisoning)."""
        return state_dict

    def on_round_start(self, round_idx: int) -> None:
        """Called at the start of each round."""
        pass

    def on_round_end(self, round_idx: int) -> None:
        """Called at the end of each round."""
        pass
