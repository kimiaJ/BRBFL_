# attacks/label_flipping.py
from .base import BaseAttack
from typing import Dict
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


class LabelFlippingAttack(BaseAttack):
    def __init__(self, flip_map: Dict[int, int]):
        super().__init__(params={"flip_map": flip_map})

    def poison_data(self, dataset: P2PFLDataset) -> P2PFLDataset:
        flip_map = self.params["flip_map"]

        def flip(example):
            example["label"] = flip_map.get(example["label"], example["label"])
            return example

        print(f"[Attack] LabelFlipping: {flip_map}")

        # Access private _data and _train_split_name
        if not hasattr(dataset, "_data"):
            raise RuntimeError("P2PFLDataset missing _data")

        train_split = getattr(dataset, "_train_split_name", "train")
        if train_split not in dataset._data:
            raise KeyError(f"Train split '{train_split}' not found in _data")

        dataset._data[train_split] = dataset._data[train_split].map(flip)
        return dataset