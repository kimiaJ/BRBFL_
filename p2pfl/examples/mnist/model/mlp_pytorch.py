#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Simple MLP on PyTorch Lightning for MNIST."""

import lightning as L
import torch
from torchmetrics import Accuracy, Metric

from brbfl.attacks import get_attack, poison_training_batch
from brbfl.attacks.poisoned_model import PoisonedLightningModel
from brbfl.evaluation.metrics import apply_mnist_trigger, backdoor_evaluation_configured, triggered_asr_counts
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed

####
# Example MLP
####


class MLP(L.LightningModule):
    """Multilayer Perceptron (MLP) with configurable parameters."""

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_sizes: list[int] | None = None,
        out_channels: int = 10,
        activation: str = "relu",
        metric: type[Metric] = Accuracy,
        lr_rate: float = 0.001,
    ) -> None:
        """Initialize the MLP."""
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        self.lr_rate = lr_rate
        if out_channels == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=out_channels)

        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(self._get_activation(activation))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(self._get_activation(activation))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], out_channels))

    def _get_activation(self, activation_name: str) -> torch.nn.Module:
        if activation_name == "relu":
            return torch.nn.ReLU()
        elif activation_name == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation_name == "tanh":
            return torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        # Flatten the input
        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1)

        for layer in self.layers:
            x = layer(x)

        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Train on one local batch after any configured online data attack."""
        x = batch["image"].float()
        y = batch["label"]

        attack = get_attack(getattr(self, "node_addr", None))
        x, y = poison_training_batch((x, y), attack)

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if attack:
            loss *= 30.0
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_metric", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Perform validation step for the MLP."""
        raise NotImplementedError("Validation step not implemented")

    def test_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> torch.Tensor:
        """Evaluate loss, accuracy, and ASR only when a trigger is configured."""
        x = batch["image"].float()
        y = batch["label"]

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        attack = get_attack(getattr(self, "node_addr", None)) or getattr(self, "backdoor_evaluation_attack", None)
        self.log("test_loss", loss)
        self.log("test_metric", acc)
        # Never call predictions on unmodified images ASR. Genuine ASR is
        # measured only after triggering non-target examples.
        if backdoor_evaluation_configured(attack):
            x_trigger = apply_mnist_trigger(x, attack.trigger)
            pred_trigger = self(x_trigger).argmax(dim=1)
            result = triggered_asr_counts(pred_trigger, y, attack.target_class, attack.source_labels)
            if result["triggered_test_asr"] is not None:
                self.log("triggered_test_asr", result["triggered_test_asr"], batch_size=result["eligible_triggered_examples"])
            self.log("triggered_test_target_prediction_count", float(result["triggered_test_target_prediction_count"]), reduce_fx="sum")
            self.log("eligible_triggered_examples", float(result["eligible_triggered_examples"]), reduce_fx="sum")

        return loss


# Export P2PFL model
def model_build_fn(*args, **kwargs) -> LightningModel:
    """
    Export the model build function.

    If a `node_addr` is supplied in kwargs the wrapper `PoisonedLightningModel`
    will be used so any attack.manipulate_update() scaling is applied when the
    model serializes parameters to be sent by the node.
    """
    # Pop values that are meant for the wrapper, not for MLP constructor
    node_addr = kwargs.pop("node_addr", None)
    compression = kwargs.pop("compression", None)

    base_model = MLP(*args, **kwargs)
    # Use poisoned wrapper so manipulate_update() is applied when sending params
    return PoisonedLightningModel(base_model, compression=compression, node_addr=node_addr)
