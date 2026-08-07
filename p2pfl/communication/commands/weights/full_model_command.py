#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""FullModelCommand."""

from collections.abc import Callable

from brbfl.validation import get_validator_gate, parameter_hash
from p2pfl.communication.commands.command import Command
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class FullModelCommand(Command):
    """FullModelCommand."""

    def __init__(self, state: NodeState, stop: Callable[[], None], aggregator: Aggregator, learner: Learner) -> None:
        """Initialize FullModelCommand."""
        self.state = state
        self.stop = stop
        self.aggregator = aggregator
        self.learner = learner

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "add_model"

    def execute(
        self,
        source: str,
        round: int,
        weights: bytes | None = None,
        **kwargs,
    ) -> None:
        """Execute the command."""
        if weights is None:
            raise ValueError("Weights, contributors and weight are required")

        # Check if Learning is running
        if self.state.round is not None:
            # Check source
            if round != self.state.round:
                self.state.record_aggregate_lifecycle(
                    round,
                    "aggregate_rejected",
                    rejection_reason=f"wrong round: message={round}, current={self.state.round}",
                    source=source,
                )
                logger.debug(
                    self.state.addr,
                    f"Model reception in a late round ({round} != {self.state.round}).",
                )
                return
            try:
                logger.info(self.state.addr, "📦 Aggregated model received.")
                # Decode into a detached copy first: malformed metadata or a
                # mutated payload must never alter the installed learner.
                model = self.learner.get_model().build_copy(params=weights)
                receipt = model.additional_info.get("canonical_round_result")
                self.state.record_aggregate_lifecycle(
                    round,
                    "aggregate_received",
                    aggregate_origin=receipt.get("origin") if isinstance(receipt, dict) else None,
                    contributors=receipt.get("contributors") if isinstance(receipt, dict) else None,
                    receipt_present=isinstance(receipt, dict),
                    source=source,
                )
                gate = get_validator_gate(self.state.addr)
                if gate is not None:
                    if not isinstance(receipt, dict):
                        raise RuntimeError("aggregate lacks canonical round-result receipt")
                    contributors = sorted(receipt.get("contributors", ()))
                    input_hashes = receipt.get("aggregation_input_hashes")
                    digest = parameter_hash(model.get_parameters())
                    receipt_key = f"add_model:{round}:{receipt.get('origin')}:{receipt.get('global_model_sha256')}"
                    self.state.record_aggregate_lifecycle(
                        round,
                        "receipt_found",
                        aggregate_origin=receipt.get("origin"),
                        contributors=contributors,
                        receipt_key=receipt_key,
                        expected_aggregate_hash=receipt.get("global_model_sha256"),
                        decoded_parameter_hash=digest,
                        receipt_present=True,
                    )
                    if receipt.get("round") != int(round):
                        raise RuntimeError(f"aggregate round mismatch: message={round}, receipt={receipt.get('round')}")
                    if receipt.get("origin") not in self.state.train_set:
                        raise RuntimeError(
                            "aggregate origin is not a selected trainer: "
                            f"origin={receipt.get('origin')}, selected={sorted(self.state.train_set)}, relay={source}"
                        )
                    active = set(self.state.nei_status) | {self.state.addr}
                    if source not in active:
                        raise RuntimeError(f"aggregate relay is not an active participant: relay={source}, active={sorted(active)}")
                    if contributors != sorted(input_hashes or {}) or not set(contributors) <= set(gate.policy.contributors):
                        raise RuntimeError("aggregate contributor set/input hashes are inconsistent with contributor policy")
                    if digest != receipt.get("global_model_sha256"):
                        raise RuntimeError(
                            f"aggregate parameter hash mismatch: transmitted={receipt.get('global_model_sha256')}, decoded={digest}"
                        )
                    self.state.record_aggregate_lifecycle(
                        round,
                        "verification_passed",
                        aggregate_origin=receipt.get("origin"),
                        contributors=contributors,
                        receipt_key=receipt_key,
                        expected_aggregate_hash=digest,
                        decoded_parameter_hash=digest,
                    )
                    previous = self.state.installed_model_hashes.get(int(round))
                    if previous is not None:
                        if previous != digest:
                            raise RuntimeError(f"conflicting duplicate aggregate: round={round}")
                        self.state.record_verified_installation(int(round), digest)
                        return
                self.learner.set_model(model)
                installed = parameter_hash(self.learner.get_model().get_parameters())
                self.state.record_aggregate_lifecycle(
                    round,
                    "model_installed",
                    expected_aggregate_hash=receipt.get("global_model_sha256") if isinstance(receipt, dict) else installed,
                    post_install_parameter_hash=installed,
                )
                if gate is not None and installed != receipt["global_model_sha256"]:
                    raise RuntimeError("installed learner parameters differ from verified aggregate")
                if gate is not None:
                    gate.observe_round_result(
                        round,
                        self.learner.get_model().get_parameters(),
                        receipt["contributors"],
                        canonical_hash_source=f"verified aggregate from {source}",
                    )
                # Publish only after set_model, post-install hashing, receipt
                # verification, and evidence observation have all succeeded.
                self.state.record_verified_installation(int(round), installed)

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError:
                logger.error(self.state.addr, "❌ Error decoding parameters.")
                self.state.record_aggregate_error(round, "error decoding aggregate parameters")
                self.stop()

            except ModelNotMatchingError:
                logger.error(self.state.addr, "❌ Models not matching.")
                self.state.record_aggregate_error(round, "aggregate model does not match learner")
                self.stop()

            except Exception as e:
                logger.error(self.state.addr, f"❌ Unknown error adding full model: {e}")
                reason = f"{type(e).__name__}: {e}"
                self.state.record_aggregate_lifecycle(round, "aggregate_rejected", rejection_reason=reason, source=source)
                self.state.record_aggregate_error(round, reason)
                self.stop()
        else:
            logger.debug(self.state.addr, "❌ Tried to add a model while learning is not running")
