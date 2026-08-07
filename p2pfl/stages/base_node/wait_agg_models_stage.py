#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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
"""Wait aggregated models stage."""

import time

from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class WaitAggregatedModelsStage(Stage):
    """Wait aggregated models stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "WaitAggregatedModelsStage"

    @staticmethod
    def execute(
        state: NodeState | None = None, communication_protocol: CommunicationProtocol | None = None, **kwargs
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None:
            raise Exception("Invalid parameters on WaitAggregatedModelsStage.")
        round_number = int(state.round)
        state.record_aggregate_lifecycle(
            round_number,
            "installation_wait_started",
            aggregate_event=state.aggregated_model_event.is_set(),
            installed_record=round_number in state.installed_model_hashes,
        )
        logger.info(state.addr, "⏳ Waiting aggregation.")
        deadline = time.monotonic() + Settings.training.AGGREGATION_TIMEOUT
        with state.aggregate_installation_condition:
            while (
                state.installed_model_hashes.get(round_number) != state.verified_model_hashes.get(round_number)
                or round_number not in state.verified_model_hashes
            ):
                error = state.aggregate_installation_errors.get(round_number)
                if error is not None:
                    raise RuntimeError(f"verified aggregate receiver failed: round={round_number}, node={state.addr}, cause={error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "aggregation timeout before verified model installation: "
                        f"round={round_number}, node={state.addr}, "
                        f"installed={state.installed_model_hashes.get(round_number)}, "
                        f"verified={state.verified_model_hashes.get(round_number)}"
                    )
                state.aggregate_installation_condition.wait(remaining)
        state.record_aggregate_lifecycle(
            round_number,
            "installation_wait_satisfied",
            expected_aggregate_hash=state.verified_model_hashes[round_number],
            installed_record=True,
        )
        logger.info(state.addr, "✅ Verified aggregate installation received.")

        # Get aggregated model
        logger.debug(
            state.addr,
            f"Broadcast aggregation done for round {state.round}",
        )
        # Share that aggregation is done
        communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

        return StageFactory.get_stage("GossipModelStage")
