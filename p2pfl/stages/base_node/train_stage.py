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
"""Train stage."""

from typing import Any

from brbfl.attacks import get_attack
from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "TrainStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        learner: Learner | None = None,
        aggregator: Aggregator | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on TrainStage.")

        try:
            attack = get_attack(state.addr)
            trace = getattr(attack, "trace", lambda *args, **fields: None)
            check_early_stop(state)

            # Set Models To Aggregate
            aggregator.set_nodes_to_aggregate(state.train_set)

            check_early_stop(state)

            # Evaluate and send metrics
            TrainStage.__evaluate(state, learner, communication_protocol)
            trace("evaluation_completed")

            check_early_stop(state)

            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            trace("local_training_started")
            training_begin = getattr(attack, "begin_local_training", None)
            if training_begin is not None:
                training_begin(state.round, learner.get_model().get_parameters())
            counter_reset = getattr(learner.get_model(), "reset_optimizer_step_count", None)
            if counter_reset is not None:
                counter_reset()
            skip_training = bool(getattr(attack, "should_skip_local_training", lambda: False)())
            if not skip_training:
                learner.fit()
            else:
                # A skipped fit must still set normal contribution metadata so
                # FedAvg treats this node exactly like every other participant.
                learner.get_model().set_contribution([state.addr], learner.get_data().get_num_samples())
            step_recorder = getattr(attack, "record_optimizer_steps", None)
            if step_recorder is not None:
                step_recorder(getattr(learner.get_model(), "optimizer_step_count", lambda: 0)())
            training_complete = getattr(attack, "complete_local_training", None)
            if training_complete is not None:
                training_complete(learner.get_model().get_parameters(), skip_training)
            logger.info(state.addr, "🎓 Training done.")
            trace("local_training_completed")
            # Lifecycle attacks are owned by the node process.  A fitted model may
            # have made a Ray round trip, so do not make the canonical audit
            # depend on a registry lookup made through that returned object.
            attack_publisher = getattr(attack, "publish_update", None)
            if attack_publisher is not None:
                submitted_parameters = attack_publisher(learner.get_model().get_parameters())
                learner.get_model().set_parameters(submitted_parameters)
            else:
                local_update_publisher = getattr(learner.get_model(), "publish_local_update", None)
                submitted_parameters = (
                    local_update_publisher() if local_update_publisher is not None else learner.get_model().get_parameters()
                )
            submission_recorder = getattr(attack, "record_submission", None)
            if submission_recorder is not None:
                submission_recorder(submitted_parameters, state.round)

            check_early_stop(state)

            # Aggregate Model
            aggregation_observer = getattr(attack, "observe_aggregation", None)
            if aggregation_observer is not None:
                aggregation_observer(learner.get_model().get_parameters())
            from brbfl.validation import get_validator_gate

            gate = get_validator_gate()
            admitted = gate is None or gate.submit_and_decide(state.round, state.addr, learner.get_model().get_parameters())
            if admitted:
                if gate is not None:
                    gate.observe_aggregation_input(state.round, state.addr, learner.get_model().get_parameters())
                models_added = aggregator.add_model(learner.get_model())
            else:
                models_added = aggregator.reject_model([state.addr])

            # send model added msg ---->> redundant (a node always owns its model)
            # TODO: print("Broadcast redundante")
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=state.round,
                )
            )
            TrainStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

            check_early_stop(state)

            # Set aggregated model
            agg_model = aggregator.wait_and_get_aggregation()
            learner.set_model(agg_model)
            global_observer = getattr(attack, "observe_global_model", None)
            if global_observer is not None:
                global_observer(learner.get_model().get_parameters())

            # Share that aggregation is done
            communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

            # Next stage
            return StageFactory.get_stage("GossipModelStage")
        except EarlyStopException:
            return None

    @staticmethod
    def __evaluate(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol) -> None:
        logger.info(state.addr, "🔬 Evaluating...")
        results = learner.evaluate()
        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )

    @staticmethod
    def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> list[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStage.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStage.__get_aggregated_models(n, state),
                )  # reemplazar por Aggregator - borrarlo de node
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            try:
                model = aggregator.get_model(TrainStage.__get_aggregated_models(node, state))
            except NoModelsToAggregateError:
                logger.debug(state.addr, f"❔ No models to aggregate for {node}.")
                return (
                    None,
                    PartialModelCommand.get_name(),
                    state.round,
                    [],
                )
            model_msg = communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            attack = get_attack(state.addr)
            recorder = getattr(attack, "record_transmission", None)
            if recorder is not None:
                recorder(node, model.get_parameters())
            return (
                model_msg,
                PartialModelCommand.get_name(),
                state.round,
                model.get_contributors(),
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True,
        )

    @staticmethod
    def __get_aggregated_models(node: str, state: NodeState) -> list[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []

    @staticmethod
    def __get_remaining_nodes(node: str, state: NodeState) -> set[str]:
        return set(state.train_set) - set(TrainStage.__get_aggregated_models(node, state))
