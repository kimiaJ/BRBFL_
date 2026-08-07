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

"""Abstract aggregator."""

import threading

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent


class NoModelsToAggregateError(Exception):
    """Exception raised when there are no models to aggregate."""

    pass


class Aggregator(NodeComponent):
    """
    Class to manage the aggregation of models.

    Args:
        node_addr: Address of the node.

    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = False  # Default, subclasses should override

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        self.__train_set: list[str] = []  # TODO: Remove the trainset from the state
        self.__models: list[P2PFLModel] = []
        self.__rejected_contributors: set[str] = set()
        self.__aggregation_round: int | None = None
        self.__finalized_admission: tuple[int, tuple[str, ...], tuple[str, ...]] | None = None

        # Initialize instance's partial_aggregation based on the class's support
        self.partial_aggregation: bool = self.__class__.SUPPORTS_PARTIAL_AGGREGATION

        # If the class supports it, allow disabling it for this instance
        if self.partial_aggregation and disable_partial_aggregation:
            self.partial_aggregation = False

        # (addr) Super
        NodeComponent.__init__(self)

        # Locks
        self.__agg_lock = threading.Lock()
        self._finish_aggregation_event = threading.Event()
        self._finish_aggregation_event.set()

        # Unhandled models
        self.__unhandled_models: list[P2PFLModel] = []

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models to aggregate.

        """
        raise NotImplementedError

    def get_required_callbacks(self) -> list[str]:
        """
        Get the required callbacks for the aggregation.

        Returns:
            List of required callbacks.

        """
        return []

    def set_nodes_to_aggregate(self, nodes_to_aggregate: list[str], round_id: int | None = None) -> None:
        """
        List with the name of nodes to aggregate. Be careful, by setting new nodes, the actual aggregation will be lost.

        Args:
            nodes_to_aggregate: List of nodes to aggregate. Empty for no aggregation.
            round_id: Canonical round owning this aggregation, when admission is enabled.

        Raises:
            Exception: If the aggregation is running.

        """
        if not self._finish_aggregation_event.is_set():
            raise Exception("It is not possible to set nodes to aggregate when the aggregation is running.")

        # Start new aggregation
        canonical_round = None if round_id is None else int(round_id)
        if self.__finalized_admission is not None and canonical_round is not None:
            finalized_round = self.__finalized_admission[0]
            if canonical_round <= finalized_round:
                raise ValueError(
                    f"aggregation round must advance after finalized admission: round={canonical_round}, "
                    f"finalized_round={finalized_round}"
                )
        self.__aggregation_round = canonical_round
        self.__finalized_admission = None
        self.__train_set = list(nodes_to_aggregate)
        self._finish_aggregation_event.clear()
        for m in self.__unhandled_models:
            self.add_model(m)
            # NOTE: Don´t need to send message indicating this aggregations. Self aggregation will be sufficient to notify the network.
        self.__unhandled_models = []

    def clear(self) -> None:
        """Clear the aggregation (remove trainset and release locks)."""
        with self.__agg_lock:
            self.__train_set = []
            self.__models = []
            self.__rejected_contributors = set()
            self.__aggregation_round = None
            self.__finalized_admission = None
            self.__unhandled_models = []
            self._finish_aggregation_event.set()

    def set_admitted_contributors(self, contributors: list[str]) -> None:
        """
        Replace pre-admission candidates with the canonical admitted set.

        Models may arrive before the final set is known; admitted ones are
        retained, while rejected ones are removed atomically.
        """
        canonical = sorted(set(contributors))
        with self.__agg_lock:
            unknown = set(canonical) - set(self.__train_set)
            if unknown:
                raise ValueError(f"admitted contributors are not in the train set: {sorted(unknown)}")
            self.__train_set = canonical
            self.__models = [m for m in self.__models if set(m.get_contributors()) <= set(canonical)]
            self.__rejected_contributors.clear()
            received = set(self.get_aggregated_models())
            if received >= set(canonical):
                self._finish_aggregation_event.set()
            else:
                self._finish_aggregation_event.clear()

    def finalize_admitted_contributors(
        self,
        round_id: int,
        configured_contributors: list[str],
        admitted_contributors: list[str],
    ) -> None:
        """Apply one immutable, round-scoped admission result to readiness."""
        canonical_round = int(round_id)
        configured = tuple(sorted(set(configured_contributors)))
        admitted = tuple(sorted(set(admitted_contributors)))
        decision = (canonical_round, configured, admitted)
        with self.__agg_lock:
            if self.__aggregation_round != canonical_round:
                raise ValueError(
                    f"admission decision is for a stale or wrong round: round={canonical_round}, "
                    f"aggregation_round={self.__aggregation_round}"
                )
            if set(admitted) - set(configured):
                raise ValueError(f"admitted contributors are not configured: {sorted(set(admitted) - set(configured))}")
            if self.__finalized_admission is not None:
                if self.__finalized_admission != decision:
                    raise ValueError(
                        f"contradictory finalized admission decision: round={canonical_round}, "
                        f"previous={self.__finalized_admission[2]}, admitted={admitted}"
                    )
                return
            if set(self.__train_set) != set(configured):
                raise ValueError(
                    "configured contributors disagree with the aggregation train set: "
                    f"configured={list(configured)}, train_set={sorted(self.__train_set)}"
                )
            self.__train_set = list(admitted)
            admitted_set = set(admitted)
            self.__models = [m for m in self.__models if set(m.get_contributors()) <= admitted_set]
            self.__rejected_contributors.clear()
            received = set(self.get_aggregated_models())
            if received >= admitted_set:
                self._finish_aggregation_event.set()
            else:
                self._finish_aggregation_event.clear()
            self.__finalized_admission = decision

    def get_expected_aggregation_models(self) -> set[str]:
        """Return the canonical contributors currently required for readiness."""
        with self.__agg_lock:
            return set(self.__train_set)

    def get_aggregated_models(self) -> list[str]:
        """
        Get the list of aggregated models.

        Returns:
            Name of nodes that colaborated to get the model.

        """
        models_added = []
        for n in self.__models:
            models_added += n.get_contributors()
        return models_added

    def reject_model(self, contributors: list[str]) -> list[str]:
        """Mark a validation-rejected submission handled without aggregating it."""
        with self.__agg_lock:
            invalid = set(contributors) - set(self.__train_set)
            if invalid:
                raise ValueError(f"rejected contributors are not in the train set: {sorted(invalid)}")
            self.__rejected_contributors.update(contributors)
            handled = set(self.get_aggregated_models()) | self.__rejected_contributors
            if len(handled) >= len(self.__train_set):
                self._finish_aggregation_event.set()
            return sorted(handled)

    def add_model(self, model: P2PFLModel) -> list[str]:
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.

        Returns:
            List of contributors.

        """
        # Verify that contributors are not empty
        if model.get_contributors() == []:
            logger.debug(self.addr, "Received a model without a list of contributors.")
            self.__agg_lock.release()
            return []

        # Lock
        self.__agg_lock.acquire()

        #
        # TODO: (optimiazacion) Si llega un modelo completamente agregado, se tiene que saltar todo esto
        # TODO: A veces se agregan repetidos
        #

        # Check if aggregation is needed
        if len(self.__train_set) > len(self.get_aggregated_models()):
            # Check if all nodes are in the train_set
            if all(n in self.__train_set for n in model.get_contributors()):
                # Check if any model was added
                any_model_added = any(n in self.get_aggregated_models() for n in model.get_contributors())
                if not any_model_added:
                    # Aggregate model
                    self.__models.append(model)
                    models_added = str(len(self.get_aggregated_models()))
                    logger.info(
                        self.addr,
                        f"🧩 Model added ({models_added}/{str(len(self.__train_set))}) from {str(model.get_contributors())}",
                    )
                    # logger.debug(self.addr, f"Models added: {self.get_aggregated_models()}")

                    # Check if all models were added
                    handled = set(self.get_aggregated_models()) | self.__rejected_contributors
                    if len(handled) >= len(self.__train_set):
                        self._finish_aggregation_event.set()

                    # Unlock and Return
                    self.__agg_lock.release()
                    return self.get_aggregated_models()
                else:
                    logger.debug(
                        self.addr,
                        f"🚫 Can't add a model from a node ({model.get_contributors()}) that is already aggregated.",
                    )
            else:
                logger.debug(
                    self.addr,
                    f"🚫 Can't add a model from a node ({model.get_contributors()}) that is not in the training set.",
                )
        else:
            logger.debug(self.addr, "🚫 Received a model when is not needed. Saving a iteration to affor bandwith.")
            self.__unhandled_models.append(model)

        # Release and return
        self.__agg_lock.release()
        return []

    def wait_and_get_aggregation(self, timeout: int = Settings.training.AGGREGATION_TIMEOUT) -> P2PFLModel:
        """
        Wait for aggregation to finish.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Aggregated model.

        Raises:
            Exception: If waiting for an aggregated model and several models were received.

        """
        # Wait for aggregation to finish (then release the lock again)
        event_set = self._finish_aggregation_event.wait(timeout=timeout)
        # Check that the aggregation is finished
        missing_models = self.get_missing_models()
        # Check if aggregation has timed out or event has been set correctly
        if not event_set:
            logger.info(self.addr, f"⏳ Aggregation wait timed out. Missing models: {missing_models}")
        else:
            if len(missing_models) > 0:
                logger.info(
                    self.addr,
                    f"❌ Aggregation event set, but missing models:  {missing_models}",
                )
            else:
                logger.info(self.addr, "🧠 Aggregating models.")

        # Notify node
        return self.aggregate(self.__models)

    def get_missing_models(self) -> set:
        """
        Obtain missing models for the aggregation.

        Returns:
            A set of missing models.

        """
        agg_models = []
        for m in self.__models:
            agg_models += m.get_contributors()
        missing_models = set(self.__train_set) - set(agg_models) - self.__rejected_contributors
        return missing_models

    def __get_partial_aggregation(self, except_nodes: list[str]) -> P2PFLModel:
        """
        Obtain a partial aggregation.

        Args:
            except_nodes: List of nodes to exclude from the aggregation.

        Return:
            Aggregated model, nodes aggregated and aggregation weight.

        """
        models_to_aggregate = []
        for m in self.__models.copy():
            if all(n not in except_nodes for n in m.get_contributors()):
                models_to_aggregate.append(m)

        # A one-model "aggregation" needlessly performs multiply/divide in
        # FedAvg.  Besides wasting work, that arithmetic can change low bits of
        # float parameters.  Preserve the actual submitted model (and its
        # transport evidence) when gossiping a single contribution.
        if len(models_to_aggregate) == 1:
            return models_to_aggregate[0]
        return self.aggregate(models_to_aggregate)

    def __get_remaining_model(self, except_nodes) -> P2PFLModel:
        """
        Obtain a random model from the remaining nodes.

        Args:
            except_nodes: List of nodes to exclude from the aggregation.

        Return:
            Aggregated model, nodes aggregated and aggregation weight.

        """
        for m in self.__models.copy():
            contributors = m.get_contributors()
            if all(n not in except_nodes for n in contributors):
                return m
        raise NoModelsToAggregateError("No remaining models available for aggregation.")

    def get_model(self, except_nodes) -> P2PFLModel:
        """
        Get corresponding aggregation depending if aggregator supports partial aggregations.

        Args:
            except_nodes: List of nodes to exclude from the aggregation.

        """
        if self.partial_aggregation:
            return self.__get_partial_aggregation(except_nodes)
        else:
            return self.__get_remaining_model(except_nodes)
