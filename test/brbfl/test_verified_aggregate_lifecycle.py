"""Regressions for round-scoped verified aggregate installation."""
# ruff: noqa: D103

import threading
import time
from unittest.mock import Mock

import pytest

from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.experiment import Experiment
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.base_node.wait_agg_models_stage import WaitAggregatedModelsStage
from p2pfl.stages.stage_factory import StageFactory


class _Protocol:
    def build_msg(self, *args, **kwargs):
        return args, kwargs

    def broadcast(self, message):
        self.message = message


@pytest.fixture(autouse=True)
def _quiet_logger(monkeypatch):
    monkeypatch.setattr(logger, "info", Mock())


def _state(rounds=2):
    state = NodeState("node-3")
    state.experiment = Experiment("verified-install", rounds)
    state.aggregated_model_event.clear()
    return state


def test_delayed_models_ready_does_not_promote_peer_to_current_round():
    """A round-zero acknowledgement received in round one remains round zero."""
    state = _state()
    state.nei_status["node-4"] = -1
    state.experiment.increase_round()
    ModelsReadyCommand(state).execute("node-4", 0)
    assert state.nei_status["node-4"] == 0


def test_aggregate_before_wait_is_not_cleared_and_returns_immediately(monkeypatch):
    state = _state()
    monkeypatch.setattr(StageFactory, "get_stage", Mock(return_value=None))
    state.record_verified_installation(0, "canonical")
    started = time.monotonic()
    WaitAggregatedModelsStage.execute(state=state, communication_protocol=_Protocol())
    assert time.monotonic() - started < 0.2
    assert state.aggregated_model_event.is_set()


def test_wait_before_aggregate_uses_round_scoped_predicate(monkeypatch):
    state = _state()
    monkeypatch.setattr(StageFactory, "get_stage", Mock(return_value=None))
    monkeypatch.setattr(Settings.training, "AGGREGATION_TIMEOUT", 1.0)
    installer = threading.Thread(target=lambda: (time.sleep(0.02), state.record_verified_installation(0, "canonical")))
    installer.start()
    WaitAggregatedModelsStage.execute(state=state, communication_protocol=_Protocol())
    installer.join()
    assert state.installed_model_hashes[0] == state.verified_model_hashes[0] == "canonical"


def test_previous_round_event_and_record_cannot_satisfy_next_round(monkeypatch):
    state = _state()
    state.record_verified_installation(0, "round-zero")
    state.experiment.increase_round()
    monkeypatch.setattr(Settings.training, "AGGREGATION_TIMEOUT", 0.01)
    with pytest.raises(RuntimeError, match=r"round=1.*installed=None, verified=None"):
        WaitAggregatedModelsStage.execute(state=state, communication_protocol=_Protocol())


def test_receiver_failure_is_raised_with_original_cause(monkeypatch):
    state = _state()
    monkeypatch.setattr(Settings.training, "AGGREGATION_TIMEOUT", 1.0)
    state.record_aggregate_error(0, "RuntimeError: aggregate lacks canonical round-result receipt")
    with pytest.raises(RuntimeError, match="aggregate lacks canonical round-result receipt"):
        WaitAggregatedModelsStage.execute(state=state, communication_protocol=_Protocol())


def test_installed_record_precedes_completion_signal():
    state = _state()
    state.record_verified_installation(0, "canonical")
    assert state.installed_model_hashes[0] == "canonical"
    assert state.aggregated_model_event.is_set()
    phases = [row["phase"] for row in state.aggregate_lifecycle]
    assert phases[-2:] == ["installed_hash_recorded", "installation_event_set"]
