import warnings
from unittest.mock import Mock

import pytest
from typing import Text

import rasa.shared.core.domain
from sanic import Sanic
from asyncio import AbstractEventLoop
from pathlib import Path
from rasa.core import run
from rasa.core.brokers.sql import SQLEventBroker
from rasa.core.utils import AvailableEndpoints

CREDENTIALS_FILE = "data/test_moodbot/credentials.yml"


def test_create_http_input_channels():
    channels = run.create_http_input_channels(None, CREDENTIALS_FILE)
    assert len(channels) == 7

    # ensure correct order
    assert {c.name() for c in channels} == {
        "twilio",
        "slack",
        "telegram",
        "mattermost",
        "facebook",
        "webexteams",
        "rocketchat",
    }


def test_create_single_input_channels():
    channels = run.create_http_input_channels("facebook", CREDENTIALS_FILE)
    assert len(channels) == 1
    assert channels[0].name() == "facebook"


def test_create_single_input_channels_by_class():
    channels = run.create_http_input_channels(
        "rasa.core.channels.rest.RestInput", CREDENTIALS_FILE
    )
    assert len(channels) == 1
    assert channels[0].name() == "rest"


def test_create_single_input_channels_by_class_wo_credentials():
    channels = run.create_http_input_channels(
        "rasa.core.channels.rest.RestInput", credentials_file=None
    )

    assert len(channels) == 1
    assert channels[0].name() == "rest"


async def test_load_agent_on_start_with_good_model_file(
    trained_rasa_model: Text, rasa_server: Sanic, loop: AbstractEventLoop
):
    agent = await run.load_agent_on_start(
        trained_rasa_model, AvailableEndpoints(), None, rasa_server, loop
    )

    assert agent.is_ready()
    assert isinstance(agent.domain, rasa.shared.core.domain.Domain)


async def test_load_agent_on_start_with_bad_model_file(
    tmp_path: Path, rasa_non_trained_server: Sanic, loop: AbstractEventLoop
):
    fake_model = tmp_path / "fake_model.tar.gz"
    fake_model.touch()
    fake_model_path = str(fake_model)

    with pytest.warns(UserWarning) as warnings:
        await run.load_agent_on_start(
            fake_model_path, AvailableEndpoints(), None, rasa_non_trained_server, loop
        )
        assert any("No valid model found at" in str(w.message) for w in warnings)


async def test_close_resources(loop: AbstractEventLoop):
    broker = SQLEventBroker()
    app = Mock()
    app.ctx.agent.tracker_store.event_broker = broker

    with warnings.catch_warnings() as record:
        await run.close_resources(app, loop)
        assert record is None
