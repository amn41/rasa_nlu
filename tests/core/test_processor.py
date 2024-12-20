import asyncio
import datetime
from http import HTTPStatus
import os.path
import shutil
import textwrap
from pathlib import Path

import freezegun
import pytest
from unittest.mock import MagicMock
from rasa.plugin import plugin_manager

import time
import uuid
import json
from _pytest.monkeypatch import MonkeyPatch
from _pytest.logging import LogCaptureFixture
from aioresponses import aioresponses
from typing import Optional, Text, List, Callable, Type, Any

from rasa.core.lock_store import InMemoryLockStore
from rasa.core.policies.ensemble import DefaultPolicyPredictionEnsemble
from rasa.core.tracker_store import InMemoryTrackerStore
import rasa.shared.utils.io
from rasa.core.actions.action import (
    ActionBotResponse,
    ActionListen,
    ActionExecutionRejection,
    ActionUnlikelyIntent,
)
from rasa.core.nlg import NaturalLanguageGenerator, TemplatedNaturalLanguageGenerator
from rasa.core.policies.policy import PolicyPrediction
from tests.conftest import (
    with_assistant_id,
    with_assistant_ids,
    with_model_id,
    with_model_ids,
)
import tests.utilities

from rasa.core import jobs
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    UserMessage,
    OutputChannel,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.exceptions import ActionLimitReached
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.constants import ASSISTANT_ID_KEY, LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import SessionConfig, Domain, KEY_ACTIONS
from rasa.shared.core.events import (
    ActionExecuted,
    ActiveLoop,
    BotUttered,
    ReminderCancelled,
    ReminderScheduled,
    Restarted,
    UserUttered,
    SessionStarted,
    Event,
    SlotSet,
    DefinePrevUserUtteredFeaturization,
    ActionExecutionRejected,
    LoopInterrupted,
)
from rasa.core.http_interpreter import RasaNLUHttpInterpreter
from rasa.core.processor import MessageProcessor
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    FULL_RETRIEVAL_INTENT_NAME_KEY,
    METADATA_MODEL_ID,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.endpoints import EndpointConfig
from rasa.shared.core.constants import (
    ACTION_EXTRACT_SLOTS,
    ACTION_RESTART_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
    DEFAULT_INTENTS,
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    EXTERNAL_MESSAGE_PREFIX,
    IS_EXTERNAL,
    SESSION_START_METADATA_SLOT,
)

import logging

logger = logging.getLogger(__name__)


async def test_message_processor(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    await default_processor.handle_message(
        UserMessage('/greet{"name":"Core"}', default_channel)
    )
    assert default_channel.latest_output() == {
        "recipient_id": "default",
        "text": "hey there Core!",
    }


async def test_message_id_logging(default_processor: MessageProcessor):
    message = UserMessage("If Meg was an egg would she still have a leg?")
    tracker = DialogueStateTracker("1", [])
    await default_processor._handle_message_with_tracker(message, tracker)
    logged_event = tracker.events[-1]

    assert logged_event.message_id == message.message_id
    assert logged_event.message_id is not None


async def test_parsing(default_processor: MessageProcessor):
    message = UserMessage('/greet{"name": "boy"}')
    parsed = await default_processor.parse_message(message)
    assert parsed["intent"][INTENT_NAME_KEY] == "greet"
    assert parsed["entities"][0]["entity"] == "name"


async def test_check_for_unseen_feature(default_processor: MessageProcessor):
    message = UserMessage('/greet{"name": "Joe"}')
    old_domain = default_processor.domain
    dict_for_new_domain = old_domain.as_dict()
    dict_for_new_domain["intents"] = [
        intent for intent in dict_for_new_domain["intents"] if intent != "greet"
    ]
    dict_for_new_domain["entities"] = [
        entity for entity in dict_for_new_domain["entities"] if entity != "name"
    ]
    new_domain = Domain.from_dict(dict_for_new_domain)
    default_processor.domain = new_domain

    parsed = await default_processor.parse_message(message)
    with pytest.warns(UserWarning) as record:
        default_processor._check_for_unseen_features(parsed)
    assert len(record) == 2

    assert record[0].message.args[0].startswith("Parsed an intent 'greet'")
    assert record[1].message.args[0].startswith("Parsed an entity 'name'")

    default_processor.domain = old_domain


@pytest.mark.parametrize("default_intent", DEFAULT_INTENTS)
async def test_default_intent_recognized(
    default_processor: MessageProcessor, default_intent: Text
):
    message = UserMessage(f"/{default_intent}")
    parsed = await default_processor.parse_message(message)
    with pytest.warns(None) as record:
        default_processor._check_for_unseen_features(parsed)
    assert len(record) == 0


async def test_http_parsing(trained_default_agent_model: Text, domain: Domain):
    message = UserMessage("lunch?")

    endpoint = EndpointConfig("https://interpreter.com")

    response_body = {
        "intent": {INTENT_NAME_KEY: "some_intent", "confidence": 1.0},
        "entities": [],
        "text": "lunch?",
    }

    with aioresponses() as mocked:
        mocked.post(
            "https://interpreter.com/model/parse",
            repeat=True,
            status=HTTPStatus.OK,
            body=json.dumps(response_body),
        )

        inter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        processor = MessageProcessor(
            trained_default_agent_model,
            InMemoryTrackerStore(domain),
            InMemoryLockStore(),
            NaturalLanguageGenerator(),
            http_interpreter=inter,
        )
        data = await processor.parse_message(message)

        r = tests.utilities.latest_request(
            mocked, "POST", "https://interpreter.com/model/parse"
        )

        assert r
        assert data == response_body


async def test_http_parsing_default_response(
    trained_default_agent_model: Text, domain: Domain
):
    message = UserMessage("lunch?")

    endpoint = EndpointConfig("https://interpreter.com")

    with aioresponses() as mocked:
        mocked.post(
            "https://interpreter.com/model/parse",
            repeat=True,
            status=HTTPStatus.OK,
            body=None,
        )

        inter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        processor = MessageProcessor(
            trained_default_agent_model,
            InMemoryTrackerStore(domain),
            InMemoryLockStore(),
            NaturalLanguageGenerator(),
            http_interpreter=inter,
        )
        data = await processor.parse_message(message)

        r = tests.utilities.latest_request(
            mocked, "POST", "https://interpreter.com/model/parse"
        )

        assert r
        assert data == {
            "intent": {INTENT_NAME_KEY: "", "confidence": 0.0},
            "entities": [],
            "text": "",
        }


async def test_reminder_scheduled(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex

    reminder = ReminderScheduled("remind", datetime.datetime.now())
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(UserUttered("test"))
    tracker.update(ActionExecuted("action_schedule_reminder"))
    tracker.update(reminder)

    await default_processor.tracker_store.save(tracker)

    await default_processor.handle_reminder(reminder, sender_id, default_channel)

    # retrieve the updated tracker
    t = await default_processor.tracker_store.retrieve(sender_id)

    assert t.events[1] == UserUttered("test")
    assert t.events[2] == ActionExecuted("action_schedule_reminder")
    assert isinstance(t.events[3], ReminderScheduled)
    assert t.events[4] == UserUttered(
        f"{EXTERNAL_MESSAGE_PREFIX}remind",
        intent={INTENT_NAME_KEY: "remind", IS_EXTERNAL: True},
    )


async def test_reminder_lock(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    caplog: LogCaptureFixture,
):
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        sender_id = uuid.uuid4().hex

        reminder = ReminderScheduled("remind", datetime.datetime.now())
        tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

        tracker.update(UserUttered("test"))
        tracker.update(ActionExecuted("action_schedule_reminder"))
        tracker.update(reminder)

        await default_processor.tracker_store.save(tracker)

        await default_processor.handle_reminder(reminder, sender_id, default_channel)

        assert f"Deleted lock for conversation '{sender_id}'." in caplog.text


async def test_trigger_external_latest_input_channel(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)
    input_channel = "test_input_channel_external"

    tracker.update(UserUttered("test1"))
    tracker.update(UserUttered("test2", input_channel=input_channel))

    await default_processor.trigger_external_user_uttered(
        "test3", None, tracker, default_channel
    )

    tracker = await default_processor.tracker_store.retrieve(sender_id)

    assert tracker.get_latest_input_channel() == input_channel


async def test_reminder_aborted(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex

    reminder = ReminderScheduled(
        "utter_greet", datetime.datetime.now(), kill_on_user_message=True
    )
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(reminder)
    tracker.update(UserUttered("test"))  # cancels the reminder

    await default_processor.tracker_store.save(tracker)
    await default_processor.handle_reminder(reminder, sender_id, default_channel)

    # retrieve the updated tracker
    t = await default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 3  # nothing should have been executed


async def wait_until_all_jobs_were_executed(
    timeout_after_seconds: Optional[float] = None,
) -> None:
    total_seconds = 0.0
    while len((await jobs.scheduler()).get_jobs()) > 0 and (
        not timeout_after_seconds or total_seconds < timeout_after_seconds
    ):
        await asyncio.sleep(0.1)
        total_seconds += 0.1

    if total_seconds >= timeout_after_seconds:
        jobs.kill_scheduler()
        raise TimeoutError


async def test_reminder_cancelled_multi_user(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
    trackers = []
    for sender_id in sender_ids:
        tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

        tracker.update(UserUttered("test"))
        tracker.update(ActionExecuted("action_reminder_reminder"))
        tracker.update(
            ReminderScheduled(
                "greet", datetime.datetime.now(), kill_on_user_message=True
            )
        )
        trackers.append(tracker)

    # cancel all reminders (one) for the first user
    trackers[0].update(ReminderCancelled())

    for tracker in trackers:
        await default_processor.tracker_store.save(tracker)
        await default_processor._schedule_reminders(
            tracker.events, tracker, default_channel
        )
    # check that the jobs were added
    assert len((await jobs.scheduler()).get_jobs()) == 2

    for tracker in trackers:
        await default_processor._cancel_reminders(tracker.events, tracker)
    # check that only one job was removed
    assert len((await jobs.scheduler()).get_jobs()) == 1

    # execute the jobs
    await wait_until_all_jobs_were_executed(timeout_after_seconds=5.0)

    tracker_0 = await default_processor.tracker_store.retrieve(sender_ids[0])
    # there should be no utter_greet action
    assert (
        UserUttered(
            f"{EXTERNAL_MESSAGE_PREFIX}greet",
            intent={INTENT_NAME_KEY: "greet", IS_EXTERNAL: True},
        )
        not in tracker_0.events
    )

    tracker_1 = await default_processor.tracker_store.retrieve(sender_ids[1])
    # there should be utter_greet action
    assert (
        UserUttered(
            f"{EXTERNAL_MESSAGE_PREFIX}greet",
            intent={INTENT_NAME_KEY: "greet", IS_EXTERNAL: True},
        )
        in tracker_1.events
    )


async def test_reminder_cancelled_cancels_job_with_name(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = "][]][xy,,=+2f'[:/;>]  <0d]A[e_,02"

    reminder = ReminderScheduled(
        intent="greet", trigger_date_time=datetime.datetime.now()
    )
    job_name = reminder.scheduled_job_name(sender_id)
    reminder_cancelled = ReminderCancelled()

    assert reminder_cancelled.cancels_job_with_name(job_name, sender_id)
    assert not reminder_cancelled.cancels_job_with_name(job_name.upper(), sender_id)


async def test_reminder_cancelled_cancels_job_with_name_special_name(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = "][]][xy,,=+2f'[:/;  >]<0d]A[e_,02"
    name = "wkjbgr,34(,*&%^^&*(OP#LKMN V#NF# # #R"

    reminder = ReminderScheduled(
        intent="greet", trigger_date_time=datetime.datetime.now(), name=name
    )
    job_name = reminder.scheduled_job_name(sender_id)
    reminder_cancelled = ReminderCancelled(name)

    assert reminder_cancelled.cancels_job_with_name(job_name, sender_id)
    assert not reminder_cancelled.cancels_job_with_name(job_name.upper(), sender_id)


async def cancel_reminder_and_check(
    tracker: DialogueStateTracker,
    default_processor: MessageProcessor,
    reminder_canceled_event: ReminderCancelled,
    num_jobs_before: int,
    num_jobs_after: int,
) -> None:
    # cancel the sixth reminder
    tracker.update(reminder_canceled_event)

    # check that the jobs were added
    assert len((await jobs.scheduler()).get_jobs()) == num_jobs_before

    await default_processor._cancel_reminders(tracker.events, tracker)

    # check that only one job was removed
    assert len((await jobs.scheduler()).get_jobs()) == num_jobs_after


async def test_reminder_cancelled_by_name(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel the sixth reminder
    await cancel_reminder_and_check(
        tracker, default_processor, ReminderCancelled("special"), 6, 5
    )


async def test_reminder_cancelled_by_entities(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel the fourth reminder
    await cancel_reminder_and_check(
        tracker,
        default_processor,
        ReminderCancelled(entities=[{"entity": "name", "value": "Bruce Wayne"}]),
        6,
        5,
    )


async def test_reminder_cancelled_by_intent(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel the third, fifth, and sixth reminder
    await cancel_reminder_and_check(
        tracker, default_processor, ReminderCancelled(intent="default"), 6, 3
    )


async def test_reminder_cancelled_all(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel all reminders
    await cancel_reminder_and_check(
        tracker, default_processor, ReminderCancelled(), 6, 0
    )


async def test_reminder_restart(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex

    reminder = ReminderScheduled(
        "utter_greet", datetime.datetime.now(), kill_on_user_message=False
    )
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(reminder)
    tracker.update(Restarted())  # cancels the reminder
    tracker.update(UserUttered("test"))

    await default_processor.tracker_store.save(tracker)
    await default_processor.handle_reminder(reminder, sender_id, default_channel)

    # retrieve the updated tracker
    t = await default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 4  # nothing should have been executed


@pytest.mark.parametrize(
    "event_to_apply,session_expiration_time_in_minutes,has_expired",
    [
        # last user event is way in the past
        (UserUttered(timestamp=1), 60, True),
        # user event are very recent
        (UserUttered("hello", timestamp=time.time()), 120, False),
        # there is user event
        (ActionExecuted(ACTION_LISTEN_NAME, timestamp=time.time()), 60, False),
        # Old event, but sessions are disabled
        (UserUttered("hello", timestamp=1), 0, False),
        # there is no event
        (None, 1, False),
    ],
)
async def test_has_session_expired(
    event_to_apply: Optional[Event],
    session_expiration_time_in_minutes: float,
    has_expired: bool,
    default_processor: MessageProcessor,
):
    sender_id = uuid.uuid4().hex

    default_processor.domain.session_config = SessionConfig(
        session_expiration_time_in_minutes, True
    )
    # create new tracker without events
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)
    tracker.events.clear()

    # apply desired event
    if event_to_apply:
        tracker.update(event_to_apply)

    # noinspection PyProtectedMember
    assert default_processor._has_session_expired(tracker) == has_expired


# noinspection PyProtectedMember


async def test_update_tracker_session(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

    # patch `_has_session_expired()` so the `_update_tracker_session()` call actually
    # does something
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    await default_processor._update_tracker_session(tracker, default_channel)

    # the save is not called in _update_tracker_session()
    await default_processor.save_tracker(tracker)

    # inspect tracker and make sure all events are present
    tracker = await default_processor.tracker_store.retrieve_full_tracker(sender_id)

    assert list(tracker.events) == [
        ActionExecuted(ACTION_LISTEN_NAME),
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]


async def test_update_tracker_session_with_metadata(
    default_processor: MessageProcessor, monkeypatch: MonkeyPatch
):
    model_id = default_processor.model_metadata.model_id
    assistant_id = default_processor.model_metadata.assistant_id
    sender_id = uuid.uuid4().hex
    message_metadata = {"metadataTestKey": "metadataTestValue"}
    message = UserMessage(
        text="hi",
        output_channel=CollectingOutputChannel(),
        sender_id=sender_id,
        metadata=message_metadata,
    )
    await default_processor.handle_message(message)

    tracker = await default_processor.tracker_store.retrieve_full_tracker(sender_id)
    events = list(tracker.events)

    with_model_ids_expected = with_model_ids(
        [
            SlotSet(SESSION_START_METADATA_SLOT, message_metadata),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            SlotSet(SESSION_START_METADATA_SLOT, message_metadata),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id,
    )
    final_expected = with_assistant_ids(with_model_ids_expected, assistant_id)

    assert events[0:5] == final_expected[0:5]
    assert tracker.slots[SESSION_START_METADATA_SLOT].value == message_metadata
    assert events[2].metadata == {
        ASSISTANT_ID_KEY: assistant_id,
        METADATA_MODEL_ID: model_id,
    }

    assert isinstance(events[5], UserUttered)


@freezegun.freeze_time("2020-02-01")
async def test_custom_action_session_start_with_metadata(
    default_processor: MessageProcessor,
):
    domain = Domain.from_dict({KEY_ACTIONS: [ACTION_SESSION_START_NAME]})
    default_processor.domain = domain
    model_id = default_processor.model_metadata.model_id
    action_server_url = "http://some-url"
    default_processor.action_endpoint = EndpointConfig(action_server_url)

    sender_id = uuid.uuid4().hex
    metadata = {"metadataTestKey": "metadataTestValue"}
    message = UserMessage(
        text="hi",
        output_channel=CollectingOutputChannel(),
        sender_id=sender_id,
        metadata=metadata,
    )

    with aioresponses() as mocked:
        mocked.post(action_server_url, payload={"events": []})
        await default_processor.handle_message(message)

    last_request = tests.utilities.latest_request(mocked, "post", action_server_url)
    tracker_for_custom_action = tests.utilities.json_of_latest_request(last_request)[
        "tracker"
    ]

    assert tracker_for_custom_action["events"] == [
        {
            "event": "slot",
            "timestamp": 1580515200.0,
            "name": SESSION_START_METADATA_SLOT,
            "value": metadata,
            "metadata": {"assistant_id": "placeholder_default", "model_id": model_id},
        }
    ]


# noinspection PyProtectedMember


async def test_update_tracker_session_with_slots(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

    # apply a user uttered and five slots
    user_event = UserUttered("some utterance")
    tracker.update(user_event)

    slot_set_events = [SlotSet(f"slot key {i}", f"test value {i}") for i in range(5)]

    for event in slot_set_events:
        tracker.update(event)

    # patch `_has_session_expired()` so the `_update_tracker_session()` call actually
    # does something
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    await default_processor._update_tracker_session(tracker, default_channel)

    # the save is not called in _update_tracker_session()
    await default_processor.save_tracker(tracker)

    # inspect tracker and make sure all events are present
    tracker = await default_processor.tracker_store.retrieve_full_tracker(sender_id)
    events = list(tracker.events)

    # the first three events should be up to the user utterance
    assert events[:2] == [ActionExecuted(ACTION_LISTEN_NAME), user_event]

    # next come the five slots
    assert events[2:7] == slot_set_events

    # the next two events are the session start sequence
    assert events[7:9] == [ActionExecuted(ACTION_SESSION_START_NAME), SessionStarted()]
    assert events[9:14] == slot_set_events

    # finally an action listen, this should also be the last event
    assert events[14] == events[-1] == ActionExecuted(ACTION_LISTEN_NAME)


async def test_fetch_tracker_and_update_session(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    model_id = default_processor.model_metadata.model_id
    assistant_id = default_processor.model_metadata.assistant_id
    sender_id = uuid.uuid4().hex
    tracker = await default_processor.fetch_tracker_and_update_session(
        sender_id, default_channel
    )

    # ensure session start sequence is present
    assert list(tracker.events) == with_assistant_ids(
        with_model_ids(
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
            model_id,
        ),
        assistant_id,
    )


@pytest.mark.parametrize(
    "initial_events,expected_event_types",
    [
        # tracker is initially not empty - when it is fetched, it will just contain
        # these four events
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered("/greet", {INTENT_NAME_KEY: "greet", "confidence": 1.0}),
            ],
            [ActionExecuted, SessionStarted, ActionExecuted, UserUttered],
        ),
        # tracker is initially empty, and contains the session start sequence when
        # fetched
        ([], [ActionExecuted, SessionStarted, ActionExecuted]),
    ],
)
async def test_fetch_tracker_with_initial_session(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    initial_events: List[Event],
    expected_event_types: List[Type[Event]],
):
    conversation_id = uuid.uuid4().hex

    tracker = DialogueStateTracker.from_events(conversation_id, initial_events)

    await default_processor.tracker_store.save(tracker)

    tracker = await default_processor.fetch_tracker_with_initial_session(
        conversation_id, default_channel
    )

    # the events in the fetched tracker are as expected
    assert len(tracker.events) == len(expected_event_types)

    assert all(
        isinstance(tracker_event, expected_event_type)
        for tracker_event, expected_event_type in zip(
            tracker.events, expected_event_types
        )
    )


async def test_fetch_tracker_with_initial_session_does_not_update_session(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    conversation_id = uuid.uuid4().hex

    # the domain has a session expiration time of one second
    monkeypatch.setattr(
        default_processor.tracker_store.domain,
        "session_config",
        SessionConfig(carry_over_slots=True, session_expiration_time=1 / 60),
    )

    now = time.time()

    # the tracker initially contains events
    initial_events = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=now - 10),
        SessionStarted(timestamp=now - 9),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=now - 8),
        UserUttered(
            "/greet", {INTENT_NAME_KEY: "greet", "confidence": 1.0}, timestamp=now - 7
        ),
    ]

    tracker = DialogueStateTracker.from_events(conversation_id, initial_events)

    await default_processor.tracker_store.save(tracker)

    tracker = await default_processor.fetch_tracker_with_initial_session(
        conversation_id, default_channel
    )

    # the conversation session has expired, but calling
    # `fetch_tracker_with_initial_session()` did not update it
    assert default_processor._has_session_expired(tracker)
    assert [event.as_dict() for event in tracker.events] == [
        event.as_dict() for event in initial_events
    ]


async def test_handle_message_with_session_start(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex
    model_id = default_processor.model_metadata.model_id
    assistant_id = default_processor.model_metadata.assistant_id

    entity = "name"
    slot_1 = {entity: "Core"}
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_1)}", default_channel, sender_id)
    )

    assert default_channel.latest_output() == {
        "recipient_id": sender_id,
        "text": "hey there Core!",
    }

    # patch processor so a session start is triggered
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    slot_2 = {entity: "post-session start hello"}
    # handle a new message
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_2)}", default_channel, sender_id)
    )

    tracker = await default_processor.tracker_store.get_or_create_full_tracker(
        sender_id
    )

    # make sure the sequence of events is as expected
    with_model_ids_expected = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                f"/greet{json.dumps(slot_1)}",
                {INTENT_NAME_KEY: "greet", "confidence": 1.0},
                [
                    {
                        "entity": entity,
                        "start": 6,
                        "end": 22,
                        "value": "Core",
                        "extractor": "RegexMessageHandler",
                    }
                ],
            ),
            SlotSet(entity, slot_1[entity]),
            DefinePrevUserUtteredFeaturization(False),
            ActionExecuted("utter_greet"),
            BotUttered("hey there Core!", metadata={"utter_action": "utter_greet"}),
            ActionExecuted(ACTION_LISTEN_NAME),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            # the initial SlotSet is reapplied after the SessionStarted sequence
            SlotSet(entity, slot_1[entity]),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                f"/greet{json.dumps(slot_2)}",
                {INTENT_NAME_KEY: "greet", "confidence": 1.0},
                [
                    {
                        "entity": entity,
                        "start": 6,
                        "end": 42,
                        "value": "post-session start hello",
                        "extractor": "RegexMessageHandler",
                    }
                ],
            ),
            SlotSet(entity, slot_2[entity]),
            DefinePrevUserUtteredFeaturization(False),
            ActionExecuted("utter_greet"),
            BotUttered(
                "hey there post-session start hello!",
                metadata={"utter_action": "utter_greet"},
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id,
    )
    expected = with_assistant_ids(with_model_ids_expected, assistant_id=assistant_id)
    assert list(tracker.events) == expected


# noinspection PyProtectedMember
@pytest.mark.parametrize(
    "action_name, should_predict_another_action",
    [
        (ACTION_LISTEN_NAME, False),
        (ACTION_SESSION_START_NAME, False),
        ("utter_greet", True),
    ],
)
async def test_should_predict_another_action(
    default_processor: MessageProcessor,
    action_name: Text,
    should_predict_another_action: bool,
):
    assert (
        default_processor.should_predict_another_action(action_name)
        == should_predict_another_action
    )


async def test_action_unlikely_intent_metadata(default_processor: MessageProcessor):
    tracker = DialogueStateTracker.from_events(
        "some-sender", evts=[ActionExecuted(ACTION_LISTEN_NAME)]
    )
    domain = Domain.empty()
    metadata = {"key1": 1, "key2": "2"}

    await default_processor._run_action(
        ActionUnlikelyIntent(),
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        PolicyPrediction([], "some policy", action_metadata=metadata),
    )

    applied_events = tracker.applied_events()
    assert applied_events == [
        ActionExecuted(ACTION_LISTEN_NAME),
        ActionExecuted(ACTION_UNLIKELY_INTENT_NAME, metadata=metadata),
    ]
    assert applied_events[1].metadata == metadata


async def test_restart_triggers_session_start(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    sender_id = uuid.uuid4().hex
    model_id = default_processor.model_metadata.model_id
    assistant_id = default_processor.model_metadata.assistant_id

    entity = "name"
    slot_1 = {entity: "name1"}
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_1)}", default_channel, sender_id)
    )

    assert default_channel.latest_output() == {
        "recipient_id": sender_id,
        "text": "hey there name1!",
    }

    # This restarts the chat
    await default_processor.handle_message(
        UserMessage("/restart", default_channel, sender_id)
    )

    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)

    with_model_ids_expected = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                f"/greet{json.dumps(slot_1)}",
                {INTENT_NAME_KEY: "greet", "confidence": 1.0},
                [
                    {
                        "entity": entity,
                        "start": 6,
                        "end": 23,
                        "value": "name1",
                        "extractor": "RegexMessageHandler",
                    }
                ],
            ),
            SlotSet(entity, slot_1[entity]),
            DefinePrevUserUtteredFeaturization(use_text_for_featurization=False),
            ActionExecuted("utter_greet"),
            BotUttered("hey there name1!", metadata={"utter_action": "utter_greet"}),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("/restart", {INTENT_NAME_KEY: "restart", "confidence": 1.0}),
            DefinePrevUserUtteredFeaturization(use_text_for_featurization=False),
            ActionExecuted(ACTION_RESTART_NAME),
            Restarted(),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            # No previous slot is set due to restart.
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id,
    )
    expected = with_assistant_ids(with_model_ids_expected, assistant_id)
    for actual, expected in zip(tracker.events, expected):
        assert actual == expected


async def test_handle_message_if_action_manually_rejects(
    default_processor: MessageProcessor, monkeypatch: MonkeyPatch
):
    conversation_id = "test"
    message = UserMessage("/greet", sender_id=conversation_id)

    rejection_events = [
        SlotSet("my_slot", "test"),
        ActionExecutionRejected("utter_greet"),
        SlotSet("some slot", "some value"),
    ]

    async def mocked_run(self, *args: Any, **kwargs: Any) -> List[Event]:
        return rejection_events

    monkeypatch.setattr(ActionBotResponse, ActionBotResponse.run.__name__, mocked_run)
    await default_processor.handle_message(message)

    tracker = await default_processor.tracker_store.retrieve(conversation_id)

    logged_events = list(tracker.events)

    assert ActionExecuted("utter_greet") not in logged_events
    assert all(event in logged_events for event in rejection_events)


async def test_policy_events_are_applied_to_tracker(
    default_processor: MessageProcessor, monkeypatch: MonkeyPatch
):
    model_id = default_processor.model_metadata.model_id
    assistant_id = default_processor.model_metadata.assistant_id
    expected_action = ACTION_LISTEN_NAME
    policy_events = [LoopInterrupted(True)]
    conversation_id = "test_policy_events_are_applied_to_tracker"
    user_message = "/greet"

    with_model_ids_expected_events = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(user_message, intent={"name": "greet"}),
            *policy_events,
        ],
        model_id,
    )
    expected_events = with_assistant_ids(with_model_ids_expected_events, assistant_id)

    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
        **kwargs: Any,
    ) -> PolicyPrediction:
        prediction = PolicyPrediction.for_action_name(
            default_processor.domain, expected_action, "some policy"
        )
        prediction.events = policy_events

        return prediction

    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble, "combine_predictions", combine_predictions
    )

    action_received_events = False

    async def mocked_run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        # The action already has access to the policy events
        nonlocal action_received_events
        action_received_events = list(tracker.events) == expected_events
        return []

    monkeypatch.setattr(ActionListen, ActionListen.run.__name__, mocked_run)

    await default_processor.handle_message(
        UserMessage(user_message, sender_id=conversation_id)
    )

    assert action_received_events

    tracker = await default_processor.get_tracker(conversation_id)
    # The action was logged on the tracker as well
    expected_events.append(
        with_assistant_id(
            with_model_id(ActionExecuted(ACTION_LISTEN_NAME), model_id), assistant_id
        )
    )

    for event, expected in zip(tracker.events, expected_events):
        assert event == expected


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "reject_fn",
    [
        lambda: [ActionExecutionRejected(ACTION_LISTEN_NAME)],
        lambda: (_ for _ in ()).throw(ActionExecutionRejection(ACTION_LISTEN_NAME)),
    ],
)
async def test_policy_events_not_applied_if_rejected(
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
    reject_fn: Callable[[], List[Event]],
):
    model_id = default_processor.model_metadata.model_id
    assistant_id = default_processor.model_metadata.assistant_id
    expected_action = ACTION_LISTEN_NAME
    expected_events = [LoopInterrupted(True)]
    conversation_id = "test_policy_events_are_applied_to_tracker"
    user_message = "/greet"

    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
        **kwargs: Any,
    ) -> PolicyPrediction:
        prediction = PolicyPrediction.for_action_name(
            default_processor.domain, expected_action, "some policy"
        )
        prediction.events = expected_events

        return prediction

    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble, "combine_predictions", combine_predictions
    )

    async def mocked_run(*args: Any, **kwargs: Any) -> List[Event]:
        return reject_fn()

    monkeypatch.setattr(ActionListen, ActionListen.run.__name__, mocked_run)

    await default_processor.handle_message(
        UserMessage(user_message, sender_id=conversation_id)
    )

    tracker = await default_processor.get_tracker(conversation_id)
    events = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(user_message, intent={"name": "greet"}),
            ActionExecutionRejected(ACTION_LISTEN_NAME),
        ],
        model_id,
    )
    expected_events = with_assistant_ids(events, assistant_id)
    for event, expected in zip(tracker.events, expected_events):
        assert event == expected


async def test_logging_of_end_to_end_action(
    default_processor: MessageProcessor, monkeypatch: MonkeyPatch
):
    model_id = default_processor.model_metadata.model_id
    assistant_id = default_processor.model_metadata.assistant_id
    end_to_end_action = "hi, how are you?"
    new_domain = Domain(
        intents=["greet"],
        entities=[],
        slots=[],
        responses={},
        action_names=[],
        forms={},
        action_texts=[end_to_end_action],
        data={},
    )

    default_processor.domain = new_domain

    conversation_id = "test_logging_of_end_to_end_action"
    user_message = "/greet"

    number_of_calls = 0

    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
        **kwargs: Any,
    ) -> PolicyPrediction:
        nonlocal number_of_calls
        if number_of_calls == 0:
            prediction = PolicyPrediction.for_action_name(
                new_domain, end_to_end_action, "some policy"
            )
            prediction.is_end_to_end_prediction = True
            number_of_calls += 1
            return prediction
        else:
            return PolicyPrediction.for_action_name(new_domain, ACTION_LISTEN_NAME)

    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble, "combine_predictions", combine_predictions
    )

    await default_processor.handle_message(
        UserMessage(user_message, sender_id=conversation_id)
    )

    tracker = await default_processor.tracker_store.retrieve(conversation_id)
    events = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(user_message, intent={"name": "greet"}),
            ActionExecuted(action_text=end_to_end_action),
            BotUttered("hi, how are you?", {}, {}, 123),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id=model_id,
    )
    expected_events = with_assistant_ids(events, assistant_id)
    for event, expected in zip(tracker.events, expected_events):
        assert event == expected


async def test_predict_next_action_with_hidden_rules(
    trained_async: Callable, tmp_path: Path
):
    rule_intent = "rule_intent"
    rule_action = "rule_action"
    story_intent = "story_intent"
    story_action = "story_action"
    rule_slot = "rule_slot"
    story_slot = "story_slot"
    domain_content = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - {rule_intent}
        - {story_intent}
        actions:
        - {rule_action}
        - {story_action}
        slots:
          {rule_slot}:
            type: text
            mappings:
            - type: from_text
          {story_slot}:
            type: text
            mappings:
            - type: from_text
        """
    )
    domain = Domain.from_yaml(domain_content)
    domain_path = tmp_path / "domain.yml"
    rasa.shared.utils.io.write_text_file(domain_content, domain_path)

    training_data = textwrap.dedent(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

    rules:
    - rule: rule
      steps:
      - intent: {rule_intent}
      - action: {rule_action}
      - slot_was_set:
          - {rule_slot}: {rule_slot}

    stories:
    - story: story
      steps:
      - intent: {story_intent}
      - action: {story_action}
      - slot_was_set:
          - {story_slot}: {story_slot}
    """
    )
    training_data_path = tmp_path / "data.yml"
    rasa.shared.utils.io.write_text_file(training_data, training_data_path)

    config = textwrap.dedent(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    assistant_id: placeholder_default
    policies:
    - name: RulePolicy
    - name: MemoizationPolicy

    """
    )
    config_path = tmp_path / "config.yml"
    rasa.shared.utils.io.write_text_file(config, config_path)
    model_path = await trained_async(
        str(domain_path), str(config_path), [str(training_data_path)]
    )
    agent = await load_agent(model_path=model_path)
    processor = agent.processor

    tracker = DialogueStateTracker.from_events(
        "casd",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": rule_intent}),
        ],
        slots=domain.slots,
    )
    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert action._name == rule_action
    assert prediction.hide_rule_turn

    processor._log_action_on_tracker(
        tracker, action, [SlotSet(rule_slot, rule_slot)], prediction
    )

    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert isinstance(action, ActionListen)
    assert prediction.hide_rule_turn

    processor._log_action_on_tracker(tracker, action, None, prediction)

    tracker.events.append(UserUttered(intent={"name": story_intent}))

    # rules are hidden correctly if memo policy predicts next actions correctly
    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert action._name == story_action
    assert not prediction.hide_rule_turn

    processor._log_action_on_tracker(
        tracker, action, [SlotSet(story_slot, story_slot)], prediction
    )

    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert isinstance(action, ActionListen)
    assert not prediction.hide_rule_turn


def test_predict_next_action_raises_limit_reached_exception(
    default_processor: MessageProcessor,
):
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("Hi!"),
            ActionExecuted("test_action"),
        ],
    )
    tracker.set_latest_action({"action_name": "test_action"})

    default_processor.max_number_of_predictions = 1
    with pytest.raises(ActionLimitReached):
        default_processor.predict_next_with_tracker_if_should(tracker)


async def test_processor_logs_text_tokens_in_tracker(
    default_agent: Agent, whitespace_tokenizer: WhitespaceTokenizer
):
    text = "Hello there"
    tokens = whitespace_tokenizer.tokenize(Message(data={"text": text}), "text")
    indices = [(t.start, t.end) for t in tokens]

    message = UserMessage(text)
    processor = default_agent.processor
    tracker = await processor.log_message(message)
    event = tracker.get_last_event_for(event_type=UserUttered)
    event_tokens = event.as_dict().get("parse_data").get("text_tokens")

    assert event_tokens == indices


async def test_processor_valid_slot_setting(default_agent: Agent):
    processor = default_agent.processor
    message = UserMessage(
        "Hiya Peter",
        CollectingOutputChannel(),
        "test",
        parse_data={
            "intent": {"name": "greet"},
            "entities": [{"entity": "name", "value": "Peter"}],
        },
    )
    await processor.handle_message(message)
    tracker = await processor.get_tracker("test")
    assert SlotSet("name", "Peter") in tracker.events


async def test_parse_message_nlu_only(trained_moodbot_nlu_path: Text):
    processor = Agent.load(model_path=trained_moodbot_nlu_path).processor
    message = UserMessage("/greet")
    result = await processor.parse_message(message)
    assert result == {
        "text": "/greet",
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [],
    }

    message = UserMessage("Hello")
    result = await processor.parse_message(message)
    assert result["intent"]["name"]


async def test_parse_message_core_only(trained_core_model: Text):
    processor = Agent.load(model_path=trained_core_model).processor
    message = UserMessage("/greet")
    result = await processor.parse_message(message)
    assert result == {
        "text": "/greet",
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [],
    }

    message = UserMessage("Hello")
    result = await processor.parse_message(message)
    assert not result["intent"]["name"]


async def test_parse_message_full_model(trained_moodbot_path: Text):
    processor = Agent.load(model_path=trained_moodbot_path).processor
    message = UserMessage("/greet")
    result = await processor.parse_message(message)
    assert result == {
        "text": "/greet",
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [],
    }

    message = UserMessage("Hello")
    result = await processor.parse_message(message)
    assert result["intent"]["name"]


def test_predict_next_with_tracker_nlu_only(trained_nlu_model: Text):
    processor = Agent.load(model_path=trained_nlu_model).processor
    tracker = DialogueStateTracker("some_id", [])
    tracker.followup_action = None
    result = processor.predict_next_with_tracker(tracker)
    assert result is None


def test_predict_next_with_tracker_core_only(trained_core_model: Text):
    processor = Agent.load(model_path=trained_core_model).processor
    tracker = DialogueStateTracker("some_id", [])
    tracker.followup_action = None
    result = processor.predict_next_with_tracker(tracker)
    assert result["policy"] == "MemoizationPolicy"


def test_predict_next_with_tracker_full_model(trained_rasa_model: Text):
    processor = Agent.load(model_path=trained_rasa_model).processor
    tracker = DialogueStateTracker("some_id", [])
    tracker.followup_action = None
    result = processor.predict_next_with_tracker(tracker)
    assert result["policy"] == "MemoizationPolicy"


async def test_get_tracker_adds_model_id(default_processor: MessageProcessor):
    model_id = default_processor.model_metadata.model_id
    tracker = await default_processor.get_tracker("bloop")
    assert tracker.model_id == model_id


# FIXME: these tests take too long to run in the CI, disabling them for now
@pytest.mark.skip_on_ci
async def _test_processor_e2e_slot_set(e2e_bot_agent: Agent, caplog: LogCaptureFixture):
    processor = e2e_bot_agent.processor
    message = UserMessage("I am feeling sad.", CollectingOutputChannel(), "test")
    with caplog.at_level(logging.DEBUG):
        await processor.handle_message(message)

    tracker = await processor.get_tracker("test")
    assert SlotSet("mood", "sad") in tracker.events
    assert any(
        "An end-to-end prediction was made which has triggered the 2nd execution of "
        "the default action 'action_extract_slots'." in message
        for message in caplog.messages
    )


async def test_model_name_is_available(trained_rasa_model: Text):
    processor = Agent.load(model_path=trained_rasa_model).processor
    assert len(processor.model_filename) > 0
    assert "/" not in processor.model_filename


async def test_loads_correct_model_from_path(
    trained_core_model: Text, trained_nlu_model: Text, tmp_path: Path
):
    # We move both models to the same directory to prove we can load models by name
    # from a directory with multiple models.
    model_dir = tmp_path / "models"
    os.makedirs(model_dir)

    trained_core_model_name = os.path.basename(trained_core_model)
    shutil.copy2(trained_core_model, model_dir)

    trained_nlu_model_name = os.path.basename(trained_nlu_model)
    shutil.copy2(trained_nlu_model, model_dir)

    core_processor = Agent.load(
        model_path=model_dir / trained_core_model_name
    ).processor
    nlu_processor = Agent.load(model_path=model_dir / trained_nlu_model_name).processor

    assert core_processor.model_filename == trained_core_model_name
    assert nlu_processor.model_filename == trained_nlu_model_name


@pytest.mark.flaky
@pytest.mark.timeout(180, func_only=True)
async def test_custom_action_triggers_action_extract_slots(
    trained_async: Callable,
    caplog: LogCaptureFixture,
):
    parent_folder = "data/test_custom_action_triggers_action_extract_slots"
    domain_path = f"{parent_folder}/domain.yml"
    config_path = f"{parent_folder}/config.yml"
    stories_path = f"{parent_folder}/stories.yml"
    nlu_path = f"{parent_folder}/nlu.yml"

    model_path = await trained_async(domain_path, config_path, [stories_path, nlu_path])
    agent = Agent.load(model_path)
    processor = agent.processor

    action_server_url = "http://some-url"
    endpoint = EndpointConfig(action_server_url)
    processor.action_endpoint = endpoint

    entity_name = "mood"
    slot_name = "mood_slot"
    slot_value = "happy"
    custom_action = "action_force_next_utter"

    sender_id = uuid.uuid4().hex
    message = UserMessage(
        text="Activate custom action.",
        output_channel=CollectingOutputChannel(),
        sender_id=sender_id,
        parse_data={
            "intent": {"name": "activate_flow", "confidence": 1},
            "entities": [],
        },
    )

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "action", "name": "action_listen"},
                    {
                        "event": "user",
                        "text": "Feeling so happy",
                        "parse_data": {
                            "intent": {"name": "mood_great", "confidence": 1.0},
                            "entities": [{"entity": entity_name, "value": slot_value}],
                        },
                    },
                ]
            },
        )
        with caplog.at_level(logging.DEBUG):
            await processor.handle_message(message)

        caplog_records = [rec.message for rec in caplog.records]

        assert (
            f"A `UserUttered` event was returned by executing "
            f"action '{custom_action}'. This will run the default action "
            f"'{ACTION_EXTRACT_SLOTS}'." in caplog_records
        )

    tracker = await processor.get_tracker(sender_id)
    assert any(
        isinstance(e, UserUttered) and e.text == "Feeling so happy"
        for e in tracker.events
    )
    assert SlotSet(slot_name, slot_value) in tracker.events
    assert tracker.get_slot(slot_name) == slot_value
    assert any(
        isinstance(e, BotUttered) and e.text == "Great, carry on!"
        for e in tracker.events
    )


async def test_processor_executes_bot_uttered_returned_by_action_extract_slots(
    default_agent: Agent,
):
    slot_name = "location"
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - inform

        entities:
        - {slot_name}

        slots:
          {slot_name}:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: {slot_name}

        actions:
        - action_validate_slot_mappings
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    processor = default_agent.processor
    processor.domain = domain

    action_server_url = "http:/my-action-server:5055/webhook"
    processor.action_endpoint = EndpointConfig(action_server_url)

    sender_id = uuid.uuid4().hex
    message = UserMessage(
        text="This is a test.",
        output_channel=CollectingOutputChannel(),
        sender_id=sender_id,
        parse_data={
            "intent": {"name": "inform", "confidence": 1},
            "entities": [{"entity": slot_name, "value": "Lisbon"}],
        },
    )

    bot_uttered_text = "This city is not yet supported."

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "bot", "text": bot_uttered_text},
                    {"event": "slot", "name": "location", "value": None},
                ]
            },
        )
        responses = await processor.handle_message(message)
        assert any(bot_uttered_text in r.get("text") for r in responses)

        tracker = await processor.get_tracker(sender_id)
        assert tracker.get_slot(slot_name) is None


@pytest.mark.flaky
@pytest.mark.timeout(180, func_only=True)
@pytest.mark.parametrize(
    "sender_id, message_text, message_intent",
    [
        ("happy_path", "Hi", "greet"),
        ("another_form_activation", "switch forms", "switch_another_form"),
    ],
)
async def test_from_trigger_intent_with_mapping_conditions_when_form_not_activated(
    trained_async: Callable,
    sender_id: Text,
    message_text: Text,
    message_intent: Text,
):
    parent_folder = "data/test_from_trigger_intent_with_mapping_conditions"
    domain_path = f"{parent_folder}/domain.yml"
    config_path = f"{parent_folder}/config.yml"
    stories_path = f"{parent_folder}/stories.yml"
    nlu_path = f"{parent_folder}/nlu.yml"

    model_path = await trained_async(domain_path, config_path, [stories_path, nlu_path])
    agent = Agent.load(model_path)
    processor = agent.processor

    slot_name = "test_trigger"
    slot_value = "testing123"

    user_messages = [
        UserMessage(
            text=message_text,
            output_channel=CollectingOutputChannel(),
            sender_id=sender_id,
            parse_data={
                "intent": {"name": message_intent, "confidence": 1},
                "entities": [],
            },
        ),
        UserMessage(
            text="great",
            output_channel=CollectingOutputChannel(),
            sender_id=sender_id,
            parse_data={
                "intent": {"name": "mood_great", "confidence": 1},
                "entities": [],
            },
        ),
    ]

    for msg in user_messages:
        await processor.handle_message(msg)

    tracker = await processor.get_tracker(sender_id)
    assert SlotSet(slot_name, slot_value) not in tracker.events
    assert tracker.get_slot(slot_name) is None


@pytest.mark.flaky
@pytest.mark.timeout(120, func_only=True)
async def test_from_trigger_intent_no_form_condition_when_form_not_activated(
    trained_async: Callable,
):
    parent_folder = "data/test_from_trigger_intent_with_no_mapping_conditions"
    domain_path = f"{parent_folder}/domain.yml"
    config_path = f"{parent_folder}/config.yml"
    stories_path = f"{parent_folder}/stories.yml"
    nlu_path = f"{parent_folder}/nlu.yml"

    model_path = await trained_async(domain_path, config_path, [stories_path, nlu_path])
    agent = Agent.load(model_path)
    processor = agent.processor

    slot_name = "test_trigger"
    slot_value = "testing123"

    sender_id = uuid.uuid4().hex
    user_messages = [
        UserMessage(
            text="Hi",
            output_channel=CollectingOutputChannel(),
            sender_id=sender_id,
            parse_data={
                "intent": {"name": "greet", "confidence": 1},
                "entities": [],
            },
        ),
        UserMessage(
            text="great",
            output_channel=CollectingOutputChannel(),
            sender_id=sender_id,
            parse_data={
                "intent": {"name": "mood_great", "confidence": 1},
                "entities": [],
            },
        ),
    ]
    for msg in user_messages:
        await processor.handle_message(msg)

    tracker = await processor.get_tracker(sender_id)
    assert SlotSet(slot_name, slot_value) not in tracker.events
    assert tracker.get_slot(slot_name) is None

    # test that the form activation path works as expected
    sender_id_form_activation = "test_form_activation"
    await processor.handle_message(
        UserMessage(
            text="great",
            output_channel=CollectingOutputChannel(),
            sender_id=sender_id_form_activation,
            parse_data={
                "intent": {"name": "mood_great", "confidence": 1},
                "entities": [],
            },
        )
    )

    tracker = await processor.get_tracker(sender_id_form_activation)
    assert ActiveLoop("test_form") in tracker.events
    assert SlotSet(slot_name, slot_value) in tracker.events
    assert tracker.get_slot(slot_name) == slot_value


@pytest.mark.timeout(120, func_only=True)
async def test_message_processor_raises_warning_if_no_assistant_id(
    trained_async: Callable,
):
    parent_folder = "data/test_moodbot"
    domain_path = f"{parent_folder}/domain.yml"
    config_path = "data/test_config/test_moodbot_config_no_assistant_id.yml"
    stories_path = f"{parent_folder}/data/stories.yml"
    nlu_path = f"{parent_folder}/data/nlu.yml"

    model_path = await trained_async(
        domain=domain_path, config=config_path, training_files=[stories_path, nlu_path]
    )
    warning_message = (
        f"The model metadata does not contain a value for the '{ASSISTANT_ID_KEY}' "
        f"attribute. Check that 'config.yml' file contains a value for "
        f"the '{ASSISTANT_ID_KEY}' key and re-train the model. "
        f"Failure to do so will result in streaming events without a "
        f"unique assistant identifier."
    )

    with pytest.warns(UserWarning, match=warning_message):
        Agent.load(model_path)


async def test_processor_fetch_full_tracker_with_initial_session_inexistent_tracker(
    default_processor: MessageProcessor,
) -> None:
    """Test that the tracker is created with the correct initial session data."""
    sender_id = uuid.uuid4().hex
    tracker = await default_processor.fetch_full_tracker_with_initial_session(sender_id)

    assert tracker.sender_id == sender_id
    assert tracker.latest_message == UserUttered.empty()
    assert tracker.latest_action_name == ACTION_LISTEN_NAME
    assert len(tracker.events) == 3

    first_recorded_event = tracker.events[0]
    assert isinstance(first_recorded_event, ActionExecuted)
    assert first_recorded_event.action_name == ACTION_SESSION_START_NAME

    assert isinstance(tracker.events[1], SessionStarted)

    last_recorded_event = tracker.events[2]
    assert isinstance(last_recorded_event, ActionExecuted)
    assert last_recorded_event.action_name == ACTION_LISTEN_NAME


async def test_processor_fetch_full_tracker_with_initial_session_existing_tracker(
    default_processor: MessageProcessor,
):
    """Test that an existing tracker is correctly retrieved."""
    sender_id = uuid.uuid4().hex
    expected_events = [
        UserUttered("hello"),
        Restarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
    tracker = DialogueStateTracker.from_events(sender_id, evts=expected_events)
    await default_processor.save_tracker(tracker)

    tracker = await default_processor.fetch_full_tracker_with_initial_session(sender_id)
    assert tracker.sender_id == sender_id
    assert all([event in expected_events for event in tracker.events])


async def test_run_anonymization_pipeline_no_pipeline(
    monkeypatch: MonkeyPatch,
    default_agent: Agent,
) -> None:
    processor = default_agent.processor
    sender_id = uuid.uuid4().hex
    tracker = await processor.tracker_store.get_or_create_tracker(sender_id)

    manager = plugin_manager()
    monkeypatch.setattr(
        manager.hook, "get_anonymization_pipeline", MagicMock(return_value=None)
    )
    event_diff = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.core.trackers.TrackerEventDiffEngine.event_difference", event_diff
    )
    await processor.run_anonymization_pipeline(tracker)

    event_diff.assert_not_called()


async def test_run_anonymization_pipeline_mocked_pipeline(
    monkeypatch: MonkeyPatch,
    default_agent: Agent,
) -> None:
    processor = default_agent.processor
    sender_id = uuid.uuid4().hex
    tracker = await processor.tracker_store.get_or_create_tracker(sender_id)

    manager = plugin_manager()
    monkeypatch.setattr(
        manager.hook,
        "get_anonymization_pipeline",
        MagicMock(return_value="mock_pipeline"),
    )
    event_diff = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.core.trackers.TrackerEventDiffEngine.event_difference", event_diff
    )
    await processor.run_anonymization_pipeline(tracker)

    event_diff.assert_called_once()


async def test_update_full_retrieval_intent(
    default_processor: MessageProcessor,
) -> None:
    parse_data = {
        "text": "I like sunny days in berlin",
        "intent": {"name": "chitchat", "confidence": 0.9},
        "entities": [],
        "response_selector": {
            "all_retrieval_intents": ["faq", "chitchat"],
            "faq": {
                "response": {
                    "responses": [{"text": "Our return policy lasts 30 days."}],
                    "confidence": 1.0,
                    "intent_response_key": "faq/what_is_return_policy",
                    "utter_action": "utter_faq/what_is_return_policy",
                },
                "ranking": [
                    {
                        "confidence": 1.0,
                        "intent_response_key": "faq/what_is_return_policy",
                    },
                    {
                        "confidence": 2.3378809862799945e-19,
                        "intent_response_key": "faq/how_can_i_track_my_order",
                    },
                ],
            },
            "chitchat": {
                "response": {
                    "responses": [
                        {
                            "text": "The sun is out today! Isn't that great?",
                        },
                    ],
                    "confidence": 1.0,
                    "intent_response_key": "chitchat/ask_weather",
                    "utter_action": "utter_chitchat/ask_weather",
                },
                "ranking": [
                    {
                        "confidence": 1.0,
                        "intent_response_key": "chitchat/ask_weather",
                    },
                    {"confidence": 0.0, "intent_response_key": "chitchat/ask_name"},
                ],
            },
        },
    }

    default_processor._update_full_retrieval_intent(parse_data)

    assert parse_data[INTENT][INTENT_NAME_KEY] == "chitchat"
    # assert that parse_data["intent"] has a key called response
    assert FULL_RETRIEVAL_INTENT_NAME_KEY in parse_data[INTENT]
    assert parse_data[INTENT][FULL_RETRIEVAL_INTENT_NAME_KEY] == "chitchat/ask_weather"
