import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Text, List
from unittest.mock import Mock

import pytest

from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain

from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.brokers.sql import SQLEventBroker
from rasa.core.constants import RASA_EXPORT_PROCESS_ID_HEADER_NAME
from rasa.shared.core.events import Event, SessionStarted, ActionExecuted
from rasa.core.tracker_store import SQLTrackerStore
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.exceptions import (
    NoConversationsInTrackerStoreError,
    NoEventsToMigrateError,
    PublishingError,
)
from tests.conftest import MockExporter, random_user_uttered_event, AsyncMock


@pytest.mark.parametrize(
    "requested_ids,available_ids,expected",
    [(["1"], ["1"], ["1"]), (["1", "2"], ["2"], ["2"]), (None, ["2"], ["2"])],
)
async def test_get_conversation_ids_to_process(
    requested_ids: Optional[List[Text]],
    available_ids: Optional[List[Text]],
    expected: Optional[List[Text]],
):
    # create and mock tracker store containing `available_ids` as keys
    tracker_store = Mock()
    tracker_store.keys = AsyncMock(return_value=available_ids)

    exporter = MockExporter(tracker_store)
    exporter.requested_conversation_ids = requested_ids

    # noinspection PyProtectedMember
    assert await exporter._get_conversation_ids_to_process() == set(expected)


@pytest.mark.parametrize(
    "requested_ids,available_ids,exception",
    [
        (["1"], [], NoConversationsInTrackerStoreError),  # no IDs in tracker store
        (None, [], NoConversationsInTrackerStoreError),  # without requested IDs
        (
            ["1", "2", "3"],
            ["4", "5", "6"],
            NoEventsToMigrateError,
        ),  # no overlap between requested IDs and those available
    ],
)
async def test_get_conversation_ids_to_process_error(
    requested_ids: Optional[List[Text]], available_ids: List[Text], exception: Exception
):
    # create and mock tracker store containing `available_ids` as keys
    tracker_store = Mock()
    tracker_store.keys = AsyncMock(return_value=available_ids)

    exporter = MockExporter(tracker_store)
    exporter.requested_conversation_ids = requested_ids

    with pytest.raises(exception):
        # noinspection PyProtectedMember
        await exporter._get_conversation_ids_to_process()


async def test_fetch_events_within_time_range():
    conversation_ids = ["some-id", "another-id"]

    # prepare events from different senders and different timestamps
    event_1 = random_user_uttered_event(1)
    event_2 = random_user_uttered_event(3)
    event_3 = random_user_uttered_event(2)
    events = {conversation_ids[0]: [event_1, event_2], conversation_ids[1]: [event_3]}

    def _get_tracker(conversation_id: Text) -> DialogueStateTracker:
        return DialogueStateTracker.from_events(
            conversation_id, events[conversation_id]
        )

    # create mock tracker store
    tracker_store = AsyncMock()
    tracker_store.retrieve_full_tracker.side_effect = _get_tracker
    tracker_store.keys = AsyncMock(return_value=conversation_ids)

    exporter = MockExporter(tracker_store)
    exporter.requested_conversation_ids = conversation_ids

    # noinspection PyProtectedMember
    fetched_events = [e async for e in exporter._fetch_events_within_time_range()]

    # events should come back for all requested conversation IDs
    assert all(
        any(_id in event["sender_id"] for event in fetched_events)
        for _id in conversation_ids
    )

    for _id in conversation_ids:
        # events are sorted by timestamp
        event_timestamps = [
            e["timestamp"] for e in fetched_events if e["sender_id"] == _id
        ]
        assert event_timestamps == [e.timestamp for e in events[_id]]


async def test_fetch_events_within_time_range_tracker_does_not_err():
    # create mock tracker store that returns `None` on `retrieve_full_tracker()`
    tracker_store = Mock()

    tracker_store.keys = AsyncMock(return_value=[uuid.uuid4()])
    tracker_store.retrieve_full_tracker = AsyncMock(return_value=None)

    exporter = MockExporter(tracker_store)

    assert not [e async for e in exporter._fetch_events_within_time_range()]


async def test_fetch_events_within_time_range_tracker_contains_no_events():
    # create mock tracker store that returns `None` on `retrieve_full_tracker()`
    tracker_store = Mock()

    tracker_store.keys = AsyncMock(return_value=["a great ID"])
    tracker_store.retrieve_full_tracker = AsyncMock(
        return_value=DialogueStateTracker.from_events("a great ID", [])
    )

    exporter = MockExporter(tracker_store)

    assert not [e async for e in exporter._fetch_events_within_time_range()]


async def mock_tracker_store(
    events: Dict[Text, List[Event]], tmp_path: Path
) -> DialogueStateTracker:
    tracker_store = SQLTrackerStore(
        dialect="sqlite",
        db=str(tmp_path / f"{uuid.uuid4().hex}.db"),
        domain=Domain.empty(),
    )

    for conversation_id, conversation_events in events.items():
        tracker = DialogueStateTracker.from_events(
            conversation_id, evts=conversation_events
        )
        await tracker_store.save(tracker)
    return tracker_store


async def test_fetch_events_within_time_range_with_session_events(tmp_path: Path):
    conversation_id = "test_fetch_events_within_time_range_with_sessions"

    events = {
        conversation_id: [
            random_user_uttered_event(1),
            SessionStarted(2),
            ActionExecuted(timestamp=3, action_name=ACTION_SESSION_START_NAME),
            random_user_uttered_event(4),
        ]
    }
    tracker_store = await mock_tracker_store(events, tmp_path)

    exporter = MockExporter(tracker_store=tracker_store)

    # noinspection PyProtectedMember
    fetched_events = [e async for e in exporter._fetch_events_within_time_range()]

    assert len(fetched_events) == len(events[conversation_id])


# noinspection PyProtectedMember
async def test_sort_and_select_events_by_timestamp(tmp_path: Path):
    conversation_id = "test_sort_and_select_events_by_timestamp"

    conversations = {
        conversation_id: [
            random_user_uttered_event(3),
            random_user_uttered_event(2),
            random_user_uttered_event(1),
        ]
    }
    tracker_store = await mock_tracker_store(conversations, tmp_path)
    exporter = MockExporter(tracker_store)
    event_ts = [e.timestamp for e in conversations[conversation_id]]

    selected_events = [e async for e in exporter._fetch_events_within_time_range()]

    # events are sorted
    assert selected_events == list(
        sorted(selected_events, key=lambda e: e["timestamp"])
    )

    # apply minimum timestamp requirement, expect to get only two events back
    exporter.minimum_timestamp = 2.0
    selected_events = [e async for e in exporter._fetch_events_within_time_range()]
    assert [e["timestamp"] for e in selected_events] == [
        event_ts[1],
        event_ts[0],
    ]
    exporter.minimum_timestamp = None

    # apply maximum timestamp requirement, expect to get only one
    exporter.maximum_timestamp = 1.1
    selected_events = [e async for e in exporter._fetch_events_within_time_range()]
    assert [e["timestamp"] for e in selected_events] == [event_ts[2]]

    # apply both requirements, get one event back
    exporter.minimum_timestamp = 2.0
    exporter.maximum_timestamp = 2.1
    selected_events = [e async for e in exporter._fetch_events_within_time_range()]
    assert [e["timestamp"] for e in selected_events] == [event_ts[1]]


# noinspection PyProtectedMember
async def test_sort_and_select_events_by_timestamp_error(tmp_path: Path):
    conversation_id = "test_sort_and_select_events_by_timestamp_error"

    conversations = {
        conversation_id: [
            random_user_uttered_event(3),
        ]
    }
    tracker_store = await mock_tracker_store(conversations, tmp_path)
    exporter = MockExporter(tracker_store)

    # supply list of events, apply timestamp constraint and no events survive
    exporter.minimum_timestamp = 3.1
    assert not [e async for e in exporter._fetch_events_within_time_range()]


def test_get_message_headers_pika_event_broker():
    event_broker = Mock(spec=PikaEventBroker)
    exporter = MockExporter(event_broker=event_broker)

    # noinspection PyProtectedMember
    headers = exporter._get_message_headers()

    assert headers.get(RASA_EXPORT_PROCESS_ID_HEADER_NAME)


def test_get_message_headers_non_pika_broker():
    event_broker = Mock()
    exporter = MockExporter(event_broker=event_broker)

    # noinspection PyProtectedMember
    assert exporter._get_message_headers() is None


def test_publish_with_headers_pika_event_broker():
    event_broker = Mock(spec=PikaEventBroker)
    exporter = MockExporter(event_broker=event_broker)

    headers = {"some": "header"}
    event = {"some": "event"}

    # noinspection PyProtectedMember
    exporter._publish_with_message_headers(event, headers)

    # the `PikaEventBroker`'s `publish()` method was called with both
    # the `event` and `headers` arguments
    event_broker.publish.assert_called_with(event=event, headers=headers)


def test_publish_with_headers_non_pika_event_broker():
    event_broker = Mock(SQLEventBroker)
    exporter = MockExporter(event_broker=event_broker)

    headers = {"some": "header"}
    event = {"some": "event"}

    # noinspection PyProtectedMember
    exporter._publish_with_message_headers(event, headers)

    # the `SQLEventBroker`'s `publish()` method was called with only the `event`
    # argument
    event_broker.publish.assert_called_with(event)


async def test_publishing_error():
    # mock event broker so it raises on `publish()`

    event_broker = Mock()
    event_broker.publish.side_effect = ValueError()

    exporter = MockExporter(event_broker=event_broker)

    user_event = random_user_uttered_event(1).as_dict()
    user_event["sender_id"] = uuid.uuid4().hex

    async def _mocked_fetch() -> AsyncIterator[Dict[Text, Any]]:
        yield user_event

    # noinspection PyProtectedMember
    exporter._fetch_events_within_time_range = _mocked_fetch

    # run the export function
    with pytest.raises(PublishingError):
        await exporter.publish_events()


async def test_closing_broker():
    exporter = MockExporter(event_broker=SQLEventBroker())

    # noinspection PyProtectedMember
    async def _mocked_fetch() -> AsyncIterator[Dict[Text, Any]]:
        # need an async generator that is empty
        if False:
            yield
        return

    # noinspection PyProtectedMember
    exporter._fetch_events_within_time_range = _mocked_fetch

    # run the export function
    with pytest.warns(None) as warnings:
        await exporter.publish_events()

    assert len(warnings) == 0
