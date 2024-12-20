from __future__ import annotations
import contextlib
import itertools
import json
import logging
import os
from inspect import isawaitable, iscoroutinefunction

from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Text,
    Union,
    TYPE_CHECKING,
    Generator,
    TypeVar,
    Generic,
)

from boto3.dynamodb.conditions import Key
from pymongo.collection import Collection

import rasa.core.utils as core_utils
import rasa.shared.utils.cli
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.plugin import plugin_manager
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.core.brokers.broker import EventBroker
from rasa.core.constants import (
    POSTGRESQL_SCHEMA,
    POSTGRESQL_MAX_OVERFLOW,
    POSTGRESQL_POOL_SIZE,
)
from rasa.shared.core.conversation import Dialogue
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SessionStarted, Event
from rasa.shared.core.trackers import (
    ActionExecuted,
    DialogueStateTracker,
    EventVerbosity,
    TrackerEventDiffEngine,
)
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.utils.endpoints import EndpointConfig
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta

if TYPE_CHECKING:
    import boto3.resources.factory.dynamodb.Table
    from sqlalchemy.engine.url import URL
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.orm import Session, Query
    from sqlalchemy import Sequence

logger = logging.getLogger(__name__)

# default values of PostgreSQL pool size and max overflow
POSTGRESQL_DEFAULT_MAX_OVERFLOW = 100
POSTGRESQL_DEFAULT_POOL_SIZE = 50

# default value for key prefix in RedisTrackerStore
DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX = "tracker:"


def check_if_tracker_store_async(tracker_store: TrackerStore) -> bool:
    """Evaluates if a tracker store object is async based on implementation of methods.

    :param tracker_store: tracker store object we're evaluating
    :return: if the tracker store correctly implements all async methods
    """
    return all(
        iscoroutinefunction(getattr(tracker_store, method))
        for method in _get_async_tracker_store_methods()
    )


def _get_async_tracker_store_methods() -> List[str]:
    return [
        attribute
        for attribute in dir(TrackerStore)
        if iscoroutinefunction(getattr(TrackerStore, attribute))
    ]


class TrackerDeserialisationException(RasaException):
    """Raised when an error is encountered while deserialising a tracker."""


SerializationType = TypeVar("SerializationType")


class SerializedTrackerRepresentation(Generic[SerializationType]):
    """Mixin class for specifying different serialization methods per tracker store."""

    @staticmethod
    def serialise_tracker(tracker: DialogueStateTracker) -> SerializationType:
        """Requires implementation to return representation of tracker."""
        raise NotImplementedError()


class SerializedTrackerAsText(SerializedTrackerRepresentation[Text]):
    """Mixin class that returns the serialized tracker as string."""

    @staticmethod
    def serialise_tracker(tracker: DialogueStateTracker) -> Text:
        """Serializes the tracker, returns representation of the tracker."""
        dialogue = tracker.as_dialogue()

        return json.dumps(dialogue.as_dict())


class SerializedTrackerAsDict(SerializedTrackerRepresentation[Dict]):
    """Mixin class that returns the serialized tracker as dictionary."""

    @staticmethod
    def serialise_tracker(tracker: DialogueStateTracker) -> Dict:
        """Serializes the tracker, returns representation of the tracker."""
        d = tracker.as_dialogue().as_dict()
        d.update({"sender_id": tracker.sender_id})
        return d


class TrackerStore:
    """Represents common behavior and interface for all `TrackerStore`s."""

    def __init__(
        self,
        domain: Optional[Domain],
        event_broker: Optional[EventBroker] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Create a TrackerStore.

        Args:
            domain: The `Domain` to initialize the `DialogueStateTracker`.
            event_broker: An event broker to publish any new events to another
                destination.
            kwargs: Additional kwargs.
        """
        self._domain = domain or Domain.empty()
        self.event_broker = event_broker
        self.max_event_history: Optional[int] = None

    @staticmethod
    def create(
        obj: Union[TrackerStore, EndpointConfig, None],
        domain: Optional[Domain] = None,
        event_broker: Optional[EventBroker] = None,
    ) -> TrackerStore:
        """Factory to create a tracker store."""
        if isinstance(obj, TrackerStore):
            return obj

        from botocore.exceptions import BotoCoreError
        import pymongo.errors
        import sqlalchemy.exc

        try:
            _tracker_store = plugin_manager().hook.create_tracker_store(
                endpoint_config=obj,
                domain=domain,
                event_broker=event_broker,
            )

            tracker_store = (
                _tracker_store
                if _tracker_store
                else create_tracker_store(obj, domain, event_broker)
            )

            return tracker_store
        except (
            BotoCoreError,
            pymongo.errors.ConnectionFailure,
            sqlalchemy.exc.OperationalError,
            ConnectionError,
            pymongo.errors.OperationFailure,
        ) as error:
            raise ConnectionException(
                "Cannot connect to tracker store." + str(error)
            ) from error

    async def get_or_create_tracker(
        self,
        sender_id: Text,
        max_event_history: Optional[int] = None,
        append_action_listen: bool = True,
    ) -> "DialogueStateTracker":
        """Returns tracker or creates one if the retrieval returns None.

        Args:
            sender_id: Conversation ID associated with the requested tracker.
            max_event_history: Value to update the tracker store's max event history to.
            append_action_listen: Whether or not to append an initial `action_listen`.
        """
        self.max_event_history = max_event_history

        tracker = await self.retrieve(sender_id)

        if tracker is None:
            tracker = await self.create_tracker(
                sender_id, append_action_listen=append_action_listen
            )

        return tracker

    def init_tracker(self, sender_id: Text) -> "DialogueStateTracker":
        """Returns a Dialogue State Tracker."""
        return DialogueStateTracker(
            sender_id,
            self.domain.slots,
            max_event_history=self.max_event_history,
        )

    async def create_tracker(
        self, sender_id: Text, append_action_listen: bool = True
    ) -> DialogueStateTracker:
        """Creates a new tracker for `sender_id`.

        The tracker begins with a `SessionStarted` event and is initially listening.

        Args:
            sender_id: Conversation ID associated with the tracker.
            append_action_listen: Whether or not to append an initial `action_listen`.

        Returns:
            The newly created tracker for `sender_id`.
        """
        tracker = self.init_tracker(sender_id)

        if append_action_listen:
            tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

        await self.save(tracker)

        return tracker

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Save method that will be overridden by specific tracker."""
        raise NotImplementedError()

    async def exists(self, conversation_id: Text) -> bool:
        """Checks if tracker exists for the specified ID.

        This method may be overridden by the specific tracker store for
        faster implementations.

        Args:
            conversation_id: Conversation ID to check if the tracker exists.

        Returns:
            `True` if the tracker exists, `False` otherwise.
        """
        return await self.retrieve(conversation_id) is not None

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session.

        This method will be overridden by the specific tracker store.

        Args:
            sender_id: Conversation ID to fetch the tracker for.

        Returns:
            Tracker containing events from the latest conversation sessions.
        """
        raise NotImplementedError()

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Retrieve method for fetching all tracker events across conversation sessions\
        that may be overridden by specific tracker.

        The default implementation uses `self.retrieve()`.

        Args:
            conversation_id: The conversation ID to retrieve the tracker for.

        Returns:
            The fetch tracker containing all events across session starts.
        """
        return await self.retrieve(conversation_id)

    async def get_or_create_full_tracker(
        self,
        sender_id: Text,
        append_action_listen: bool = True,
    ) -> "DialogueStateTracker":
        """Returns tracker or creates one if the retrieval returns None.

        Args:
            sender_id: Conversation ID associated with the requested tracker.
            append_action_listen: Whether to append an initial `action_listen`.

        Returns:
            The tracker for the conversation ID.
        """
        tracker = await self.retrieve_full_tracker(sender_id)

        if tracker is None:
            tracker = await self.create_tracker(
                sender_id, append_action_listen=append_action_listen
            )

        return tracker

    async def stream_events(self, tracker: DialogueStateTracker) -> None:
        """Streams events to a message broker."""
        if self.event_broker is None:
            logger.debug("No event broker configured. Skipping streaming events.")
            return None

        old_tracker = await self.retrieve(tracker.sender_id)
        new_events = TrackerEventDiffEngine.event_difference(old_tracker, tracker)

        await self._stream_new_events(self.event_broker, new_events, tracker.sender_id)

    async def _stream_new_events(
        self,
        event_broker: EventBroker,
        new_events: List[Event],
        sender_id: Text,
    ) -> None:
        """Publishes new tracker events to a message broker."""
        for event in new_events:
            body = {"sender_id": sender_id}
            body.update(event.as_dict())
            event_broker.publish(body)

    async def keys(self) -> Iterable[Text]:
        """Returns the set of values for the tracker store's primary key."""
        raise NotImplementedError()

    def deserialise_tracker(
        self, sender_id: Text, serialised_tracker: Union[Text, bytes]
    ) -> Optional[DialogueStateTracker]:
        """Deserializes the tracker and returns it."""
        tracker = self.init_tracker(sender_id)

        try:
            dialogue = Dialogue.from_parameters(json.loads(serialised_tracker))
        except UnicodeDecodeError as e:
            raise TrackerDeserialisationException(
                "Tracker cannot be deserialised. "
                "Trackers must be serialised as json. "
                "Support for deserialising pickled trackers has been removed."
            ) from e

        tracker.recreate_from_dialogue(dialogue)

        return tracker

    @property
    def domain(self) -> Domain:
        """Returns the domain of the tracker store."""
        return self._domain

    @domain.setter
    def domain(self, domain: Optional[Domain]) -> None:
        self._domain = domain or Domain.empty()


class InMemoryTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Stores conversation history in memory."""

    def __init__(
        self,
        domain: Domain,
        event_broker: Optional[EventBroker] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initializes the tracker store."""
        self.store: Dict[Text, Text] = {}
        super().__init__(domain, event_broker, **kwargs)

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Updates and saves the current conversation state."""
        await self.stream_events(tracker)
        serialised = InMemoryTrackerStore.serialise_tracker(tracker)
        self.store[tracker.sender_id] = serialised

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id."""
        return await self._retrieve(sender_id, fetch_all_sessions=False)

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the Tracker Store in memory."""
        return self.store.keys()

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=True)

    async def _retrieve(
        self, sender_id: Text, fetch_all_sessions: bool
    ) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
            fetch_all_sessions: Whether to fetch all sessions or only the last one.
        """
        if sender_id not in self.store:
            logger.debug(f"Could not find tracker for conversation ID '{sender_id}'.")
            return None

        logger.debug(f"Recreating tracker for id '{sender_id}'")

        tracker = self.deserialise_tracker(sender_id, self.store[sender_id])

        if not tracker:
            logger.debug(f"Could not find tracker for conversation ID '{sender_id}'.")
            return None

        if fetch_all_sessions:
            return tracker

        # only return the last session
        multiple_tracker_sessions = (
            rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)
        )

        if 0 <= len(multiple_tracker_sessions) <= 1:
            return tracker

        return multiple_tracker_sessions[-1]


class RedisTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Stores conversation history in Redis."""

    def __init__(
        self,
        domain: Domain,
        host: Text = "localhost",
        port: int = 6379,
        db: int = 0,
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        event_broker: Optional[EventBroker] = None,
        record_exp: Optional[float] = None,
        key_prefix: Optional[Text] = None,
        use_ssl: bool = False,
        ssl_keyfile: Optional[Text] = None,
        ssl_certfile: Optional[Text] = None,
        ssl_ca_certs: Optional[Text] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initializes the tracker store."""
        import redis

        self.red = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            username=username,
            password=password,
            ssl=use_ssl,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_ca_certs=ssl_ca_certs,
            decode_responses=True,
        )
        self.record_exp = record_exp

        self.key_prefix = DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX
        if key_prefix:
            logger.debug(f"Setting non-default redis key prefix: '{key_prefix}'.")
            self._set_key_prefix(key_prefix)

        super().__init__(domain, event_broker, **kwargs)

    def _set_key_prefix(self, key_prefix: Text) -> None:
        if isinstance(key_prefix, str) and key_prefix.isalnum():
            self.key_prefix = key_prefix + ":" + DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX
        else:
            logger.warning(
                f"Omitting provided non-alphanumeric redis key prefix: '{key_prefix}'. "
                f"Using default '{self.key_prefix}' instead."
            )

    def _get_key_prefix(self) -> Text:
        return self.key_prefix

    async def save(
        self, tracker: DialogueStateTracker, timeout: Optional[float] = None
    ) -> None:
        """Saves the current conversation state."""
        await self.stream_events(tracker)

        if not timeout and self.record_exp:
            timeout = self.record_exp

        stored = self.red.get(self.key_prefix + tracker.sender_id)

        if stored is not None:
            prior_tracker = self.deserialise_tracker(tracker.sender_id, stored)

            tracker = self._merge_trackers(prior_tracker, tracker)

        serialised_tracker = self.serialise_tracker(tracker)
        self.red.set(
            self.key_prefix + tracker.sender_id, serialised_tracker, ex=timeout
        )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session.

        The Redis key is formed by appending a prefix to sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.

        Returns:
            Tracker containing events from the latest conversation sessions.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=False)

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for all conversation sessions.

        The Redis key is formed by appending a prefix to sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.

        Returns:
            Tracker containing events from all conversation sessions.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=True)

    async def _retrieve(
        self, sender_id: Text, fetch_all_sessions: bool
    ) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
            fetch_all_sessions: Whether to fetch all sessions or only the last one.
        """
        stored = self.red.get(self.key_prefix + sender_id)
        if stored is None:
            logger.debug(f"Could not find tracker for conversation ID '{sender_id}'.")
            return None

        tracker = self.deserialise_tracker(sender_id, stored)
        if fetch_all_sessions:
            return tracker

        # only return the last session
        multiple_tracker_sessions = (
            rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)
        )

        if 0 <= len(multiple_tracker_sessions) <= 1:
            return tracker

        return multiple_tracker_sessions[-1]

    async def keys(self) -> Iterable[Text]:
        """Returns keys of the Redis Tracker Store."""
        return self.red.keys(self.key_prefix + "*")

    @staticmethod
    def _merge_trackers(
        prior_tracker: DialogueStateTracker, tracker: DialogueStateTracker
    ) -> DialogueStateTracker:
        """Merges two trackers.

        Args:
            prior_tracker: Tracker containing events from the previous conversation
                sessions.
            tracker: Tracker containing events from the current conversation session.
        """
        if not prior_tracker.events:
            return tracker

        last_event_timestamp = prior_tracker.events[-1].timestamp
        past_tracker = tracker.travel_back_in_time(target_time=last_event_timestamp)

        if past_tracker.events == prior_tracker.events:
            return tracker

        merged = tracker.init_copy()
        merged.update_with_events(
            list(prior_tracker.events), override_timestamp=False, domain=None
        )

        for new_event in tracker.events:
            # Event subclasses implement `__eq__` method that make it difficult
            # to compare events. We use `as_dict` to compare events.
            if all(
                [
                    new_event.as_dict() != existing_event.as_dict()
                    for existing_event in merged.events
                ]
            ):
                merged.update(new_event)

        return merged


class DynamoTrackerStore(TrackerStore, SerializedTrackerAsDict):
    """Stores conversation history in DynamoDB."""

    def __init__(
        self,
        domain: Domain,
        table_name: Text = "states",
        region: Text = "us-east-1",
        event_broker: Optional[EndpointConfig] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initialize `DynamoTrackerStore`.

        Args:
            domain: Domain associated with this tracker store.
            table_name: The name of the DynamoDB table, does not need to be present a
                priori.
            region: The name of the region associated with the client.
                A client is associated with a single region.
            event_broker: An event broker used to publish events.
            kwargs: Additional kwargs.
        """
        import boto3

        self.client = boto3.client("dynamodb", region_name=region)
        self.region = region
        self.table_name = table_name
        self.db = self.get_or_create_table(table_name)
        super().__init__(domain, event_broker, **kwargs)

    def get_or_create_table(
        self, table_name: Text
    ) -> "boto3.resources.factory.dynamodb.Table":
        """Returns table or creates one if the table name is not in the table list."""
        import boto3

        dynamo = boto3.resource("dynamodb", region_name=self.region)
        try:
            self.client.describe_table(TableName=table_name)
        except self.client.exceptions.ResourceNotFoundException:
            table = dynamo.create_table(
                TableName=self.table_name,
                KeySchema=[{"AttributeName": "sender_id", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "sender_id", "AttributeType": "S"}
                ],
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )

            # Wait until the table exists.
            table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
        else:
            table = dynamo.Table(table_name)

        return table

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Saves the current conversation state."""
        await self.stream_events(tracker)
        serialized = self.serialise_tracker(tracker)

        self.db.put_item(Item=serialized)

    @staticmethod
    def serialise_tracker(
        tracker: "DialogueStateTracker",
    ) -> Dict:
        """Serializes the tracker, returns object with decimal types.

        DynamoDB cannot store `float`s, so we'll convert them to `Decimal`s.
        """
        return core_utils.replace_floats_with_decimals(
            SerializedTrackerAsDict.serialise_tracker(tracker)
        )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieve dialogues for a sender_id in reverse-chronological order.

        Based on the session_date sort key.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=False)

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for all conversation sessions.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=True)

    async def _retrieve(
        self, sender_id: Text, fetch_all_sessions: bool
    ) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
            fetch_all_sessions: Whether to fetch all sessions or only the last one.
        """
        dialogues = self.db.query(
            KeyConditionExpression=Key("sender_id").eq(sender_id),
            ScanIndexForward=False,
        )["Items"]

        if not dialogues:
            return None

        if fetch_all_sessions:
            events_with_floats = []
            for dialogue in dialogues:
                if dialogue.get("events"):
                    events = core_utils.replace_decimals_with_floats(dialogue["events"])
                    events_with_floats += events
        else:
            events = dialogues[0].get("events", [])
            # `float`s are stored as `Decimal` objects - we need to convert them back
            events_with_floats = core_utils.replace_decimals_with_floats(events)

        if self.domain is None:
            slots = []
        else:
            slots = self.domain.slots

        return DialogueStateTracker.from_dict(sender_id, events_with_floats, slots)

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the `DynamoTrackerStore`."""
        response = self.db.scan(ProjectionExpression="sender_id")
        sender_ids = [i["sender_id"] for i in response["Items"]]

        while response.get("LastEvaluatedKey"):
            response = self.db.scan(
                ProjectionExpression="sender_id",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            sender_ids.extend([i["sender_id"] for i in response["Items"]])

        return sender_ids


class MongoTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Stores conversation history in Mongo.

    Property methods:
        conversations: returns the current conversation
    """

    def __init__(
        self,
        domain: Domain,
        host: Optional[Text] = "mongodb://localhost:27017",
        db: Optional[Text] = "rasa",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        auth_source: Optional[Text] = "admin",
        collection: Text = "conversations",
        event_broker: Optional[EventBroker] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        from pymongo.database import Database
        from pymongo import MongoClient

        self.client: MongoClient = MongoClient(
            host,
            username=username,
            password=password,
            authSource=auth_source,
            # delay connect until process forking is done
            connect=False,
        )

        self.db = Database(self.client, db)
        self.collection = collection
        super().__init__(domain, event_broker, **kwargs)

        self._ensure_indices()

    @property
    def conversations(self) -> Collection:
        """Returns the current conversation."""
        return self.db[self.collection]

    def _ensure_indices(self) -> None:
        """Create an index on the sender_id."""
        self.conversations.create_index("sender_id")

    @staticmethod
    def _current_tracker_state_without_events(tracker: DialogueStateTracker) -> Dict:
        # get current tracker state and remove `events` key from state
        # since events are pushed separately in the `update_one()` operation
        state = tracker.current_state(EventVerbosity.ALL)
        state.pop("events", None)

        return state

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Saves the current conversation state."""
        await self.stream_events(tracker)

        additional_events = self._additional_events(tracker)

        self.conversations.update_one(
            {"sender_id": tracker.sender_id},
            {
                "$set": self._current_tracker_state_without_events(tracker),
                "$push": {
                    "events": {"$each": [e.as_dict() for e in additional_events]}
                },
            },
            upsert=True,
        )

    def _additional_events(self, tracker: DialogueStateTracker) -> Iterator:
        """Return events from the tracker which aren't currently stored.

        Args:
            tracker: Tracker to inspect.

        Returns:
            List of serialised events that aren't currently stored.

        """
        stored = self.conversations.find_one({"sender_id": tracker.sender_id}) or {}
        all_events = self._events_from_serialized_tracker(stored)

        number_events_since_last_session = len(
            self._events_since_last_session_start(all_events)
        )

        return itertools.islice(
            tracker.events, number_events_since_last_session, len(tracker.events)
        )

    @staticmethod
    def _events_from_serialized_tracker(serialised: Dict) -> List[Dict]:
        return serialised.get("events", [])

    @staticmethod
    def _events_since_last_session_start(events: List[Dict]) -> List[Dict]:
        """Retrieve events since and including the latest `SessionStart` event.

        Args:
            events: All events for a conversation ID.

        Returns:
            List of serialised events since and including the latest `SessionStarted`
            event. Returns all events if no such event is found.

        """
        events_after_session_start = []
        for event in reversed(events):
            events_after_session_start.append(event)
            if event["event"] == SessionStarted.type_name:
                break

        return list(reversed(events_after_session_start))

    async def _retrieve(
        self, sender_id: Text, fetch_events_from_all_sessions: bool
    ) -> Optional[List[Dict[Text, Any]]]:
        stored = self.conversations.find_one({"sender_id": sender_id})

        # look for conversations which have used an `int` sender_id in the past
        # and update them.
        if not stored and sender_id.isdigit():
            from pymongo import ReturnDocument

            stored = self.conversations.find_one_and_update(
                {"sender_id": int(sender_id)},
                {"$set": {"sender_id": str(sender_id)}},
                return_document=ReturnDocument.AFTER,
            )

        if not stored:
            return None

        events = self._events_from_serialized_tracker(stored)

        if not fetch_events_from_all_sessions:
            events = self._events_since_last_session_start(events)

        return events

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session."""
        events = await self._retrieve(sender_id, fetch_events_from_all_sessions=False)

        if not events:
            return None

        return DialogueStateTracker.from_dict(sender_id, events, self.domain.slots)

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Fetching all tracker events across conversation sessions."""
        events = await self._retrieve(
            conversation_id, fetch_events_from_all_sessions=True
        )

        if not events:
            return None

        return DialogueStateTracker.from_dict(
            conversation_id, events, self.domain.slots
        )

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the Mongo Tracker Store."""
        return [c["sender_id"] for c in self.conversations.find()]


def _create_sequence(table_name: Text) -> "Sequence":
    """Creates a sequence object for a specific table name.

    If using Oracle you will need to create a sequence in your database,
    as described here: https://rasa.com/docs/rasa/tracker-stores#sqltrackerstore
    Args:
        table_name: The name of the table, which gets a Sequence assigned

    Returns: A `Sequence` object
    """
    from sqlalchemy.ext.declarative import declarative_base

    sequence_name = f"{table_name}_seq"
    Base = declarative_base()
    return sa.Sequence(sequence_name, metadata=Base.metadata, optional=True)


def is_postgresql_url(url: Union[Text, "URL"]) -> bool:
    """Determine whether `url` configures a PostgreSQL connection.

    Args:
        url: SQL connection URL.

    Returns:
        `True` if `url` is a PostgreSQL connection URL.
    """
    if isinstance(url, str):
        return "postgresql" in url

    return url.drivername == "postgresql"


def create_engine_kwargs(url: Union[Text, "URL"]) -> Dict[Text, Any]:
    """Get `sqlalchemy.create_engine()` kwargs.

    Args:
        url: SQL connection URL.

    Returns:
        kwargs to be passed into `sqlalchemy.create_engine()`.
    """
    if not is_postgresql_url(url):
        return {}

    kwargs: Dict[Text, Any] = {}

    schema_name = os.environ.get(POSTGRESQL_SCHEMA)

    if schema_name:
        logger.debug(f"Using PostgreSQL schema '{schema_name}'.")
        kwargs["connect_args"] = {"options": f"-csearch_path={schema_name}"}

    # pool_size and max_overflow can be set to control the number of
    # connections that are kept in the connection pool. Not available
    # for SQLite, and only  tested for PostgreSQL. See
    # https://docs.sqlalchemy.org/en/13/core/pooling.html#sqlalchemy.pool.QueuePool
    kwargs["pool_size"] = int(
        os.environ.get(POSTGRESQL_POOL_SIZE, POSTGRESQL_DEFAULT_POOL_SIZE)
    )
    kwargs["max_overflow"] = int(
        os.environ.get(POSTGRESQL_MAX_OVERFLOW, POSTGRESQL_DEFAULT_MAX_OVERFLOW)
    )

    return kwargs


def ensure_schema_exists(session: "Session") -> None:
    """Ensure that the requested PostgreSQL schema exists in the database.

    Args:
        session: Session used to inspect the database.

    Raises:
        `ValueError` if the requested schema does not exist.
    """
    schema_name = os.environ.get(POSTGRESQL_SCHEMA)

    if not schema_name:
        return

    engine = session.get_bind()

    if is_postgresql_url(engine.url):
        query = sa.exists(
            sa.select([(sa.text("schema_name"))])
            .select_from(sa.text("information_schema.schemata"))
            .where(sa.text(f"schema_name = '{schema_name}'"))
        )
        if not session.query(query).scalar():
            raise ValueError(schema_name)


def validate_port(port: Any) -> Optional[int]:
    """Ensure that port can be converted to integer.

    Raises:
        RasaException if port cannot be cast to integer.
    """
    if port is not None and not isinstance(port, int):
        try:
            port = int(port)
        except ValueError as e:
            raise RasaException(f"The port '{port}' cannot be cast to integer.") from e

    return port


class SQLTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Store which can save and retrieve trackers from an SQL database."""

    Base: DeclarativeMeta = declarative_base()

    class SQLEvent(Base):
        """Represents an event in the SQL Tracker Store."""

        __tablename__ = "events"

        # `create_sequence` is needed to create a sequence for databases that
        # don't autoincrement Integer primary keys (e.g. Oracle)
        id = sa.Column(sa.Integer, _create_sequence(__tablename__), primary_key=True)
        sender_id = sa.Column(sa.String(255), nullable=False, index=True)
        type_name = sa.Column(sa.String(255), nullable=False)
        timestamp = sa.Column(sa.Float)
        intent_name = sa.Column(sa.String(255))
        action_name = sa.Column(sa.String(255))
        data = sa.Column(sa.Text)

    def __init__(
        self,
        domain: Optional[Domain] = None,
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa.db",
        username: Text = None,
        password: Text = None,
        event_broker: Optional[EventBroker] = None,
        login_db: Optional[Text] = None,
        query: Optional[Dict] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        import sqlalchemy.exc

        port = validate_port(port)

        engine_url = self.get_db_url(
            dialect, host, port, db, username, password, login_db, query
        )

        self.engine = sa.create_engine(engine_url, **create_engine_kwargs(engine_url))

        logger.debug(
            f"Attempting to connect to database via '{repr(self.engine.url)}'."
        )

        # Database might take a while to come up
        while True:
            try:
                # if `login_db` has been provided, use current channel with
                # that database to create working database `db`
                if login_db:
                    self._create_database_and_update_engine(db, engine_url)

                try:
                    self.Base.metadata.create_all(self.engine)
                except (
                    sqlalchemy.exc.OperationalError,
                    sqlalchemy.exc.ProgrammingError,
                ) as e:
                    # Several Rasa services started in parallel may attempt to
                    # create tables at the same time. That is okay so long as
                    # the first services finishes the table creation.
                    logger.error(f"Could not create tables: {e}")

                self.sessionmaker = sa.orm.session.sessionmaker(bind=self.engine)
                break
            except (
                sqlalchemy.exc.OperationalError,
                sqlalchemy.exc.IntegrityError,
            ) as error:

                logger.warning(error)
                sleep(5)

        logger.debug(f"Connection to SQL database '{db}' successful.")

        super().__init__(domain, event_broker, **kwargs)

    @staticmethod
    def get_db_url(
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa.db",
        username: Text = None,
        password: Text = None,
        login_db: Optional[Text] = None,
        query: Optional[Dict] = None,
    ) -> Union[Text, "URL"]:
        """Build an SQLAlchemy `URL` object representing the parameters needed
        to connect to an SQL database.

        Args:
            dialect: SQL database type.
            host: Database network host.
            port: Database network port.
            db: Database name.
            username: User name to use when connecting to the database.
            password: Password for database user.
            login_db: Alternative database name to which initially connect, and create
                the database specified by `db` (PostgreSQL only).
            query: Dictionary of options to be passed to the dialect and/or the
                DBAPI upon connect.

        Returns:
            URL ready to be used with an SQLAlchemy `Engine` object.
        """
        from urllib import parse

        # Users might specify a url in the host
        if host and "://" in host:
            # assumes this is a complete database host name including
            # e.g. `postgres://...`
            return host
        elif host:
            # add fake scheme to properly parse components
            parsed = parse.urlsplit(f"scheme://{host}")

            # users might include the port in the url
            port = parsed.port or port
            host = parsed.hostname or host

        return sa.engine.url.URL(
            dialect,
            username,
            password,
            host,
            port,
            database=login_db if login_db else db,
            query=query,
        )

    def _create_database_and_update_engine(self, db: Text, engine_url: "URL") -> None:
        """Creates database `db` and updates engine accordingly."""
        from sqlalchemy import create_engine

        if not self.engine.dialect.name == "postgresql":
            rasa.shared.utils.io.raise_warning(
                "The parameter 'login_db' can only be used with a postgres database."
            )
            return

        self._create_database(self.engine, db)
        self.engine.dispose()
        engine_url = sa.engine.url.URL(
            drivername=engine_url.drivername,
            username=engine_url.username,
            password=engine_url.password,
            host=engine_url.host,
            port=engine_url.port,
            database=db,
            query=engine_url.query,
        )
        self.engine = create_engine(engine_url)

    @staticmethod
    def _create_database(engine: "Engine", database_name: Text) -> None:
        """Create database `db` on `engine` if it does not exist."""
        import sqlalchemy.exc

        conn = engine.connect()

        matching_rows = (
            conn.execution_options(isolation_level="AUTOCOMMIT")
            .execute(
                sa.text(
                    "SELECT 1 FROM pg_catalog.pg_database "
                    "WHERE datname = :database_name"
                ),
                database_name=database_name,
            )
            .rowcount
        )

        if not matching_rows:
            try:
                conn.execute(f"CREATE DATABASE {database_name}")
            except (
                sqlalchemy.exc.ProgrammingError,
                sqlalchemy.exc.IntegrityError,
            ) as e:
                logger.error(f"Could not create database '{database_name}': {e}")

        conn.close()

    @contextlib.contextmanager
    def session_scope(self) -> Generator["Session", None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.sessionmaker()
        try:
            ensure_schema_exists(session)
            yield session
        except ValueError as e:
            rasa.shared.utils.cli.print_error_and_exit(
                f"Requested PostgreSQL schema '{e}' was not found in the database. To "
                f"continue, please create the schema by running 'CREATE DATABASE {e};' "
                f"or unset the '{POSTGRESQL_SCHEMA}' environment variable in order to "
                f"use the default schema. Exiting application."
            )
        finally:
            session.close()

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the SQLTrackerStore."""
        with self.session_scope() as session:
            sender_ids = session.query(self.SQLEvent.sender_id).distinct().all()
            return [sender_id for (sender_id,) in sender_ids]

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session."""
        return await self._retrieve(sender_id, fetch_events_from_all_sessions=False)

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Fetching all tracker events across conversation sessions."""
        return await self._retrieve(
            conversation_id, fetch_events_from_all_sessions=True
        )

    async def _retrieve(
        self, sender_id: Text, fetch_events_from_all_sessions: bool
    ) -> Optional[DialogueStateTracker]:
        with self.session_scope() as session:

            serialised_events = self._event_query(
                session,
                sender_id,
                fetch_events_from_all_sessions=fetch_events_from_all_sessions,
            ).all()

            events = [json.loads(event.data) for event in serialised_events]

            if self.domain and len(events) > 0:
                logger.debug(f"Recreating tracker from sender id '{sender_id}'")
                return DialogueStateTracker.from_dict(
                    sender_id, events, self.domain.slots
                )
            else:
                logger.debug(
                    f"Can't retrieve tracker matching "
                    f"sender id '{sender_id}' from SQL storage. "
                    f"Returning `None` instead."
                )
                return None

    def _event_query(
        self, session: "Session", sender_id: Text, fetch_events_from_all_sessions: bool
    ) -> "Query":
        """Provide the query to retrieve the conversation events for a specific sender.
        The events are ordered by ID to ensure correct sequence of events.
        As `timestamp` is not guaranteed to be unique and low-precision (float), it
        cannot be used to order the events.

        Args:
            session: Current database session.
            sender_id: Sender id whose conversation events should be retrieved.
            fetch_events_from_all_sessions: Whether to fetch events from all
                conversation sessions. If `False`, only fetch events from the
                latest conversation session.

        Returns:
            Query to get the conversation events.
        """
        # Subquery to find the timestamp of the latest `SessionStarted` event
        session_start_sub_query = (
            session.query(sa.func.max(self.SQLEvent.timestamp).label("session_start"))
            .filter(
                self.SQLEvent.sender_id == sender_id,
                self.SQLEvent.type_name == SessionStarted.type_name,
            )
            .subquery()
        )

        event_query = session.query(self.SQLEvent).filter(
            self.SQLEvent.sender_id == sender_id
        )
        if not fetch_events_from_all_sessions:
            event_query = event_query.filter(
                # Find events after the latest `SessionStarted` event or return all
                # events
                sa.or_(
                    self.SQLEvent.timestamp >= session_start_sub_query.c.session_start,
                    session_start_sub_query.c.session_start.is_(None),
                )
            )

        return event_query.order_by(self.SQLEvent.id)

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Update database with events from the current conversation."""
        await self.stream_events(tracker)

        with self.session_scope() as session:
            # only store recent events
            events = self._additional_events(session, tracker)

            for event in events:
                data = event.as_dict()
                intent = (
                    data.get("parse_data", {}).get("intent", {}).get(INTENT_NAME_KEY)
                )
                action = data.get("name")
                timestamp = data.get("timestamp")

                # noinspection PyArgumentList
                session.add(
                    self.SQLEvent(
                        sender_id=tracker.sender_id,
                        type_name=event.type_name,
                        timestamp=timestamp,
                        intent_name=intent,
                        action_name=action,
                        data=json.dumps(data),
                    )
                )
            session.commit()

        logger.debug(f"Tracker with sender_id '{tracker.sender_id}' stored to database")

    def _additional_events(
        self, session: "Session", tracker: DialogueStateTracker
    ) -> Iterator:
        """Return events from the tracker which aren't currently stored."""
        number_of_events_since_last_session = self._event_query(
            session, tracker.sender_id, fetch_events_from_all_sessions=False
        ).count()

        return itertools.islice(
            tracker.events, number_of_events_since_last_session, len(tracker.events)
        )


class FailSafeTrackerStore(TrackerStore):
    """Tracker store wrapper.

    Allows a fallback to a different tracker store in case of errors.
    """

    def __init__(
        self,
        tracker_store: TrackerStore,
        on_tracker_store_error: Optional[Callable[[Exception], None]] = None,
        fallback_tracker_store: Optional[TrackerStore] = None,
    ) -> None:
        """Create a `FailSafeTrackerStore`.

        Args:
            tracker_store: Primary tracker store.
            on_tracker_store_error: Callback which is called when there is an error
                in the primary tracker store.
            fallback_tracker_store: Fallback tracker store.
        """
        self._fallback_tracker_store: Optional[TrackerStore] = fallback_tracker_store
        self._tracker_store = tracker_store
        self._on_tracker_store_error = on_tracker_store_error

        super().__init__(tracker_store.domain, tracker_store.event_broker)

    @property
    def domain(self) -> Domain:
        """Returns the domain of the primary tracker store."""
        return self._tracker_store.domain

    @domain.setter
    def domain(self, domain: Domain) -> None:
        self._tracker_store.domain = domain

        if self._fallback_tracker_store:
            self._fallback_tracker_store.domain = domain

    @property
    def fallback_tracker_store(self) -> TrackerStore:
        """Returns the fallback tracker store."""
        if not self._fallback_tracker_store:
            self._fallback_tracker_store = InMemoryTrackerStore(
                self._tracker_store.domain, self._tracker_store.event_broker
            )

        return self._fallback_tracker_store

    def on_tracker_store_error(self, error: Exception) -> None:
        """Calls the callback when there is an error in the primary tracker store."""
        if self._on_tracker_store_error:
            self._on_tracker_store_error(error)
        else:
            logger.error(
                f"Error happened when trying to save conversation tracker to "
                f"'{self._tracker_store.__class__.__name__}'. Falling back to use "
                f"the '{InMemoryTrackerStore.__name__}'. Please "
                f"investigate the following error: {error}."
            )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Calls `retrieve` method of primary tracker store."""
        try:
            return await self._tracker_store.retrieve(sender_id)
        except Exception as e:
            self.on_tracker_store_retrieve_error(e)
            return None

    async def keys(self) -> Iterable[Text]:
        """Calls `keys` method of primary tracker store."""
        try:
            return await self._tracker_store.keys()
        except Exception as e:
            self.on_tracker_store_error(e)
            return []

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Calls `save` method of primary tracker store."""
        try:
            await self._tracker_store.save(tracker)
        except Exception as e:
            self.on_tracker_store_error(e)
            await self.fallback_tracker_store.save(tracker)

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Calls `retrieve_full_tracker` method of primary tracker store.

        Args:
            sender_id: The sender id of the tracker to retrieve.
        """
        try:
            return await self._tracker_store.retrieve_full_tracker(sender_id)
        except Exception as e:
            self.on_tracker_store_retrieve_error(e)
            return None

    def on_tracker_store_retrieve_error(self, error: Exception) -> None:
        """Calls `_on_tracker_store_error` callable attribute if set.

        Otherwise, logs the error.

        Args:
            error: The error that occurred.
        """
        if self._on_tracker_store_error:
            self._on_tracker_store_error(error)
        else:
            logger.error(
                f"Error happened when trying to retrieve conversation tracker from "
                f"'{self._tracker_store.__class__.__name__}'. Falling back to use "
                f"the '{InMemoryTrackerStore.__name__}'. Please "
                f"investigate the following error: {error}."
            )


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
    domain: Optional[Domain] = None,
    event_broker: Optional[EventBroker] = None,
) -> TrackerStore:
    """Given an endpoint configuration, create a proper tracker store object."""
    domain = domain or Domain.empty()

    if endpoint_config is None or endpoint_config.type is None:
        # default tracker store if no type is set
        tracker_store: TrackerStore = InMemoryTrackerStore(domain, event_broker)
    elif endpoint_config.type.lower() == "redis":
        tracker_store = RedisTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "mongod":
        tracker_store = MongoTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "sql":
        tracker_store = SQLTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "dynamo":
        tracker_store = DynamoTrackerStore(
            domain=domain, event_broker=event_broker, **endpoint_config.kwargs
        )
    else:
        tracker_store = _load_from_module_name_in_endpoint_config(
            domain, endpoint_config, event_broker
        )

    logger.debug(f"Connected to {tracker_store.__class__.__name__}.")

    return tracker_store


def _load_from_module_name_in_endpoint_config(
    domain: Domain, store: EndpointConfig, event_broker: Optional[EventBroker] = None
) -> TrackerStore:
    """Initializes a custom tracker.

    Defaults to the InMemoryTrackerStore if the module path can not be found.

    Args:
        domain: defines the universe in which the assistant operates
        store: the specific tracker store
        event_broker: an event broker to publish events

    Returns:
        a tracker store from a specified type in a stores endpoint configuration
    """
    try:
        tracker_store_class = rasa.shared.utils.common.class_from_module_path(
            store.type
        )

        return tracker_store_class(
            host=store.url, domain=domain, event_broker=event_broker, **store.kwargs
        )
    except (AttributeError, ImportError):
        rasa.shared.utils.io.raise_warning(
            f"Tracker store with type '{store.type}' not found. "
            f"Using `InMemoryTrackerStore` instead."
        )
        return InMemoryTrackerStore(domain)


def create_tracker_store(
    endpoint_config: Optional[EndpointConfig],
    domain: Optional[Domain] = None,
    event_broker: Optional[EventBroker] = None,
) -> TrackerStore:
    """Creates a tracker store based on the current configuration."""
    tracker_store = _create_from_endpoint_config(endpoint_config, domain, event_broker)

    if not check_if_tracker_store_async(tracker_store):
        rasa.shared.utils.io.raise_deprecation_warning(
            f"Tracker store implementation "
            f"{tracker_store.__class__.__name__} "
            f"is not asynchronous. Non-asynchronous tracker stores "
            f"are currently deprecated and will be removed in 4.0. "
            f"Please make the following methods async: "
            f"{_get_async_tracker_store_methods()}"
        )
        tracker_store = AwaitableTrackerStore(tracker_store)

    return tracker_store


class AwaitableTrackerStore(TrackerStore):
    """Wraps a tracker store so it can be implemented with async overrides."""

    def __init__(
        self,
        tracker_store: TrackerStore,
    ) -> None:
        """Create a `AwaitableTrackerStore`.

        Args:
            tracker_store: the wrapped tracker store.
        """
        self._tracker_store = tracker_store

        super().__init__(tracker_store.domain, tracker_store.event_broker)

    @property
    def domain(self) -> Domain:
        """Returns the domain of the primary tracker store."""
        return self._tracker_store.domain

    @domain.setter
    def domain(self, domain: Optional[Domain]) -> None:
        """Setter method to modify the wrapped tracker store's domain field."""
        self._tracker_store.domain = domain or Domain.empty()

    @staticmethod
    def create(
        obj: Union[TrackerStore, EndpointConfig, None],
        domain: Optional[Domain] = None,
        event_broker: Optional[EventBroker] = None,
    ) -> TrackerStore:
        """Wrapper to call `create` method of primary tracker store."""
        if isinstance(obj, TrackerStore):
            return AwaitableTrackerStore(obj)
        elif isinstance(obj, EndpointConfig):
            return AwaitableTrackerStore(_create_from_endpoint_config(obj))
        else:
            raise ValueError(
                f"{type(obj).__name__} supplied "
                f"but expected object of type {TrackerStore.__name__} or "
                f"of type {EndpointConfig.__name__}."
            )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Wrapper to call `retrieve` method of primary tracker store."""
        result = self._tracker_store.retrieve(sender_id)
        return (
            await result
            if isawaitable(result)
            else result  # type: ignore[return-value]
        )

    async def keys(self) -> Iterable[Text]:
        """Wrapper to call `keys` method of primary tracker store."""
        result = self._tracker_store.keys()
        return await result if isawaitable(result) else result

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Wrapper to call `save` method of primary tracker store."""
        result = self._tracker_store.save(tracker)
        return await result if isawaitable(result) else result

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Wrapper to call `retrieve_full_tracker` method of primary tracker store."""
        result = self._tracker_store.retrieve_full_tracker(conversation_id)
        return (
            await result
            if isawaitable(result)
            else result  # type: ignore[return-value]
        )
