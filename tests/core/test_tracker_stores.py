import logging
import warnings
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import fakeredis
import pytest
import sqlalchemy
import uuid

from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from moto import mock_dynamodb
from pymongo.errors import OperationFailure

from rasa.core.agent import Agent
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.constants import DEFAULT_SENDER_ID
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from sqlalchemy.dialects.oracle.base import OracleDialect
from sqlalchemy.engine.url import URL
from typing import Any, Tuple, Text, Type, Dict, List, Union, Optional, ContextManager
from unittest.mock import MagicMock, Mock

import rasa.core.tracker_store
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
)
from rasa.core.constants import POSTGRESQL_SCHEMA
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    SlotSet,
    ActionExecuted,
    Restarted,
    UserUttered,
    SessionStarted,
    BotUttered,
    Event,
)
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.core.tracker_store import (
    TrackerStore,
    InMemoryTrackerStore,
    RedisTrackerStore,
    DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX,
    SQLTrackerStore,
    DynamoTrackerStore,
    FailSafeTrackerStore,
    AwaitableTrackerStore,
)
from rasa.shared.core.trackers import DialogueStateTracker, TrackerEventDiffEngine
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.conftest import AsyncMock
from tests.core.conftest import MockedMongoTrackerStore

test_domain = Domain.load("data/test_domains/default.yml")


async def get_or_create_tracker_store(store: TrackerStore) -> None:
    slot_key = "location"
    slot_val = "Easter Island"

    tracker = await store.get_or_create_tracker(DEFAULT_SENDER_ID)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)
    assert tracker.get_slot(slot_key) == slot_val

    await store.save(tracker)

    again = await store.get_or_create_tracker(DEFAULT_SENDER_ID)
    assert again.get_slot(slot_key) == slot_val


def test_get_or_create():
    get_or_create_tracker_store(InMemoryTrackerStore(test_domain))


# noinspection PyPep8Naming
@mock_dynamodb
def test_dynamo_get_or_create():
    get_or_create_tracker_store(DynamoTrackerStore(test_domain))


@mock_dynamodb
async def test_dynamo_tracker_floats():
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)
    tracker = await tracker_store.get_or_create_tracker(
        conversation_id, append_action_listen=False
    )

    # save `slot` event with known `float`-type timestamp
    timestamp = 13423.23434623
    tracker.update(SlotSet("key", "val", timestamp=timestamp))
    await tracker_store.save(tracker)

    # retrieve tracker and the event timestamp is retrieved as a `float`
    tracker = await tracker_store.get_or_create_tracker(conversation_id)
    retrieved_timestamp = tracker.events[0].timestamp
    assert isinstance(retrieved_timestamp, float)
    assert retrieved_timestamp == timestamp


async def test_restart_after_retrieval_from_tracker_store(domain: Domain):
    store = InMemoryTrackerStore(domain)
    tr = await store.get_or_create_tracker("myuser")
    synth = [ActionExecuted("action_listen") for _ in range(4)]

    for e in synth:
        tr.update(e)

    tr.update(Restarted())
    latest_restart = tr.idx_after_latest_restart()

    await store.save(tr)
    tr2 = await store.retrieve("myuser")
    latest_restart_after_loading = tr2.idx_after_latest_restart()
    assert latest_restart == latest_restart_after_loading


async def test_tracker_store_remembers_max_history(domain: Domain):
    store = InMemoryTrackerStore(domain)
    tr = await store.get_or_create_tracker("myuser", max_event_history=42)
    tr.update(Restarted())

    await store.save(tr)
    tr2 = await store.retrieve("myuser")
    assert tr._max_event_history == tr2._max_event_history == 42


def test_tracker_store_endpoint_config_loading(endpoints_path: Text):
    cfg = read_endpoint_config(endpoints_path, "tracker_store")

    assert cfg == EndpointConfig.from_dict(
        {
            "type": "redis",
            "url": "localhost",
            "port": 6379,
            "db": 0,
            "username": "username",
            "password": "password",
            "timeout": 30000,
            "use_ssl": True,
            "ssl_keyfile": "keyfile.key",
            "ssl_certfile": "certfile.crt",
            "ssl_ca_certs": "my-bundle.ca-bundle",
        }
    )


def test_create_tracker_store_from_endpoint_config(
    domain: Domain, endpoints_path: Text
):
    store = read_endpoint_config(endpoints_path, "tracker_store")
    tracker_store = RedisTrackerStore(
        domain=domain,
        host="localhost",
        port=6379,
        db=0,
        username="username",
        password="password",
        record_exp=3000,
        use_ssl=True,
        ssl_keyfile="keyfile.key",
        ssl_certfile="certfile.crt",
        ssl_ca_certs="my-bundle.ca-bundle",
    )

    assert isinstance(tracker_store, type(TrackerStore.create(store, domain)))


def test_redis_tracker_store_invalid_key_prefix(domain: Domain):

    test_invalid_key_prefix = "$$ &!"

    tracker_store = RedisTrackerStore(
        domain=domain,
        host="localhost",
        port=6379,
        db=0,
        password="password",
        key_prefix=test_invalid_key_prefix,
        record_exp=3000,
    )

    assert tracker_store._get_key_prefix() == DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX


def test_redis_tracker_store_valid_key_prefix(domain: Domain):
    test_valid_key_prefix = "spanish"

    tracker_store = RedisTrackerStore(
        domain=domain,
        host="localhost",
        port=6379,
        db=0,
        password="password",
        key_prefix=test_valid_key_prefix,
        record_exp=3000,
    )

    assert (
        tracker_store._get_key_prefix()
        == f"{test_valid_key_prefix}:{DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX}"
    )


def test_exception_tracker_store_from_endpoint_config(
    domain: Domain, monkeypatch: MonkeyPatch, endpoints_path: Text
):
    """Check if tracker store properly handles exceptions.

    If we can not create a tracker store by instantiating the
    expected type (e.g. due to an exception) we should fallback to
    the default `InMemoryTrackerStore`."""

    store = read_endpoint_config(endpoints_path, "tracker_store")
    mock = Mock(side_effect=Exception("test exception"))
    monkeypatch.setattr(rasa.core.tracker_store, "RedisTrackerStore", mock)

    with pytest.raises(Exception) as e:
        TrackerStore.create(store, domain)

    assert "test exception" in str(e.value)


def test_raise_connection_exception_redis_tracker_store_creation(
    domain: Domain, monkeypatch: MonkeyPatch, endpoints_path: Text
):
    store = read_endpoint_config(endpoints_path, "tracker_store")
    monkeypatch.setattr(
        rasa.core.tracker_store,
        "RedisTrackerStore",
        Mock(side_effect=ConnectionError()),
    )

    with pytest.raises(ConnectionException):
        TrackerStore.create(store, domain)


def test_mongo_tracker_store_raise_exception(domain: Domain, monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        rasa.core.tracker_store,
        "MongoTrackerStore",
        Mock(
            side_effect=OperationFailure("not authorized on logs to execute command.")
        ),
    )
    with pytest.raises(ConnectionException) as error:
        TrackerStore.create(
            EndpointConfig(username="username", password="password", type="mongod"),
            domain,
        )

    assert "not authorized on logs to execute command." in str(error.value)


class HostExampleTrackerStore(RedisTrackerStore):
    pass


class NonAsyncTrackerStore(TrackerStore):
    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        pass

    def save(self, tracker: DialogueStateTracker) -> None:
        pass


def test_tracker_store_with_host_argument_from_string(domain: Domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "tests.core.test_tracker_stores.HostExampleTrackerStore"

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("error")
        tracker_store = TrackerStore.create(store_config, domain)

    assert len(record) == 0

    assert isinstance(tracker_store, HostExampleTrackerStore)


def test_tracker_store_from_invalid_module(domain: Domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "a.module.which.cannot.be.found"

    with pytest.warns(UserWarning):
        tracker_store = TrackerStore.create(store_config, domain)

    assert isinstance(tracker_store, InMemoryTrackerStore)


def test_tracker_store_from_invalid_string(domain: Domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "any string"

    with pytest.warns(UserWarning):
        tracker_store = TrackerStore.create(store_config, domain)

    assert isinstance(tracker_store, InMemoryTrackerStore)


async def _tracker_store_and_tracker_with_slot_set() -> Tuple[
    InMemoryTrackerStore, DialogueStateTracker
]:
    # returns an InMemoryTrackerStore containing a tracker with a slot set

    slot_key = "cuisine"
    slot_val = "French"

    store = InMemoryTrackerStore(test_domain)
    tracker = await store.get_or_create_tracker(DEFAULT_SENDER_ID)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)

    return store, tracker


async def test_tracker_serialisation():
    store, tracker = await _tracker_store_and_tracker_with_slot_set()
    serialised = store.serialise_tracker(tracker)

    assert tracker == store.deserialise_tracker(DEFAULT_SENDER_ID, serialised)


@pytest.mark.parametrize(
    "full_url",
    [
        "postgresql://localhost",
        "postgresql://localhost:5432",
        "postgresql://user:secret@localhost",
        "sqlite:///",
    ],
)
def test_get_db_url_with_fully_specified_url(full_url: Text):
    assert SQLTrackerStore.get_db_url(host=full_url) == full_url


def test_get_db_url_with_port_in_host():
    host = "localhost:1234"
    dialect = "postgresql"
    db = "mydb"

    expected = f"{dialect}://{host}/{db}"

    assert (
        str(SQLTrackerStore.get_db_url(dialect=dialect, host=host, db=db)) == expected
    )


def test_db_get_url_with_sqlite():
    expected = "sqlite:///rasa.db"
    assert str(SQLTrackerStore.get_db_url(dialect="sqlite", db="rasa.db")) == expected


def test_get_db_url_with_correct_host():
    expected = "postgresql://localhost:5005/mydb"

    assert (
        str(
            SQLTrackerStore.get_db_url(
                dialect="postgresql", host="localhost", port=5005, db="mydb"
            )
        )
        == expected
    )


def test_get_db_url_with_query():
    expected = "postgresql://localhost:5005/mydb?driver=my-driver"

    assert (
        str(
            SQLTrackerStore.get_db_url(
                dialect="postgresql",
                host="localhost",
                port=5005,
                db="mydb",
                query={"driver": "my-driver"},
            )
        )
        == expected
    )


def test_sql_tracker_store_logs_do_not_show_password(caplog: LogCaptureFixture):
    dialect = "postgresql"
    host = "localhost"
    port = 9901
    db = "some-database"
    username = "db-user"
    password = "some-password"

    with caplog.at_level(logging.DEBUG):
        _ = SQLTrackerStore(None, dialect, host, port, db, username, password)

    # the URL in the logs does not contain the password
    assert password not in caplog.text

    # instead the password is displayed as '***'
    assert f"postgresql://{username}:***@{host}:{port}/{db}" in caplog.text


def test_db_url_with_query_from_endpoint_config(tmp_path: Path):
    endpoint_config = """
    tracker_store:
      dialect: postgresql
      url: localhost
      port: 5123
      username: user
      password: pw
      login_db: login-db
      query:
        driver: my-driver
        another: query
    """
    f = tmp_path / "tmp_config_file.yml"
    f.write_text(endpoint_config)
    store_config = read_endpoint_config(str(f), "tracker_store")

    url = SQLTrackerStore.get_db_url(**store_config.kwargs)

    import itertools

    # order of query dictionary in yaml is random, test against both permutations
    connection_url = "postgresql://user:pw@:5123/login-db?"
    assert any(
        str(url) == connection_url + "&".join(permutation)
        for permutation in (
            itertools.permutations(("another=query", "driver=my-driver"))
        )
    )


async def test_fail_safe_tracker_store_if_no_errors():
    mocked_tracker_store = Mock()

    tracker_store = FailSafeTrackerStore(mocked_tracker_store, None)

    # test save
    mocked_tracker_store.save = AsyncMock()
    await tracker_store.save(None)
    mocked_tracker_store.save.assert_called_once()

    # test retrieve
    expected = [1]
    mocked_tracker_store.retrieve = AsyncMock(return_value=expected)
    sender_id = "10"
    assert await tracker_store.retrieve(sender_id) == expected
    mocked_tracker_store.retrieve.assert_called_once_with(sender_id)

    # test keys
    expected = ["sender 1", "sender 2"]
    mocked_tracker_store.keys = AsyncMock(return_value=expected)
    assert await tracker_store.keys() == expected
    mocked_tracker_store.keys.assert_called_once()


async def test_fail_safe_tracker_store_with_save_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.save = Mock(side_effect=Exception())

    fallback_tracker_store = Mock()
    fallback_tracker_store.save = AsyncMock()

    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(
        mocked_tracker_store, on_error_callback, fallback_tracker_store
    )
    await tracker_store.save(None)

    fallback_tracker_store.save.assert_called_once()
    on_error_callback.assert_called_once()


async def test_fail_safe_tracker_store_with_keys_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.keys = Mock(side_effect=Exception())

    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(mocked_tracker_store, on_error_callback)
    assert await tracker_store.keys() == []
    on_error_callback.assert_called_once()


async def test_fail_safe_tracker_store_with_retrieve_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.retrieve = Mock(side_effect=Exception())

    fallback_tracker_store = Mock()
    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(
        mocked_tracker_store, on_error_callback, fallback_tracker_store
    )

    assert await tracker_store.retrieve("sender_id") is None
    on_error_callback.assert_called_once()


def test_set_fail_safe_tracker_store_domain(domain: Domain):
    tracker_store = InMemoryTrackerStore(domain)
    fallback_tracker_store = InMemoryTrackerStore(None)
    failsafe_store = FailSafeTrackerStore(tracker_store, None, fallback_tracker_store)

    failsafe_store.domain = domain
    assert failsafe_store.domain is domain
    assert tracker_store.domain is failsafe_store.domain
    assert fallback_tracker_store.domain is failsafe_store.domain


async def create_tracker_with_partially_saved_events(
    tracker_store: TrackerStore,
) -> Tuple[List[Event], DialogueStateTracker]:
    # creates a tracker with two events and saved it to the tracker store
    # following that, it adds three more events that are not saved to the tracker store
    sender_id = uuid.uuid4().hex

    # create tracker with two events and save it
    events = [UserUttered("hello"), BotUttered("what")]
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # add more events to the tracker, do not yet save it
    events = [ActionExecuted(ACTION_LISTEN_NAME), UserUttered("123"), BotUttered("yes")]
    for event in events:
        tracker.update(event)

    return events, tracker


async def _saved_tracker_with_multiple_session_starts(
    tracker_store: TrackerStore, sender_id: Text
) -> DialogueStateTracker:
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
        ],
    )

    await tracker_store.save(tracker)
    return await tracker_store.retrieve(sender_id)


async def test_mongo_additional_events(domain: Domain):
    tracker_store = MockedMongoTrackerStore(domain)
    events, tracker = await create_tracker_with_partially_saved_events(tracker_store)

    # make sure only new events are returned
    # noinspection PyProtectedMember
    assert list(tracker_store._additional_events(tracker)) == events


async def test_mongo_additional_events_with_session_start(domain: Domain):
    sender = "test_mongo_additional_events_with_session_start"
    tracker_store = MockedMongoTrackerStore(domain)
    tracker = await _saved_tracker_with_multiple_session_starts(tracker_store, sender)

    tracker.update(UserUttered("hi2"))

    # noinspection PyProtectedMember
    additional_events = list(tracker_store._additional_events(tracker))

    assert len(additional_events) == 1
    assert isinstance(additional_events[0], UserUttered)


# we cannot parametrise over this and the previous test due to the different ways of
# calling _additional_events()
async def test_sql_additional_events(domain: Domain):
    tracker_store = SQLTrackerStore(domain)
    additional_events, tracker = await create_tracker_with_partially_saved_events(
        tracker_store
    )

    # make sure only new events are returned
    with tracker_store.session_scope() as session:
        # noinspection PyProtectedMember
        assert (
            list(tracker_store._additional_events(session, tracker))
            == additional_events
        )


async def test_sql_additional_events_with_session_start(domain: Domain):
    sender = "test_sql_additional_events_with_session_start"
    tracker_store = SQLTrackerStore(domain)
    tracker = await _saved_tracker_with_multiple_session_starts(tracker_store, sender)

    tracker.update(UserUttered("hi2"), domain)

    # make sure only new events are returned
    with tracker_store.session_scope() as session:
        # noinspection PyProtectedMember
        additional_events = list(tracker_store._additional_events(session, tracker))
        assert len(additional_events) == 1
        assert isinstance(additional_events[0], UserUttered)


async def test_tracker_store_retrieve_ordered_by_id(
    domain: Domain,
):
    tracker_store_kwargs = {"host": "sqlite:///"}
    tracker_store = SQLTrackerStore(domain, **tracker_store_kwargs)
    events = [
        SessionStarted(timestamp=1),
        UserUttered("Hola", {"name": "greet"}, timestamp=2),
        BotUttered("Hi", timestamp=2),
        UserUttered("How are you?", {"name": "greet"}, timestamp=2),
        BotUttered("I am good, whats up", timestamp=2),
        UserUttered("Ciao", {"name": "greet"}, timestamp=2),
        BotUttered("Bye", timestamp=2),
    ]
    sender_id = "test_sql_tracker_store_events_order"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    # Retrieve tracker with events since latest SessionStarted
    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 7
    # assert the order of events is same as the order in which they were added
    assert all((event == tracker.events[i] for i, event in enumerate(events)))


@pytest.mark.parametrize(
    "tracker_store_type,tracker_store_kwargs",
    [(MockedMongoTrackerStore, {}), (SQLTrackerStore, {"host": "sqlite:///"})],
)
async def test_tracker_store_retrieve_with_session_started_events(
    tracker_store_type: Type[TrackerStore],
    tracker_store_kwargs: Dict,
    domain: Domain,
):
    tracker_store = tracker_store_type(domain, **tracker_store_kwargs)
    events = [
        UserUttered("Hola", {"name": "greet"}, timestamp=1),
        BotUttered("Hi", timestamp=2),
        SessionStarted(timestamp=3),
        UserUttered("Ciao", {"name": "greet"}, timestamp=4),
    ]
    sender_id = "test_sql_tracker_store_with_session_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    # Retrieve tracker with events since latest SessionStarted
    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 2
    assert all((event == tracker.events[i] for i, event in enumerate(events[2:])))


@pytest.mark.parametrize(
    "tracker_store_type,tracker_store_kwargs",
    [(MockedMongoTrackerStore, {}), (SQLTrackerStore, {"host": "sqlite:///"})],
)
async def test_tracker_store_retrieve_without_session_started_events(
    tracker_store_type: Type[TrackerStore],
    tracker_store_kwargs: Dict,
    domain,
):
    tracker_store = tracker_store_type(domain, **tracker_store_kwargs)

    # Create tracker with a SessionStarted event
    events = [
        UserUttered("Hola", {"name": "greet"}),
        BotUttered("Hi"),
        UserUttered("Ciao", {"name": "greet"}),
        BotUttered("Hi2"),
    ]

    sender_id = "test_sql_tracker_store_retrieve_without_session_started_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 4
    assert all(event == tracker.events[i] for i, event in enumerate(events))


@pytest.mark.parametrize(
    "tracker_store_type,tracker_store_kwargs",
    [
        (MockedMongoTrackerStore, {}),
        (SQLTrackerStore, {"host": "sqlite:///"}),
        (InMemoryTrackerStore, {}),
    ],
)
async def test_tracker_store_retrieve_with_events_from_previous_sessions(
    tracker_store_type: Type[TrackerStore], tracker_store_kwargs: Dict
):
    tracker_store = tracker_store_type(Domain.empty(), **tracker_store_kwargs)

    conversation_id = uuid.uuid4().hex
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
        ],
    )
    await tracker_store.save(tracker)

    actual = await tracker_store.retrieve_full_tracker(conversation_id)

    assert len(actual.events) == len(tracker.events)


def test_session_scope_error(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture, domain: Domain
):
    tracker_store = SQLTrackerStore(domain)
    tracker_store.sessionmaker = Mock()

    requested_schema = uuid.uuid4().hex

    # `ensure_schema_exists()` raises `ValueError`
    mocked_ensure_schema_exists = Mock(side_effect=ValueError(requested_schema))
    monkeypatch.setattr(
        rasa.core.tracker_store, "ensure_schema_exists", mocked_ensure_schema_exists
    )

    # `SystemExit` is triggered by failing `ensure_schema_exists()`
    with pytest.raises(SystemExit):
        with tracker_store.session_scope() as _:
            pass

    # error message is printed
    assert (
        f"Requested PostgreSQL schema '{requested_schema}' was not found in the "
        f"database." in capsys.readouterr()[0]
    )


@pytest.mark.parametrize(
    "url,is_postgres_url",
    [
        (f"{PGDialect.name}://admin:pw@localhost:5432/rasa", True),
        (f"{SQLiteDialect.name}:///", False),
        (URL(PGDialect.name), True),
        (URL(SQLiteDialect.name), False),
    ],
)
def test_is_postgres_url(url: Union[Text, URL], is_postgres_url: bool):
    assert rasa.core.tracker_store.is_postgresql_url(url) == is_postgres_url


def set_or_delete_postgresql_schema_env_var(
    monkeypatch: MonkeyPatch, value: Optional[Text]
) -> None:
    """Set `POSTGRESQL_SCHEMA` environment variable using `MonkeyPatch`.

    Args:
        monkeypatch: Instance of `MonkeyPatch` to use for patching.
        value: Value of the `POSTGRESQL_SCHEMA` environment variable to set.
    """
    if value is None:
        monkeypatch.delenv(POSTGRESQL_SCHEMA, raising=False)
    else:
        monkeypatch.setenv(POSTGRESQL_SCHEMA, value)


@pytest.mark.parametrize(
    "url,schema_env,kwargs",
    [
        # postgres without schema
        (
            f"{PGDialect.name}://admin:pw@localhost:5432/rasa",
            None,
            {
                "pool_size": rasa.core.tracker_store.POSTGRESQL_DEFAULT_POOL_SIZE,
                "max_overflow": rasa.core.tracker_store.POSTGRESQL_DEFAULT_MAX_OVERFLOW,
            },
        ),
        # postgres with schema
        (
            f"{PGDialect.name}://admin:pw@localhost:5432/rasa",
            "schema1",
            {
                "connect_args": {"options": "-csearch_path=schema1"},
                "pool_size": rasa.core.tracker_store.POSTGRESQL_DEFAULT_POOL_SIZE,
                "max_overflow": rasa.core.tracker_store.POSTGRESQL_DEFAULT_MAX_OVERFLOW,
            },
        ),
        # oracle without schema
        (f"{OracleDialect.name}://admin:pw@localhost:5432/rasa", None, {}),
        # oracle with schema
        (f"{OracleDialect.name}://admin:pw@localhost:5432/rasa", "schema1", {}),
        # sqlite
        (f"{SQLiteDialect.name}:///", None, {}),
    ],
)
def test_create_engine_kwargs(
    monkeypatch: MonkeyPatch,
    url: Union[Text, URL],
    schema_env: Optional[Text],
    kwargs: Dict[Text, Dict[Text, Union[Text, int]]],
):
    set_or_delete_postgresql_schema_env_var(monkeypatch, schema_env)

    assert rasa.core.tracker_store.create_engine_kwargs(url) == kwargs


@contextmanager
def does_not_raise():
    """Contextmanager to be used when an expression is not expected to raise an
    exception.

    This contextmanager can be used in parametrized tests, where some input objects
    are expected to raise and others are not.

    Example:

        @pytest.mark.parametrize(
            "a,b,raises_context",
            [
                # 5/6 is a legal divison
                (5, 6, does_not_raise()),
                # 5/0 raises a `ZeroDivisionError`
                (5, 0, pytest.raises(ZeroDivisionError)),
            ],
        )
        def test_divide(
            a: int, b: int, raises_context: ContextManager,
        ):
            with raises_context:
                _ = a / b

    """
    yield


@pytest.mark.parametrize(
    "is_postgres,schema_env,schema_exists,raises_context",
    [
        (True, "schema1", True, does_not_raise()),
        (True, "schema1", False, pytest.raises(ValueError)),
        (False, "schema1", False, does_not_raise()),
        (True, None, False, does_not_raise()),
        (False, None, False, does_not_raise()),
    ],
)
def test_ensure_schema_exists(
    monkeypatch: MonkeyPatch,
    is_postgres: bool,
    schema_env: Optional[Text],
    schema_exists: bool,
    raises_context: ContextManager,
):
    set_or_delete_postgresql_schema_env_var(monkeypatch, schema_env)
    monkeypatch.setattr(
        rasa.core.tracker_store, "is_postgresql_url", lambda _: is_postgres
    )
    monkeypatch.setattr(sqlalchemy, "exists", Mock())

    # mock the `session.query().scalar()` query which returns whether the schema
    # exists in the db
    scalar = Mock(return_value=schema_exists)
    query = Mock(scalar=scalar)
    session = Mock()
    session.query = Mock(return_value=query)

    with raises_context:
        rasa.core.tracker_store.ensure_schema_exists(session)


def test_current_state_without_events(domain: Domain):
    tracker_store = MockedMongoTrackerStore(domain)

    # insert some events
    events = [
        UserUttered("Hola", {"name": "greet"}),
        BotUttered("Hi"),
        UserUttered("Ciao", {"name": "greet"}),
        BotUttered("Hi2"),
    ]

    sender_id = "test_mongo_tracker_store_current_state_without_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)

    # get current state without events
    # noinspection PyProtectedMember
    state = tracker_store._current_tracker_state_without_events(tracker)

    # `events` key should not be in there
    assert state and "events" not in state


def test_login_db_with_no_postgresql(tmp_path: Path):
    with pytest.warns(UserWarning):
        SQLTrackerStore(db=str(tmp_path / "rasa.db"), login_db=str(tmp_path / "other"))


@pytest.mark.parametrize(
    "config",
    [
        {
            "type": "mongod",
            "url": "mongodb://0.0.0.0:42/?serverSelectionTimeoutMS=5000",
        },
        {"type": "dynamo"},
    ],
)
def test_tracker_store_connection_error(config: Dict, domain: Domain):
    store = EndpointConfig.from_dict(config)

    with pytest.raises(ConnectionException):
        TrackerStore.create(store, domain)


async def prepare_token_serialisation(
    tracker_store: TrackerStore, response_selector_agent: Agent, sender_id: Text
):
    text = "Good morning"
    tokenizer = WhitespaceTokenizer(WhitespaceTokenizer.get_default_config())
    tokens = tokenizer.tokenize(Message(data={"text": text}), "text")
    indices = [[t.start, t.end] for t in tokens]

    tracker = await tracker_store.get_or_create_tracker(sender_id=sender_id)
    parse_data = await response_selector_agent.parse_message(text)
    event = UserUttered(
        "Good morning",
        parse_data.get("intent"),
        parse_data.get("entities", []),
        parse_data,
    )

    tracker.update(event)
    await tracker_store.save(tracker)

    retrieved_tracker = await tracker_store.retrieve(sender_id=sender_id)
    event = retrieved_tracker.get_last_event_for(event_type=UserUttered)
    event_tokens = event.as_dict().get("parse_data").get("text_tokens")

    assert event_tokens == indices


def test_inmemory_tracker_store_with_token_serialisation(
    domain: Domain, default_agent: Agent
):
    tracker_store = InMemoryTrackerStore(domain)
    prepare_token_serialisation(tracker_store, default_agent, "inmemory")


def test_mongo_tracker_store_with_token_serialisation(
    domain: Domain, response_selector_agent: Agent
):
    tracker_store = MockedMongoTrackerStore(domain)
    prepare_token_serialisation(tracker_store, response_selector_agent, "mongo")


def test_sql_tracker_store_with_token_serialisation(
    domain: Domain, response_selector_agent: Agent
):
    tracker_store = SQLTrackerStore(domain, **{"host": "sqlite:///"})
    prepare_token_serialisation(tracker_store, response_selector_agent, "sql")


def test_sql_tracker_store_creation_with_invalid_port(domain: Domain):
    with pytest.raises(RasaException) as error:
        TrackerStore.create(
            EndpointConfig(port="$DB_PORT", type="sql"),
            domain,
        )
    assert "port '$DB_PORT' cannot be cast to integer." in str(error.value)


def test_create_non_async_tracker_store(domain: Domain):
    endpoint_config = EndpointConfig(
        type="tests.core.test_tracker_stores.NonAsyncTrackerStore"
    )
    with pytest.warns(FutureWarning):
        tracker_store = TrackerStore.create(endpoint_config)
    assert isinstance(tracker_store, AwaitableTrackerStore)
    assert isinstance(tracker_store._tracker_store, NonAsyncTrackerStore)


def test_create_awaitable_tracker_store_with_endpoint_config():
    endpoint_config = EndpointConfig(
        type="tests.core.test_tracker_stores.NonAsyncTrackerStore"
    )
    tracker_store = AwaitableTrackerStore.create(endpoint_config)

    assert isinstance(tracker_store, AwaitableTrackerStore)
    assert isinstance(tracker_store._tracker_store, NonAsyncTrackerStore)


@pytest.mark.parametrize(
    "endpoints_file, expected_type",
    [
        (None, InMemoryTrackerStore),
        ("data/test_endpoints/endpoints_sql.yml", SQLTrackerStore),
        ("data/test_endpoints/endpoints_redis.yml", RedisTrackerStore),
    ],
)
def test_create_tracker_store_from_endpoints_file(
    endpoints_file: Optional[Text], expected_type: Any, domain: Domain
) -> None:
    endpoint_config = read_endpoint_config(endpoints_file, "tracker_store")
    tracker_store = rasa.core.tracker_store.create_tracker_store(
        endpoint_config, domain
    )

    assert rasa.core.tracker_store.check_if_tracker_store_async(tracker_store) is True
    assert isinstance(tracker_store, expected_type)


async def test_fail_safe_tracker_store_retrieve_full_tracker(
    domain: Domain, tracker_with_restarted_event: DialogueStateTracker
) -> None:
    primary_tracker_store = SQLTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    tracker_store = FailSafeTrackerStore(primary_tracker_store)
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_fail_safe_tracker_store_retrieve_full_tracker_with_exception(
    caplog: LogCaptureFixture,
) -> None:
    primary_tracker_store = MagicMock()
    primary_tracker_store.domain = Domain.empty()
    primary_tracker_store.event_broker = None

    exception = Exception("Something went wrong")
    primary_tracker_store.retrieve_full_tracker = AsyncMock(side_effect=exception)

    tracker_store = FailSafeTrackerStore(primary_tracker_store)
    with caplog.at_level(logging.ERROR):
        await tracker_store.retrieve_full_tracker("some_id")

    assert "Error happened when trying to retrieve conversation tracker" in caplog.text
    assert f"Please investigate the following error: {exception}." in caplog.text


async def test_sql_get_or_create_full_tracker_without_action_listen() -> None:
    tracker_store = SQLTrackerStore(Domain.empty())
    sender_id = uuid.uuid4().hex
    tracker = await tracker_store.get_or_create_full_tracker(
        sender_id=sender_id, append_action_listen=False
    )
    assert tracker.sender_id == sender_id
    assert tracker.events == deque()


async def test_sql_get_or_create_full_tracker_with_action_listen() -> None:
    tracker_store = SQLTrackerStore(Domain.empty())
    sender_id = uuid.uuid4().hex
    tracker = await tracker_store.get_or_create_full_tracker(
        sender_id=sender_id, append_action_listen=True
    )
    assert tracker.sender_id == sender_id
    assert tracker.events == deque([ActionExecuted(ACTION_LISTEN_NAME)])


async def test_sql_get_or_create_full_tracker_with_existing_tracker(
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    tracker_store = SQLTrackerStore(Domain.empty())
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.get_or_create_full_tracker(
        sender_id=sender_id, append_action_listen=False
    )
    assert tracker.sender_id == sender_id
    assert tracker.events == deque(tracker_with_restarted_event.events)


async def test_sql_tracker_store_retrieve_full_tracker(
    domain: Domain, tracker_with_restarted_event: DialogueStateTracker
) -> None:
    tracker_store = SQLTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_sql_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = SQLTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the SQLTrackerStore filters
    # only the events after `session_started` event
    assert list(tracker.events) == events_after_restart[1:]


async def test_in_memory_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = InMemoryTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_in_memory_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = InMemoryTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)
    assert list(tracker.events) == events_after_restart


async def test_mongo_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = MockedMongoTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_mongo_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = MockedMongoTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the MongoTrackerStore filters
    # only the events after `session_started` event
    assert list(tracker.events) == events_after_restart[1:]


class MockedRedisTrackerStore(RedisTrackerStore):
    def __init__(
        self,
        domain: Domain,
    ) -> None:
        self.red = fakeredis.FakeStrictRedis()
        self.key_prefix = DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX
        self.record_exp = None
        super(RedisTrackerStore, self).__init__(domain, None)


async def test_redis_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = MockedRedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_redis_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = MockedRedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)
    assert list(tracker.events) == events_after_restart


async def test_redis_tracker_store_merge_trackers_same_session() -> None:
    start_session_sequence = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
    events: List[Event] = start_session_sequence + [UserUttered("hello")]
    prior_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )

    events += [BotUttered("Hey! How can I help you?")]

    new_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )

    actual_tracker = RedisTrackerStore._merge_trackers(prior_tracker, new_tracker)

    assert actual_tracker == new_tracker


def test_redis_tracker_store_merge_trackers_overlapping_session() -> None:
    prior_tracker_events: List[Event] = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=1),
        SessionStarted(timestamp=2),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=3),
        UserUttered("hello", timestamp=4),
        BotUttered("Hey! How can I help you?", timestamp=5),
        UserUttered("/restart", timestamp=6),
        ActionExecuted(ACTION_RESTART_NAME, timestamp=7),
    ]

    new_start_session = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=8),
        SessionStarted(timestamp=9),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=10),
    ]

    prior_tracker_events += new_start_session
    prior_tracker = DialogueStateTracker.from_events(
        "overlapping-session",
        evts=prior_tracker_events,
    )

    after_restart_event = [UserUttered("hi again", timestamp=11)]
    new_tracker_events = new_start_session + after_restart_event

    new_tracker = DialogueStateTracker.from_events(
        "overlapping-session",
        evts=new_tracker_events,
    )

    actual_tracker = RedisTrackerStore._merge_trackers(prior_tracker, new_tracker)

    expected_events = prior_tracker_events + after_restart_event

    assert list(actual_tracker.events) == expected_events


def test_redis_tracker_store_merge_trackers_different_session() -> None:
    prior_tracker_events: List[Event] = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=1),
        SessionStarted(timestamp=2),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=3),
        UserUttered("hello", timestamp=4),
        BotUttered("Hey! How can I help you?", timestamp=5),
    ]
    prior_tracker = DialogueStateTracker.from_events(
        "different-session",
        evts=prior_tracker_events,
    )

    new_session = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=8),
        SessionStarted(timestamp=9),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=10),
        UserUttered("I need help.", timestamp=11),
    ]

    new_tracker = DialogueStateTracker.from_events(
        "different-session",
        evts=new_session,
    )

    actual_tracker = RedisTrackerStore._merge_trackers(prior_tracker, new_tracker)

    expected_events = prior_tracker_events + new_session
    assert list(actual_tracker.events) == expected_events


async def test_tracker_event_diff_engine_event_difference() -> None:
    start_session_sequence = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
    events: List[Event] = start_session_sequence + [UserUttered("hello")]
    prior_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )
    new_events = [BotUttered("Hey! How can I help you?")]
    events += new_events

    new_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )

    event_diff = TrackerEventDiffEngine.event_difference(prior_tracker, new_tracker)

    assert new_events == event_diff
