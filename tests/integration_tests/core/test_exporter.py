import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from pytest import MonkeyPatch

from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.exporter import Exporter
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.mark.broker
async def test_exporter_publishes_to_kafka_broker_success(
    tmp_path: Path,
) -> None:
    tracker_store = InMemoryTrackerStore(domain=Domain.empty())
    tracker = DialogueStateTracker.from_events(
        "test_export",
        [
            ActionExecuted("action_listen"),
        ],
    )

    await tracker_store.save(tracker)

    kafka_broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
    )

    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
        event_broker:
          type: kafka
          topic: rasa
          url: localhost:9092
          client_id: kafka-python-rasa
          partition_by_sender: true
          security_protocol: SASL_PLAINTEXT
          sasl_username: admin
          sasl_password: password
          sasl_mechanism: PLAIN
        """
        )
    )

    exporter = Exporter(tracker_store, kafka_broker, str(endpoints_file))

    published_events = await exporter.publish_events()
    assert published_events == 1


@pytest.mark.broker
async def test_exporter_publishes_to_kafka_broker_fail(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    tracker_store = InMemoryTrackerStore(domain=Domain.empty())
    tracker = DialogueStateTracker.from_events(
        "test_export",
        [
            ActionExecuted("action_listen"),
        ],
    )

    await tracker_store.save(tracker)

    kafka_broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
    )

    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
        event_broker:
          type: kafka
          topic: rasa
          url: localhost:9092
          client_id: kafka-python-rasa
          partition_by_sender: true
          security_protocol: SASL_PLAINTEXT
          sasl_username: admin
          sasl_password: password
          sasl_mechanism: PLAIN
        """
        )
    )

    exporter = Exporter(tracker_store, kafka_broker, str(endpoints_file))

    # patch the exporter to raise an exception when publishing events
    monkeypatch.setattr(exporter, "publish_events", Mock(side_effect=Exception))

    with pytest.raises(Exception) as error:
        await exporter.publish_events()
        assert "Producer terminating with 1 messages" in str(error.value)
        assert (
            "still in queue or transit: use flush() to wait for "
            "outstanding message delivery" in str(error.value)
        )
    # necessary for producer teardown
    await kafka_broker.close()
