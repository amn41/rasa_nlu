import pytest

from rasa.core.brokers.kafka import KafkaEventBroker
from pytest import LogCaptureFixture
import logging.config


@pytest.mark.broker
async def test_kafka_event_broker_valid():
    broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
    )

    try:
        broker.publish(
            {"sender_id": "valid_test", "event": "user", "text": "hello world!"},
            retries=5,
        )
        assert broker.producer.poll() == 1
    finally:
        await broker.close()


@pytest.mark.broker
async def test_kafka_event_broker_buffer_error_is_handled(caplog: LogCaptureFixture):
    broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
        queue_size=1,
    )

    event_count = 100
    try:
        for i in range(event_count):
            with caplog.at_level(logging.DEBUG):
                broker.publish(
                    {
                        "sender_id": "valid_test",
                        "event": "user",
                        "text": "hello world!",
                    },
                    retries=5,
                )
        assert "Queue full" in caplog.text
        assert broker.producer.poll() == 1
    finally:
        await broker.close()
