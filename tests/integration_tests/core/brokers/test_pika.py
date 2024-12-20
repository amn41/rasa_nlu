from typing import Text

import docker
import pytest
import randomname
from pytest import LogCaptureFixture
from structlog.testing import capture_logs

from rasa.core.brokers.pika import PikaEventBroker, RABBITMQ_EXCHANGE
from .conftest import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_DEFAULT_QUEUE,
)


@pytest.mark.broker
async def test_pika_event_broker_connect():
    broker = PikaEventBroker(
        host=RABBITMQ_HOST,
        username=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
        port=RABBITMQ_PORT,
        queues=[RABBITMQ_DEFAULT_QUEUE],
    )
    try:
        await broker.connect()
        assert broker.is_ready()
    finally:
        await broker.close()


@pytest.mark.broker
@pytest.mark.xdist_group("rabbitmq")
async def test_pika_event_broker_publish_after_restart(
    docker_client: docker.DockerClient,
    caplog: LogCaptureFixture,
    rabbitmq_username: Text,
    rabbitmq_password: Text,
) -> None:
    environment = {
        "RABBITMQ_DEFAULT_USER": rabbitmq_username,
        "RABBITMQ_DEFAULT_PASS": rabbitmq_password,
    }

    rabbitmq_container = docker_client.containers.run(
        image="healthcheck/rabbitmq",
        detach=True,
        environment=environment,
        name="rabbitmq",
        ports={f"{RABBITMQ_PORT}/tcp": RABBITMQ_PORT},
    )
    rabbitmq_container.reload()
    assert rabbitmq_container.status == "running"

    broker = PikaEventBroker(
        host=RABBITMQ_HOST,
        username=rabbitmq_username,
        password=rabbitmq_password,
        port=RABBITMQ_PORT,
        queues=[RABBITMQ_DEFAULT_QUEUE],
        should_keep_unpublished_messages=True,
    )

    await broker.connect()
    assert broker.is_ready()
    broker.publish({"event": "test"})

    rabbitmq_container.stop()
    rabbitmq_container.reload()

    assert rabbitmq_container.status == "exited"

    with capture_logs() as cap_logs:
        event = {"event": "test_while_closed"}
        await broker._publish(event)

        assert cap_logs[-1]["log_level"] == "error"
        assert "pika.events.publish" in cap_logs[-1]["event"]
        assert cap_logs[-1]["rasa_event"] == event

    # reconnect with the same broker
    rabbitmq_container.restart()
    rabbitmq_container.reload()
    assert rabbitmq_container.status == "running"

    with capture_logs() as cap_logs:
        await broker.connect()
        assert broker.is_ready()

        after_restart_event = {"event": "test_after_restart"}
        await broker._publish(after_restart_event)

        assert cap_logs[-1]["log_level"] == "debug"
        assert "pika.events.publish" in cap_logs[-1]["event"]
        assert cap_logs[-1]["rasa_event"] == after_restart_event
        assert cap_logs[-1]["rabbitmq_exchange"] == RABBITMQ_EXCHANGE

    await broker.close()

    rabbitmq_container.stop()
    rabbitmq_container.remove()


@pytest.mark.broker
@pytest.mark.xdist_group("rabbitmq")
@pytest.mark.parametrize("host_component", ["localhost", "myuser:mypassword@localhost"])
async def test_pika_event_broker_connect_with_path_and_query_params_in_url(
    docker_client: docker.DockerClient,
    host_component: Text,
) -> None:
    username = "myuser"
    password = "mypassword"
    vhost = "myvhost"
    hostname = "my-rabbitmq"

    environment = {
        "RABBITMQ_DEFAULT_USER": username,
        "RABBITMQ_DEFAULT_PASS": password,
        "RABBITMQ_DEFAULT_VHOST": vhost,
    }

    rabbitmq_container = docker_client.containers.run(
        image="rabbitmq:3-management",
        detach=True,
        environment=environment,
        name=f"rabbitmq_{randomname.generate(5)}",
        hostname=hostname,
        ports={f"{RABBITMQ_PORT}/tcp": RABBITMQ_PORT, "15672/tcp": 15672},
    )
    rabbitmq_container.reload()
    assert rabbitmq_container.status == "running"

    query_param = "heartbeat=5"

    broker = PikaEventBroker(
        host=f"amqp://{host_component}/{vhost}?{query_param}",
        username=username,
        password=password,
        port=RABBITMQ_PORT,
        queues=[RABBITMQ_DEFAULT_QUEUE],
    )

    try:
        await broker.connect()
        assert broker.is_ready()
    finally:
        await broker.close()

        rabbitmq_container.stop()
        rabbitmq_container.remove()
