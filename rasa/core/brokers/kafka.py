import asyncio
import os
import json
import logging
import structlog
import threading
from asyncio import AbstractEventLoop
from typing import Any, Text, List, Optional, Union, Dict, TYPE_CHECKING
import time

from rasa.core.brokers.broker import EventBroker
from rasa.core.exceptions import KafkaProducerInitializationError
from rasa.shared.utils.io import DEFAULT_ENCODING
from rasa.utils.endpoints import EndpointConfig
import rasa.shared.utils.common

if TYPE_CHECKING:
    from confluent_kafka import KafkaError, Producer, Message

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


class KafkaEventBroker(EventBroker):
    """Kafka event broker."""

    def __init__(
        self,
        url: Union[Text, List[Text], None],
        topic: Text = "rasa_core_events",
        client_id: Optional[Text] = None,
        partition_by_sender: bool = False,
        sasl_username: Optional[Text] = None,
        sasl_password: Optional[Text] = None,
        sasl_mechanism: Optional[Text] = "PLAIN",
        ssl_cafile: Optional[Text] = None,
        ssl_certfile: Optional[Text] = None,
        ssl_keyfile: Optional[Text] = None,
        ssl_check_hostname: bool = False,
        security_protocol: Text = "SASL_PLAINTEXT",
        **kwargs: Any,
    ) -> None:
        """Kafka event broker.

        Args:
            url: 'url[:port]' string (or list of 'url[:port]'
                strings) that the producer should contact to bootstrap initial
                cluster metadata. This does not have to be the full node list.
                It just needs to have at least one broker that will respond to a
                Metadata API Request.
            topic: Topics to subscribe to.
            client_id: A name for this client. This string is passed in each request
                to servers and can be used to identify specific server-side log entries
                that correspond to this client. Also submitted to `GroupCoordinator` for
                logging with respect to producer group administration.
            partition_by_sender: Flag to configure whether messages are partitioned by
                sender_id or not
            sasl_username: Username for plain authentication.
            sasl_password: Password for plain authentication.
            sasl_mechanism: Authentication mechanism when security_protocol is
                configured for SASL_PLAINTEXT or SASL_SSL.
                Valid values are: PLAIN, GSSAPI, OAUTHBEARER, SCRAM-SHA-256,
                SCRAM-SHA-512. Default: `PLAIN`
            ssl_cafile: Optional filename of ca file to use in certificate
                verification.
            ssl_certfile: Optional filename of file in pem format containing
                the client certificate, as well as any ca certificates needed to
                establish the certificate's authenticity.
            ssl_keyfile: Optional filename containing the client private key.
            ssl_check_hostname: Flag to configure whether ssl handshake
                should verify that the certificate matches the broker's hostname.
            security_protocol: Protocol used to communicate with brokers.
                Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
        """
        self.producer: Optional[Producer] = None
        self.url = url
        self.topic = topic
        self.client_id = client_id
        self.partition_by_sender = partition_by_sender
        self.security_protocol = security_protocol.upper()
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.sasl_mechanism = sasl_mechanism
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.queue_size = kwargs.get("queue_size")
        self.ssl_check_hostname = "https" if ssl_check_hostname else None

        # Async producer implementation followed from confluent-kafka asyncio example:
        # https://github.com/confluentinc/confluent-kafka-python/blob/master/examples/asyncio_example.py#L88  # noqa: E501
        self._loop = asyncio.get_event_loop()
        self._cancelled = False
        self._poll_thread = threading.Thread(target=self._poll_loop)
        self._poll_thread.start()

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> Optional["KafkaEventBroker"]:
        """Creates broker. See the parent class for more information."""
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(
        self,
        event: Dict[Text, Any],
        retries: int = 60,
        retry_delay_in_seconds: float = 5,
    ) -> None:
        """Publishes events."""
        from confluent_kafka import KafkaException

        if retries == 1:
            retries = 2

        if self.producer is None:
            self.producer = self._create_producer()
            try:
                self._check_kafka_connection()
                logger.debug("Connection to kafka successful.")
            except KafkaException:
                logger.debug("Failed to connect kafka.")
                return
        while retries:
            try:
                self._publish(event)
                return
            except BufferError as e:
                logger.error(
                    f"Could not publish message to kafka url '{self.url}'. "
                    f"Failed with error: {e}"
                )
                self.producer.poll(1)
                retries -= 1
            except Exception as e:
                logger.error(
                    f"Could not publish message to kafka url '{self.url}'. "
                    f"Failed with error: {e}"
                )
                try:
                    self._check_kafka_connection()
                except KafkaException:
                    logger.debug("Connection to kafka lost, reconnecting...")
                    self.producer = self._create_producer()
                    try:
                        self._check_kafka_connection()
                        logger.debug("Reconnection to kafka successful")
                        self._publish(event)
                        return
                    except KafkaException:
                        pass
                retries -= 1
                time.sleep(retry_delay_in_seconds)

        logger.error("Failed to publish Kafka event.")

    def _check_kafka_connection(self) -> None:
        """Verifies connection with Kafka.

        Raises:
            KafkaException: if Kafka is disconnected.
        """
        if self.producer is not None:
            self.producer.list_topics(timeout=5)

    def _get_kafka_config(self) -> Dict[Text, Any]:
        config = {
            "client.id": self.client_id,
            "bootstrap.servers": self.url,
            "error_cb": kafka_error_callback,
        }
        if self.queue_size:
            config["queue.buffering.max.messages"] = self.queue_size

        if self.security_protocol == "PLAINTEXT":
            authentication_params: Dict[Text, Any] = {
                "security.protocol": self.security_protocol,
            }
        elif self.security_protocol == "SASL_PLAINTEXT":
            authentication_params = {
                "sasl.username": self.sasl_username,
                "sasl.password": self.sasl_password,
                "sasl.mechanism": self.sasl_mechanism,
                "security.protocol": self.security_protocol,
            }
        elif self.security_protocol == "SSL":
            authentication_params = {
                "ssl.ca.location": self.ssl_cafile,
                "ssl.certificate.location": self.ssl_certfile,
                "ssl.key.location": self.ssl_keyfile,
                "security.protocol": self.security_protocol,
            }
        elif self.security_protocol == "SASL_SSL":
            authentication_params = {
                "sasl.username": self.sasl_username,
                "sasl.password": self.sasl_password,
                "ssl.ca.location": self.ssl_cafile,
                "ssl.certificate.location": self.ssl_certfile,
                "ssl.key.location": self.ssl_keyfile,
                "ssl.endpoint.identification.algorithm": self.ssl_check_hostname,
                "security.protocol": self.security_protocol,
                "sasl.mechanism": self.sasl_mechanism,
            }
        else:
            raise ValueError(
                f"Cannot initialise `KafkaEventBroker`: "
                f"Invalid `security_protocol` ('{self.security_protocol}')."
            )

        return {**config, **authentication_params}

    def _create_producer(self) -> "Producer":
        import confluent_kafka

        try:
            return confluent_kafka.Producer(self._get_kafka_config())
        except confluent_kafka.KafkaException as e:
            raise KafkaProducerInitializationError(
                f"Cannot initialise `KafkaEventBroker`: {e}"
            )

    def _publish(self, event: Dict[Text, Any]) -> None:
        if self.partition_by_sender:
            partition_key = bytes(event.get("sender_id"), encoding=DEFAULT_ENCODING)
        else:
            partition_key = None

        headers = []
        if self.rasa_environment:
            headers = [
                (
                    "RASA_ENVIRONMENT",
                    bytes(self.rasa_environment, encoding=DEFAULT_ENCODING),
                )
            ]

        reduced_event = rasa.shared.core.events.remove_parse_data(event)
        structlogger.debug(
            "kafka.publish.event",
            event_info="Logging a reduced version of the Kafka event",
            topic=self.topic,
            rasa_event=reduced_event,
            partition_key=partition_key,
            headers=headers,
        )

        serialized_event = json.dumps(event).encode(DEFAULT_ENCODING)

        if self.producer is not None:
            self.producer.produce(
                self.topic,
                value=serialized_event,
                key=partition_key,
                headers=headers,
                on_delivery=delivery_report,
            )

    async def close(self) -> None:
        self._cancelled = True
        self._poll_thread.join()
        if self.producer:
            self.producer.flush()

    @rasa.shared.utils.common.lazy_property
    def rasa_environment(self) -> Optional[Text]:
        """Get value of the `RASA_ENVIRONMENT` environment variable."""
        return os.environ.get("RASA_ENVIRONMENT", "RASA_ENVIRONMENT_NOT_SET")

    def _poll_loop(self) -> None:
        """Polls the producer for events.

        Required to trigger the on_delivery callback passed to produce method.
        """
        if self.producer is not None:
            while not self._cancelled:
                self.producer.poll(0.1)


def kafka_error_callback(err: "KafkaError") -> None:
    """Callback for Kafka errors.

    Any exception raised from this callback will be re-raised from the
    triggering flush() call.
    """
    from confluent_kafka import KafkaException, KafkaError

    # handle authentication / connection related issues, likely pointing
    # to a configuration error
    if (
        err.code() == KafkaError._ALL_BROKERS_DOWN
        or err.code() == KafkaError._AUTHENTICATION
        or err.code() == KafkaError._MAX_POLL_EXCEEDED
    ):
        raise KafkaException(err)
    else:
        logger.warning("A KafkaError has been raised.", exc_info=True)


def delivery_report(err: Exception, msg: "Message") -> None:
    """Reports the failure or success of a message delivery.

    Args:
        err (KafkaError): The error that occurred on None on success.
        msg (Message): The message that was produced or failed.
    """
    if err is not None:
        logger.error(f"Delivery failed for User record {msg.key()}: {err}")
        return

    logger.info(
        f"User record {msg.key()} successfully produced to "
        f"{msg.topic()} [{msg.partition()}] at offset {msg.offset()}."
    )
