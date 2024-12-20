from __future__ import annotations
from asyncio import AbstractEventLoop, CancelledError
import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Text, Union
import uuid

import aiohttp
from aiohttp import ClientError

from rasa.core import jobs
from rasa.core.channels.channel import OutputChannel, UserMessage
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.core.http_interpreter import RasaNLUHttpInterpreter
from rasa.shared.core.domain import Domain
from rasa.core.exceptions import AgentNotReady
from rasa.shared.constants import DEFAULT_SENDER_ID
from rasa.core.lock_store import InMemoryLockStore, LockStore
from rasa.core.nlg import NaturalLanguageGenerator, TemplatedNaturalLanguageGenerator
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import FailSafeTrackerStore, InMemoryTrackerStore
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.exceptions import ModelNotFound
from rasa.nlu.utils import is_url
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name
from rasa.utils.endpoints import EndpointConfig

from rasa.core.tracker_store import TrackerStore
from rasa.core.utils import AvailableEndpoints

logger = logging.getLogger(__name__)


async def load_from_server(agent: Agent, model_server: EndpointConfig) -> Agent:
    """Load a persisted model from a server."""
    # We are going to pull the model once first, and then schedule a recurring
    # job. the benefit of this approach is that we can be sure that there
    # is a model after this function completes -> allows to do proper
    # "is alive" check on a startup server's `/status` endpoint. If the server
    # is started, we can be sure that it also already loaded (or tried to)
    # a model.
    await _update_model_from_server(model_server, agent)

    wait_time_between_pulls = model_server.kwargs.get("wait_time_between_pulls", 100)

    if wait_time_between_pulls:
        # continuously pull the model every `wait_time_between_pulls` seconds
        await _schedule_model_pulling(model_server, int(wait_time_between_pulls), agent)

    return agent


def _load_and_set_updated_model(
    agent: Agent, model_directory: Text, fingerprint: Text
) -> None:
    """Load the persisted model into memory and set the model on the agent.

    Args:
        agent: Instance of `Agent` to update with the new model.
        model_directory: Rasa model directory.
        fingerprint: Fingerprint of the supplied model at `model_directory`.
    """
    logger.debug(f"Found new model with fingerprint {fingerprint}. Loading...")
    agent.load_model(model_directory, fingerprint)

    logger.debug("Finished updating agent to new model.")


async def _update_model_from_server(model_server: EndpointConfig, agent: Agent) -> None:
    """Load a zipped Rasa Core model from a URL and update the passed agent."""
    if not is_url(model_server.url):
        raise aiohttp.InvalidURL(model_server.url)

    with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:
        try:
            new_fingerprint = await _pull_model_and_fingerprint(
                model_server, agent.fingerprint, temporary_directory
            )

            if new_fingerprint:
                _load_and_set_updated_model(agent, temporary_directory, new_fingerprint)
            else:
                logger.debug(f"No new model found at URL {model_server.url}")
        except Exception:  # skipcq: PYL-W0703
            # TODO: Make this exception more specific, possibly print different log
            # for each one.
            logger.exception(
                "Failed to update model. The previous model will stay loaded instead."
            )


async def _pull_model_and_fingerprint(
    model_server: EndpointConfig, fingerprint: Optional[Text], model_directory: Text
) -> Optional[Text]:
    """Queries the model server.

    Args:
        model_server: Model server endpoint information.
        fingerprint: Current model fingerprint.
        model_directory: Directory where to download model to.

    Returns:
        Value of the response's <ETag> header which contains the model
        hash. Returns `None` if no new model is found.
    """
    headers = {"If-None-Match": fingerprint}

    logger.debug(f"Requesting model from server {model_server.url}...")

    async with model_server.session() as session:
        try:
            params = model_server.combine_parameters()
            async with session.request(
                "GET",
                model_server.url,
                timeout=DEFAULT_REQUEST_TIMEOUT,
                headers=headers,
                params=params,
            ) as resp:

                if resp.status in [204, 304]:
                    logger.debug(
                        "Model server returned {} status code, "
                        "indicating that no new model is available. "
                        "Current fingerprint: {}"
                        "".format(resp.status, fingerprint)
                    )
                    return None
                elif resp.status == 404:
                    logger.debug(
                        "Model server could not find a model at the requested "
                        "endpoint '{}'. It's possible that no model has been "
                        "trained, or that the requested tag hasn't been "
                        "assigned.".format(model_server.url)
                    )
                    return None
                elif resp.status != 200:
                    logger.debug(
                        "Tried to fetch model from server, but server response "
                        "status code is {}. We'll retry later..."
                        "".format(resp.status)
                    )
                    return None

                model_path = Path(model_directory) / resp.headers.get(
                    "filename", "model.tar.gz"
                )
                with open(model_path, "wb") as file:
                    file.write(await resp.read())

                logger.debug("Saved model to '{}'".format(os.path.abspath(model_path)))

                # return the new fingerprint
                return resp.headers.get("ETag")

        except aiohttp.ClientError as e:
            logger.debug(
                "Tried to fetch model from server, but "
                "couldn't reach server. We'll retry later... "
                "Error: {}.".format(e)
            )
            return None


async def _run_model_pulling_worker(model_server: EndpointConfig, agent: Agent) -> None:
    # noinspection PyBroadException
    try:
        await _update_model_from_server(model_server, agent)
    except CancelledError:
        logger.warning("Stopping model pulling (cancelled).")
    except ClientError:
        logger.exception(
            "An exception was raised while fetching a model. Continuing anyways..."
        )


async def _schedule_model_pulling(
    model_server: EndpointConfig, wait_time_between_pulls: int, agent: Agent
) -> None:
    (await jobs.scheduler()).add_job(
        _run_model_pulling_worker,
        "interval",
        seconds=wait_time_between_pulls,
        args=[model_server, agent],
        id="pull-model-from-server",
        replace_existing=True,
    )


async def load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    loop: Optional[AbstractEventLoop] = None,
) -> Agent:
    """Loads agent from server, remote storage or disk.

    Args:
        model_path: Path to the model if it's on disk.
        model_server: Configuration for a potential server which serves the model.
        remote_storage: URL of remote storage for model.
        endpoints: Endpoint configuration.
        loop: Optional async loop to pass to broker creation.

    Returns:
        The instantiated `Agent` or `None`.
    """
    from rasa.core.tracker_store import TrackerStore
    from rasa.core.brokers.broker import EventBroker

    tracker_store = None
    lock_store = None
    generator = None
    action_endpoint = None
    http_interpreter = None

    if endpoints:
        broker = await EventBroker.create(endpoints.event_broker, loop=loop)
        tracker_store = TrackerStore.create(
            endpoints.tracker_store, event_broker=broker
        )
        lock_store = LockStore.create(endpoints.lock_store)
        generator = endpoints.nlg
        action_endpoint = endpoints.action
        model_server = endpoints.model if endpoints.model else model_server
        if endpoints.nlu:
            http_interpreter = RasaNLUHttpInterpreter(endpoints.nlu)

    agent = Agent(
        generator=generator,
        tracker_store=tracker_store,
        lock_store=lock_store,
        action_endpoint=action_endpoint,
        model_server=model_server,
        remote_storage=remote_storage,
        http_interpreter=http_interpreter,
    )

    try:
        if model_server is not None:
            return await load_from_server(agent, model_server)

        elif remote_storage is not None:
            agent.load_model_from_remote_storage(model_path)

        elif model_path is not None and os.path.exists(model_path):
            try:
                agent.load_model(model_path)
            except ModelNotFound:
                rasa.shared.utils.io.raise_warning(
                    f"No valid model found at {model_path}!"
                )
        else:
            rasa.shared.utils.io.raise_warning(
                "No valid configuration given to load agent. "
                "Agent loaded with no model!"
            )
        return agent

    except Exception as e:
        logger.error(f"Could not load model due to {e}.", exc_info=True)
        return agent


def agent_must_be_ready(f: Callable[..., Any]) -> Callable[..., Any]:
    """Any Agent method decorated with this will raise if the agent is not ready."""

    @functools.wraps(f)
    def decorated(self: Agent, *args: Any, **kwargs: Any) -> Any:
        if not self.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. You need to set a "
                "processor and a tracker store."
            )
        return f(self, *args, **kwargs)

    return decorated


class Agent:
    """The Agent class provides an interface for the most important Rasa functionality.

    This includes training, handling messages, loading a dialogue model,
    getting the next action, and handling a channel.
    """

    def __init__(
        self,
        domain: Optional[Domain] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        fingerprint: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        http_interpreter: Optional[RasaNLUHttpInterpreter] = None,
    ):
        """Initializes an `Agent`."""
        self.domain = domain
        self.processor: Optional[MessageProcessor] = None

        self.nlg = NaturalLanguageGenerator.create(generator, self.domain)
        self.tracker_store = self._create_tracker_store(tracker_store, self.domain)
        self.lock_store = self._create_lock_store(lock_store)
        self.action_endpoint = action_endpoint
        self.http_interpreter = http_interpreter

        self._set_fingerprint(fingerprint)
        self.model_server = model_server
        self.remote_storage = remote_storage

    @classmethod
    def load(
        cls,
        model_path: Union[Text, Path],
        domain: Optional[Domain] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        fingerprint: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        http_interpreter: Optional[RasaNLUHttpInterpreter] = None,
    ) -> Agent:
        """Constructs a new agent and loads the processer and model."""
        agent = Agent(
            domain=domain,
            generator=generator,
            tracker_store=tracker_store,
            lock_store=lock_store,
            action_endpoint=action_endpoint,
            fingerprint=fingerprint,
            model_server=model_server,
            remote_storage=remote_storage,
            http_interpreter=http_interpreter,
        )
        agent.load_model(model_path=model_path, fingerprint=fingerprint)
        return agent

    def load_model(
        self, model_path: Union[Text, Path], fingerprint: Optional[Text] = None
    ) -> None:
        """Loads the agent's model and processor given a new model path."""
        self.processor = MessageProcessor(
            model_path=model_path,
            tracker_store=self.tracker_store,
            lock_store=self.lock_store,
            action_endpoint=self.action_endpoint,
            generator=self.nlg,
            http_interpreter=self.http_interpreter,
        )
        self.domain = self.processor.domain

        self._set_fingerprint(fingerprint)

        # update domain on all instances
        self.tracker_store.domain = self.domain
        if isinstance(self.nlg, TemplatedNaturalLanguageGenerator):
            self.nlg.responses = self.domain.responses if self.domain else {}

    @property
    def model_id(self) -> Optional[Text]:
        """Returns the model_id from processor's model_metadata."""
        return self.processor.model_metadata.model_id if self.processor else None

    @property
    def model_name(self) -> Optional[Text]:
        """Returns the model name from processor's model_path."""
        return self.processor.model_path.name if self.processor else None

    def is_ready(self) -> bool:
        """Check if all necessary components are instantiated to use agent."""
        return self.tracker_store is not None and self.processor is not None

    @agent_must_be_ready
    async def parse_message(self, message_data: Text) -> Dict[Text, Any]:
        """Handles message text and intent payload input messages.

        The return value of this function is parsed_data.

        Args:
            message_data (Text): Contain the received message in text or\
            intent payload format.

        Returns:
            The parsed message.

        Example:
                {\
                    "text": '/greet{"name":"Rasa"}',\
                    "intent": {"name": "greet", "confidence": 1.0},\
                    "intent_ranking": [{"name": "greet", "confidence": 1.0}],\
                    "entities": [{"entity": "name", "start": 6,\
                                  "end": 21, "value": "Rasa"}],\
                }

        """
        message = UserMessage(message_data)

        return await self.processor.parse_message(message)  # type: ignore[union-attr]

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message."""
        if not self.is_ready():
            logger.info("Ignoring message as there is no agent to handle it.")
            return None

        async with self.lock_store.lock(message.sender_id):
            return await self.processor.handle_message(  # type: ignore[union-attr]
                message
            )

    @agent_must_be_ready
    async def predict_next_for_sender_id(
        self, sender_id: Text
    ) -> Optional[Dict[Text, Any]]:
        """Predict the next action for a sender id."""
        return await self.processor.predict_next_for_sender_id(  # type: ignore[union-attr] # noqa:E501
            sender_id
        )

    @agent_must_be_ready
    def predict_next_with_tracker(
        self,
        tracker: DialogueStateTracker,
        verbosity: EventVerbosity = EventVerbosity.AFTER_RESTART,
    ) -> Optional[Dict[Text, Any]]:
        """Predicts the next action."""
        return self.processor.predict_next_with_tracker(  # type: ignore[union-attr]
            tracker, verbosity
        )

    @agent_must_be_ready
    async def log_message(self, message: UserMessage) -> DialogueStateTracker:
        """Append a message to a dialogue - does not predict actions."""
        return await self.processor.log_message(message)  # type: ignore[union-attr]

    @agent_must_be_ready
    async def execute_action(
        self,
        sender_id: Text,
        action: Text,
        output_channel: OutputChannel,
        policy: Optional[Text],
        confidence: Optional[float],
    ) -> Optional[DialogueStateTracker]:
        """Executes an action."""
        prediction = PolicyPrediction.for_action_name(
            self.domain, action, policy, confidence or 0.0
        )
        return await self.processor.execute_action(  # type: ignore[union-attr]
            sender_id, action, output_channel, self.nlg, prediction
        )

    @agent_must_be_ready
    async def trigger_intent(
        self,
        intent_name: Text,
        entities: List[Dict[Text, Any]],
        output_channel: OutputChannel,
        tracker: DialogueStateTracker,
    ) -> None:
        """Trigger a user intent, e.g. triggered by an external event."""
        await self.processor.trigger_external_user_uttered(  # type: ignore[union-attr]
            intent_name, entities, tracker, output_channel
        )

    @agent_must_be_ready
    async def handle_text(
        self,
        text_message: Union[Text, Dict[Text, Any]],
        output_channel: Optional[OutputChannel] = None,
        sender_id: Optional[Text] = DEFAULT_SENDER_ID,
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message.

        If a message preprocessor is passed, the message will be passed to that
        function first and the return value is then used as the
        input for the dialogue engine.

        The return value of this function depends on the ``output_channel``. If
        the output channel is not set, set to ``None``, or set
        to ``CollectingOutputChannel`` this function will return the messages
        the bot wants to respond.

        :Example:

            >>> from rasa.core.agent import Agent
            >>> agent = Agent.load("examples/moodbot/models")
            >>> await agent.handle_text("hello")
            [u'how can I help you?']

        """
        if isinstance(text_message, str):
            text_message = {"text": text_message}

        msg = UserMessage(text_message.get("text"), output_channel, sender_id)

        return await self.handle_message(msg)

    def _set_fingerprint(self, fingerprint: Optional[Text] = None) -> None:

        if fingerprint:
            self.fingerprint = fingerprint
        else:
            self.fingerprint = uuid.uuid4().hex

    @staticmethod
    def _create_tracker_store(
        store: Optional[TrackerStore], domain: Domain
    ) -> TrackerStore:
        if store is not None:
            store.domain = domain
            tracker_store = store
        else:
            tracker_store = InMemoryTrackerStore(domain)

        return FailSafeTrackerStore(tracker_store)

    @staticmethod
    def _create_lock_store(store: Optional[LockStore]) -> LockStore:
        if store is not None:
            return store

        return InMemoryLockStore()

    def load_model_from_remote_storage(self, model_name: Text) -> None:
        """Loads an Agent from remote storage."""
        from rasa.nlu.persistor import get_persistor

        persistor = get_persistor(self.remote_storage)

        if persistor is not None:
            with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:
                persistor.retrieve(model_name, temporary_directory)
                self.load_model(temporary_directory)

        else:
            raise RasaException(
                f"Persistor not found for remote storage: '{self.remote_storage}'."
            )
