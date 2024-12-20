import copy
import logging
import structlog
import os
from pathlib import Path
import tarfile
import time
from types import LambdaType
from typing import Any, Dict, List, Optional, Text, Tuple, Union

from rasa.core.http_interpreter import RasaNLUHttpInterpreter
from rasa.engine import loader
from rasa.engine.constants import PLACEHOLDER_MESSAGE, PLACEHOLDER_TRACKER
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelMetadata
from rasa.model import get_latest_model
from rasa.plugin import plugin_manager
from rasa.shared.data import TrainingType
import rasa.shared.utils.io
import rasa.core.actions.action
from rasa.core import jobs
from rasa.core.actions.action import Action
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    OutputChannel,
    UserMessage,
)
import rasa.core.utils
from rasa.core.policies.policy import PolicyPrediction
from rasa.engine.runner.interface import GraphRunner
from rasa.exceptions import ActionLimitReached, ModelNotFound
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    FOLLOWUP_ACTION,
    SESSION_START_METADATA_SLOT,
    ACTION_EXTRACT_SLOTS,
)
from rasa.shared.core.events import (
    ActionExecutionRejected,
    BotUttered,
    Event,
    ReminderCancelled,
    ReminderScheduled,
    SlotSet,
    UserUttered,
    ActionExecuted,
)
from rasa.shared.constants import (
    ASSISTANT_ID_KEY,
    DOCS_URL_DOMAINS,
    DEFAULT_SENDER_ID,
    DOCS_URL_POLICIES,
    UTTER_PREFIX,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.lock_store import LockStore
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name
import rasa.core.tracker_store
import rasa.core.actions.action
import rasa.shared.core.trackers
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.nlu.constants import (
    ENTITIES,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RESPONSE_KEY,
    PREDICTED_CONFIDENCE_KEY,
    FULL_RETRIEVAL_INTENT_NAME_KEY,
    RESPONSE_SELECTOR,
    RESPONSE,
    TEXT,
)
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()

MAX_NUMBER_OF_PREDICTIONS = int(os.environ.get("MAX_NUMBER_OF_PREDICTIONS", "10"))


class MessageProcessor:
    """The message processor is interface for communicating with a bot model."""

    def __init__(
        self,
        model_path: Union[Text, Path],
        tracker_store: rasa.core.tracker_store.TrackerStore,
        lock_store: LockStore,
        generator: NaturalLanguageGenerator,
        action_endpoint: Optional[EndpointConfig] = None,
        max_number_of_predictions: int = MAX_NUMBER_OF_PREDICTIONS,
        on_circuit_break: Optional[LambdaType] = None,
        http_interpreter: Optional[RasaNLUHttpInterpreter] = None,
    ) -> None:
        """Initializes a `MessageProcessor`."""
        self.nlg = generator
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.max_number_of_predictions = max_number_of_predictions
        self.on_circuit_break = on_circuit_break
        self.action_endpoint = action_endpoint
        self.model_filename, self.model_metadata, self.graph_runner = self._load_model(
            model_path
        )

        if self.model_metadata.assistant_id is None:
            rasa.shared.utils.io.raise_warning(
                f"The model metadata does not contain a value for the "
                f"'{ASSISTANT_ID_KEY}' attribute. Check that 'config.yml' "
                f"file contains a value for the '{ASSISTANT_ID_KEY}' key "
                f"and re-train the model. Failure to do so will result in "
                f"streaming events without a unique assistant identifier.",
                UserWarning,
            )

        self.model_path = Path(model_path)
        self.domain = self.model_metadata.domain
        self.http_interpreter = http_interpreter

    @staticmethod
    def _load_model(
        model_path: Union[Text, Path]
    ) -> Tuple[Text, ModelMetadata, GraphRunner]:
        """Unpacks a model from a given path using the graph model loader."""
        try:
            if os.path.isfile(model_path):
                model_tar = model_path
            else:
                model_file_path = get_latest_model(model_path)
                if not model_file_path:
                    raise ModelNotFound(f"No model found at path '{model_path}'.")
                model_tar = model_file_path
        except TypeError:
            raise ModelNotFound(f"Model {model_path} can not be loaded.")

        logger.info(f"Loading model {model_tar}...")
        with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:
            try:
                metadata, runner = loader.load_predict_graph_runner(
                    Path(temporary_directory),
                    Path(model_tar),
                    LocalModelStorage,
                    DaskGraphRunner,
                )
                return os.path.basename(model_tar), metadata, runner
            except tarfile.ReadError:
                raise ModelNotFound(f"Model {model_path} can not be loaded.")

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message with this processor."""
        # preprocess message if necessary
        tracker = await self.log_message(message, should_save_tracker=False)

        if self.model_metadata.training_type == TrainingType.NLU:
            await self.save_tracker(tracker)
            rasa.shared.utils.io.raise_warning(
                "No core model. Skipping action prediction and execution.",
                docs=DOCS_URL_POLICIES,
            )
            return None

        tracker = await self.run_action_extract_slots(message.output_channel, tracker)

        await self._run_prediction_loop(message.output_channel, tracker)

        await self.run_anonymization_pipeline(tracker)

        await self.save_tracker(tracker)

        if isinstance(message.output_channel, CollectingOutputChannel):
            return message.output_channel.messages

        return None

    async def run_action_extract_slots(
        self, output_channel: OutputChannel, tracker: DialogueStateTracker
    ) -> DialogueStateTracker:
        """Run action to extract slots and update the tracker accordingly.

        Args:
            output_channel: Output channel associated with the incoming user message.
            tracker: A tracker representing a conversation state.

        Returns:
            the given (updated) tracker
        """
        action_extract_slots = rasa.core.actions.action.action_for_name_or_text(
            ACTION_EXTRACT_SLOTS, self.domain, self.action_endpoint
        )
        extraction_events = await action_extract_slots.run(
            output_channel, self.nlg, tracker, self.domain
        )

        await self._send_bot_messages(extraction_events, tracker, output_channel)

        tracker.update_with_events(extraction_events, self.domain)

        structlogger.debug(
            "processor.extract.slots",
            action_extract_slot=ACTION_EXTRACT_SLOTS,
            len_extraction_events=len(extraction_events),
            rasa_events=copy.deepcopy(extraction_events),
        )

        return tracker

    async def run_anonymization_pipeline(self, tracker: DialogueStateTracker) -> None:
        """Run the anonymization pipeline on the new tracker events.

        Args:
            tracker: A tracker representing a conversation state.
        """
        anonymization_pipeline = plugin_manager().hook.get_anonymization_pipeline()
        if anonymization_pipeline is None:
            return None

        old_tracker = await self.tracker_store.retrieve(tracker.sender_id)
        new_events = rasa.shared.core.trackers.TrackerEventDiffEngine.event_difference(
            old_tracker, tracker
        )

        for event in new_events:
            body = {"sender_id": tracker.sender_id}
            body.update(event.as_dict())
            anonymization_pipeline.run(body)

    async def predict_next_for_sender_id(
        self, sender_id: Text
    ) -> Optional[Dict[Text, Any]]:
        """Predict the next action for the given sender_id.

        Args:
            sender_id: Conversation ID.

        Returns:
            The prediction for the next action. `None` if no domain or policies loaded.
        """
        tracker = await self.fetch_tracker_and_update_session(sender_id)
        result = self.predict_next_with_tracker(tracker)

        # save tracker state to continue conversation from this state
        await self.save_tracker(tracker)

        return result

    def predict_next_with_tracker(
        self,
        tracker: DialogueStateTracker,
        verbosity: EventVerbosity = EventVerbosity.AFTER_RESTART,
    ) -> Optional[Dict[Text, Any]]:
        """Predict the next action for a given conversation state.

        Args:
            tracker: A tracker representing a conversation state.
            verbosity: Verbosity for the returned conversation state.

        Returns:
            The prediction for the next action. `None` if no domain or policies loaded.
        """
        if self.model_metadata.training_type == TrainingType.NLU:
            rasa.shared.utils.io.raise_warning(
                "No core model. Skipping action prediction and execution.",
                docs=DOCS_URL_POLICIES,
            )
            return None

        prediction = self._predict_next_with_tracker(tracker)

        scores = [
            {"action": a, "score": p}
            for a, p in zip(self.domain.action_names_or_texts, prediction.probabilities)
        ]
        return {
            "scores": scores,
            "policy": prediction.policy_name,
            "confidence": prediction.max_confidence,
            "tracker": tracker.current_state(verbosity),
        }

    async def _update_tracker_session(
        self,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Check the current session in `tracker` and update it if expired.

        An 'action_session_start' is run if the latest tracker session has expired,
        or if the tracker does not yet contain any events (only those after the last
        restart are considered).

        Args:
            metadata: Data sent from client associated with the incoming user message.
            tracker: Tracker to inspect.
            output_channel: Output channel for potential utterances in a custom
                `ActionSessionStart`.
        """
        if not tracker.applied_events() or self._has_session_expired(tracker):
            logger.debug(
                f"Starting a new session for conversation ID '{tracker.sender_id}'."
            )

            action_session_start = self._get_action(ACTION_SESSION_START_NAME)

            if metadata:
                tracker.update(
                    SlotSet(SESSION_START_METADATA_SLOT, metadata), self.domain
                )

            await self._run_action(
                action=action_session_start,
                tracker=tracker,
                output_channel=output_channel,
                nlg=self.nlg,
                prediction=PolicyPrediction.for_action_name(
                    self.domain, ACTION_SESSION_START_NAME
                ),
            )

    async def fetch_tracker_and_update_session(
        self,
        sender_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Fetches tracker for `sender_id` and updates its conversation session.

        If a new tracker is created, `action_session_start` is run.

        Args:
            metadata: Data sent from client associated with the incoming user message.
            output_channel: Output channel associated with the incoming user message.
            sender_id: Conversation ID for which to fetch the tracker.

        Returns:
              Tracker for `sender_id`.
        """
        tracker = await self.get_tracker(sender_id)

        await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def fetch_tracker_with_initial_session(
        self,
        sender_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Fetches tracker for `sender_id` and runs a session start if it's a new
        tracker.

        Args:
            metadata: Data sent from client associated with the incoming user message.
            output_channel: Output channel associated with the incoming user message.
            sender_id: Conversation ID for which to fetch the tracker.

        Returns:
              Tracker for `sender_id`.
        """
        tracker = await self.get_tracker(sender_id)

        # run session start only if the tracker is empty
        if not tracker.events:
            await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def get_tracker(self, conversation_id: Text) -> DialogueStateTracker:
        """Get the tracker for a conversation.

        In contrast to `fetch_tracker_and_update_session` this does not add any
        `action_session_start` or `session_start` events at the beginning of a
        conversation.

        Args:
            conversation_id: The ID of the conversation for which the history should be
                retrieved.

        Returns:
            Tracker for the conversation. Creates an empty tracker in case it's a new
            conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        tracker = await self.tracker_store.get_or_create_tracker(
            conversation_id, append_action_listen=False
        )
        tracker.model_id = self.model_metadata.model_id
        if tracker.assistant_id is None:
            tracker.assistant_id = self.model_metadata.assistant_id
        return tracker

    async def fetch_full_tracker_with_initial_session(
        self,
        conversation_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Get the full tracker for a conversation, including events after a restart.

        Args:
            conversation_id: The ID of the conversation for which the history should be
                retrieved.
            output_channel: Output channel associated with the incoming user message.
            metadata: Data sent from client associated with the incoming user message.

        Returns:
            Tracker for the conversation. Creates an empty tracker with a new session
            initialized in case it's a new conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        tracker = await self.tracker_store.get_or_create_full_tracker(
            conversation_id, False
        )
        tracker.model_id = self.model_metadata.model_id

        if tracker.assistant_id is None:
            tracker.assistant_id = self.model_metadata.assistant_id

        if not tracker.events:
            await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def get_trackers_for_all_conversation_sessions(
        self, conversation_id: Text
    ) -> List[DialogueStateTracker]:
        """Fetches all trackers for a conversation.

        Individual trackers are returned for each conversation session found
        for `conversation_id`.

        Args:
            conversation_id: The ID of the conversation for which the trackers should
                be retrieved.

        Returns:
            Trackers for the conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        tracker = await self.tracker_store.retrieve_full_tracker(conversation_id)

        return rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)

    async def log_message(
        self, message: UserMessage, should_save_tracker: bool = True
    ) -> DialogueStateTracker:
        """Log `message` on tracker belonging to the message's conversation_id.

        Optionally save the tracker if `should_save_tracker` is `True`. Tracker saving
        can be skipped if the tracker returned by this method is used for further
        processing and saved at a later stage.
        """
        tracker = await self.fetch_tracker_and_update_session(
            message.sender_id, message.output_channel, message.metadata
        )

        await self._handle_message_with_tracker(message, tracker)

        if should_save_tracker:
            await self.save_tracker(tracker)

        return tracker

    async def execute_action(
        self,
        sender_id: Text,
        action_name: Text,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        prediction: PolicyPrediction,
    ) -> Optional[DialogueStateTracker]:
        """Execute an action for a conversation.

        Note that this might lead to unexpected bot behavior. Rather use an intent
        to execute certain behavior within a conversation (e.g. by using
        `trigger_external_user_uttered`).

        Args:
            sender_id: The ID of the conversation.
            action_name: The name of the action which should be executed.
            output_channel: The output channel which should be used for bot responses.
            nlg: The response generator.
            prediction: The prediction for the action.

        Returns:
            The new conversation state. Note that the new state is also persisted.
        """
        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = await self.fetch_tracker_and_update_session(sender_id, output_channel)

        action = self._get_action(action_name)
        await self._run_action(action, tracker, output_channel, nlg, prediction)

        # save tracker state to continue conversation from this state
        await self.save_tracker(tracker)

        return tracker

    def predict_next_with_tracker_if_should(
        self, tracker: DialogueStateTracker
    ) -> Tuple[rasa.core.actions.action.Action, PolicyPrediction]:
        """Predicts the next action the bot should take after seeing x.

        This should be overwritten by more advanced policies to use
        ML to predict the action.

        Returns:
             The index of the next action and prediction of the policy.

        Raises:
            ActionLimitReached if the limit of actions to predict has been reached.
        """
        should_predict_another_action = self.should_predict_another_action(
            tracker.latest_action_name
        )

        if self.is_action_limit_reached(tracker, should_predict_another_action):
            raise ActionLimitReached(
                "The limit of actions to predict has been reached."
            )

        prediction = self._predict_next_with_tracker(tracker)

        action = rasa.core.actions.action.action_for_index(
            prediction.max_confidence_index, self.domain, self.action_endpoint
        )

        logger.debug(
            f"Predicted next action '{action.name()}' with confidence "
            f"{prediction.max_confidence:.2f}."
        )

        return action, prediction

    @staticmethod
    def _is_reminder(e: Event, name: Text) -> bool:
        return isinstance(e, ReminderScheduled) and e.name == name

    @staticmethod
    def _is_reminder_still_valid(
        tracker: DialogueStateTracker, reminder_event: ReminderScheduled
    ) -> bool:
        """Check if the conversation has been restarted after reminder."""
        for e in reversed(tracker.applied_events()):
            if MessageProcessor._is_reminder(e, reminder_event.name):
                return True
        return False  # not found in applied events --> has been restarted

    @staticmethod
    def _has_message_after_reminder(
        tracker: DialogueStateTracker, reminder_event: ReminderScheduled
    ) -> bool:
        """Check if the user sent a message after the reminder."""
        for e in reversed(tracker.events):
            if MessageProcessor._is_reminder(e, reminder_event.name):
                return False

            if isinstance(e, UserUttered) and e.text:
                return True

        return True  # tracker has probably been restarted

    async def handle_reminder(
        self,
        reminder_event: ReminderScheduled,
        sender_id: Text,
        output_channel: OutputChannel,
    ) -> None:
        """Handle a reminder that is triggered asynchronously."""
        async with self.lock_store.lock(sender_id):
            tracker = await self.fetch_tracker_and_update_session(
                sender_id, output_channel
            )

            if (
                reminder_event.kill_on_user_message
                and self._has_message_after_reminder(tracker, reminder_event)
                or not self._is_reminder_still_valid(tracker, reminder_event)
            ):
                logger.debug(
                    f"Canceled reminder because it is outdated ({reminder_event})."
                )
            else:
                intent = reminder_event.intent
                entities: Union[List[Dict], Dict] = reminder_event.entities or {}
                await self.trigger_external_user_uttered(
                    intent, entities, tracker, output_channel
                )

    async def trigger_external_user_uttered(
        self,
        intent_name: Text,
        entities: Optional[Union[List[Dict[Text, Any]], Dict[Text, Text]]],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Triggers an external message.

        Triggers an external message (like a user message, but invisible;
        used, e.g., by a reminder or the trigger_intent endpoint).

        Args:
            intent_name: Name of the intent to be triggered.
            entities: Entities to be passed on.
            tracker: The tracker to which the event should be added.
            output_channel: The output channel.
        """
        if isinstance(entities, list):
            entity_list = entities
        elif isinstance(entities, dict):
            # Allow for a short-hand notation {"ent1": "val1", "ent2": "val2", ...}.
            # Useful if properties like 'start', 'end', or 'extractor' are not given,
            # e.g. for external events.
            entity_list = [
                {"entity": ent, "value": val} for ent, val in entities.items()
            ]
        elif not entities:
            entity_list = []
        else:
            rasa.shared.utils.io.raise_warning(
                f"Invalid entity specification: {entities}. Assuming no entities."
            )
            entity_list = []

        # Set the new event's input channel to the latest input channel, so
        # that we don't lose this property.
        input_channel = tracker.get_latest_input_channel()

        tracker.update(
            UserUttered.create_external(intent_name, entity_list, input_channel),
            self.domain,
        )

        tracker = await self.run_action_extract_slots(output_channel, tracker)

        await self._run_prediction_loop(output_channel, tracker)
        # save tracker state to continue conversation from this state
        await self.save_tracker(tracker)

    @staticmethod
    def _log_slots(tracker: DialogueStateTracker) -> None:
        # Log currently set slots
        slot_values = "\n".join(
            [f"\t{s.name}: {s.value}" for s in tracker.slots.values()]
        )
        if slot_values.strip():
            structlogger.debug(
                "processor.slots.log", slot_values=copy.deepcopy(slot_values)
            )

    def _check_for_unseen_features(self, parse_data: Dict[Text, Any]) -> None:
        """Warns the user if the NLU parse data contains unrecognized features.

        Checks intents and entities picked up by the NLU parsing
        against the domain and warns the user of those that don't match.
        Also considers a list of default intents that are valid but don't
        need to be listed in the domain.

        Args:
            parse_data: Message parse data to check against the domain.
        """
        if not self.domain or self.domain.is_empty():
            return

        intent = parse_data["intent"][INTENT_NAME_KEY]
        if intent and intent not in self.domain.intents:
            rasa.shared.utils.io.raise_warning(
                f"Parsed an intent '{intent}' "
                f"which is not defined in the domain. "
                f"Please make sure all intents are listed in the domain.",
                docs=DOCS_URL_DOMAINS,
            )

        entities = parse_data["entities"] or []
        for element in entities:
            entity = element["entity"]
            if entity and entity not in self.domain.entities:
                rasa.shared.utils.io.raise_warning(
                    f"Parsed an entity '{entity}' "
                    f"which is not defined in the domain. "
                    f"Please make sure all entities are listed in the domain.",
                    docs=DOCS_URL_DOMAINS,
                )

    def _get_action(
        self, action_name: Text
    ) -> Optional[rasa.core.actions.action.Action]:
        return rasa.core.actions.action.action_for_name_or_text(
            action_name, self.domain, self.action_endpoint
        )

    async def parse_message(
        self,
        message: UserMessage,
        tracker: Optional[DialogueStateTracker] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Interprets the passed message.

        Args:
            message: Message to handle.
            tracker: Tracker to use.
            only_output_properties: If `True`, restrict the output to
                Message.only_output_properties.

        Returns:
            Parsed data extracted from the message.
        """
        if self.http_interpreter:
            parse_data = await self.http_interpreter.parse(message)
        else:
            if tracker is None:
                tracker = DialogueStateTracker.from_events(message.sender_id, [])
            parse_data = self._parse_message_with_graph(
                message, tracker, only_output_properties
            )

        self._update_full_retrieval_intent(parse_data)
        structlogger.debug(
            "processor.message.parse",
            parse_data_text=copy.deepcopy(parse_data["text"]),
            parse_data_intent=parse_data["intent"],
            parse_data_entities=copy.deepcopy(parse_data["entities"]),
        )

        self._check_for_unseen_features(parse_data)

        return parse_data

    def _update_full_retrieval_intent(self, parse_data: Dict[Text, Any]) -> None:
        """Update the parse data with the full retrieval intent.

        Args:
            parse_data: Message parse data to update.
        """
        intent_name = parse_data.get(INTENT, {}).get(INTENT_NAME_KEY)
        response_selector = parse_data.get(RESPONSE_SELECTOR, {})
        all_retrieval_intents = response_selector.get("all_retrieval_intents", [])
        if intent_name and intent_name in all_retrieval_intents:
            retrieval_intent = (
                response_selector.get(intent_name, {})
                .get(RESPONSE, {})
                .get(INTENT_RESPONSE_KEY)
            )
            parse_data[INTENT][FULL_RETRIEVAL_INTENT_NAME_KEY] = retrieval_intent

    def _parse_message_with_graph(
        self,
        message: UserMessage,
        tracker: DialogueStateTracker,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Interprets the passed message.

        Arguments:
            message: Message to handle
            tracker: Tracker to use
            only_output_properties: If `True`, restrict the output to
                Message.only_output_properties.

        Returns:
            Parsed data extracted from the message.
        """
        results = self.graph_runner.run(
            inputs={PLACEHOLDER_MESSAGE: [message], PLACEHOLDER_TRACKER: tracker},
            targets=[self.model_metadata.nlu_target],
        )
        parsed_messages = results[self.model_metadata.nlu_target]
        parsed_message = parsed_messages[0]
        parse_data = {
            TEXT: "",
            INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
            ENTITIES: [],
        }
        parse_data.update(
            parsed_message.as_dict(only_output_properties=only_output_properties)
        )
        return parse_data

    async def _handle_message_with_tracker(
        self, message: UserMessage, tracker: DialogueStateTracker
    ) -> None:

        if message.parse_data:
            parse_data = message.parse_data
        else:
            parse_data = await self.parse_message(message, tracker)

        # don't ever directly mutate the tracker
        # - instead pass its events to log
        tracker.update(
            UserUttered(
                message.text,
                parse_data["intent"],
                parse_data["entities"],
                parse_data,
                input_channel=message.input_channel,
                message_id=message.message_id,
                metadata=message.metadata,
            ),
            self.domain,
        )

        if parse_data["entities"]:
            self._log_slots(tracker)

        logger.debug(
            f"Logged UserUtterance - tracker now has {len(tracker.events)} events."
        )

    @staticmethod
    def _should_handle_message(tracker: DialogueStateTracker) -> bool:
        return not tracker.is_paused() or (
            tracker.latest_message is not None
            and tracker.latest_message.intent.get(INTENT_NAME_KEY)
            == USER_INTENT_RESTART
        )

    def is_action_limit_reached(
        self, tracker: DialogueStateTracker, should_predict_another_action: bool
    ) -> bool:
        """Check whether the maximum number of predictions has been met.

        Args:
            tracker: instance of DialogueStateTracker.
            should_predict_another_action: Whether the last executed action allows
            for more actions to be predicted or not.

        Returns:
            `True` if the limit of actions to predict has been reached.
        """
        reversed_events = list(tracker.events)[::-1]
        num_predicted_actions = 0

        for e in reversed_events:
            if isinstance(e, ActionExecuted):
                if e.action_name in (ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME):
                    break
                num_predicted_actions += 1

        return (
            num_predicted_actions >= self.max_number_of_predictions
            and should_predict_another_action
        )

    async def _run_prediction_loop(
        self, output_channel: OutputChannel, tracker: DialogueStateTracker
    ) -> None:
        # keep taking actions decided by the policy until it chooses to 'listen'
        should_predict_another_action = True

        # action loop. predicts actions until we hit action listen
        while should_predict_another_action and self._should_handle_message(tracker):
            # this actually just calls the policy's method by the same name
            try:
                action, prediction = self.predict_next_with_tracker_if_should(tracker)
            except ActionLimitReached:
                logger.warning(
                    "Circuit breaker tripped. Stopped predicting "
                    f"more actions for sender '{tracker.sender_id}'."
                )
                if self.on_circuit_break:
                    # call a registered callback
                    self.on_circuit_break(tracker, output_channel, self.nlg)
                break

            if prediction.is_end_to_end_prediction:
                logger.debug(
                    f"An end-to-end prediction was made which has triggered the 2nd "
                    f"execution of the default action '{ACTION_EXTRACT_SLOTS}'."
                )
                tracker = await self.run_action_extract_slots(output_channel, tracker)

            should_predict_another_action = await self._run_action(
                action, tracker, output_channel, self.nlg, prediction
            )

    @staticmethod
    def should_predict_another_action(action_name: Text) -> bool:
        """Determine whether the processor should predict another action.

        Args:
            action_name: Name of the latest executed action.

        Returns:
            `False` if `action_name` is `ACTION_LISTEN_NAME` or
            `ACTION_SESSION_START_NAME`, otherwise `True`.
        """
        return action_name not in (ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME)

    async def execute_side_effects(
        self,
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Send bot messages, schedule and cancel reminders that are logged
        in the events array.
        """
        await self._send_bot_messages(events, tracker, output_channel)
        await self._schedule_reminders(events, tracker, output_channel)
        await self._cancel_reminders(events, tracker)

    @staticmethod
    async def _send_bot_messages(
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Send all the bot messages that are logged in the events array."""
        for e in events:
            if not isinstance(e, BotUttered):
                continue

            await output_channel.send_response(tracker.sender_id, e.message())

    async def _schedule_reminders(
        self,
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Uses the scheduler to time a job to trigger the passed reminder.

        Reminders with the same `id` property will overwrite one another
        (i.e. only one of them will eventually run).
        """
        for e in events:
            if not isinstance(e, ReminderScheduled):
                continue

            (await jobs.scheduler()).add_job(
                self.handle_reminder,
                "date",
                run_date=e.trigger_date_time,
                args=[e, tracker.sender_id, output_channel],
                id=e.name,
                replace_existing=True,
                name=e.scheduled_job_name(tracker.sender_id),
            )

    @staticmethod
    async def _cancel_reminders(
        events: List[Event], tracker: DialogueStateTracker
    ) -> None:
        """Cancel reminders that match the `ReminderCancelled` event."""
        # All Reminders specified by ReminderCancelled events will be cancelled
        for event in events:
            if isinstance(event, ReminderCancelled):
                scheduler = await jobs.scheduler()
                for scheduled_job in scheduler.get_jobs():
                    if event.cancels_job_with_name(
                        scheduled_job.name, tracker.sender_id
                    ):
                        scheduler.remove_job(scheduled_job.id)

    async def _run_action(
        self,
        action: rasa.core.actions.action.Action,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        prediction: PolicyPrediction,
    ) -> bool:
        # events and return values are used to update
        # the tracker state after an action has been taken
        try:
            # Use temporary tracker as we might need to discard the policy events in
            # case of a rejection.
            temporary_tracker = tracker.copy()
            temporary_tracker.update_with_events(prediction.events, self.domain)
            events = await action.run(
                output_channel, nlg, temporary_tracker, self.domain
            )
        except rasa.core.actions.action.ActionExecutionRejection:
            events = [
                ActionExecutionRejected(
                    action.name(), prediction.policy_name, prediction.max_confidence
                )
            ]
            tracker.update(events[0])
            return self.should_predict_another_action(action.name())
        except Exception:
            logger.exception(
                f"Encountered an exception while running action '{action.name()}'."
                "Bot will continue, but the actions events are lost. "
                "Please check the logs of your action server for "
                "more information."
            )
            events = []

        self._log_action_on_tracker(tracker, action, events, prediction)

        if any(isinstance(e, UserUttered) for e in events):
            logger.debug(
                f"A `UserUttered` event was returned by executing "
                f"action '{action.name()}'. This will run the default action "
                f"'{ACTION_EXTRACT_SLOTS}'."
            )
            tracker = await self.run_action_extract_slots(output_channel, tracker)

        if action.name() != ACTION_LISTEN_NAME and not action.name().startswith(
            UTTER_PREFIX
        ):
            self._log_slots(tracker)

        await self.execute_side_effects(events, tracker, output_channel)

        return self.should_predict_another_action(action.name())

    def _log_action_on_tracker(
        self,
        tracker: DialogueStateTracker,
        action: Action,
        events: Optional[List[Event]],
        prediction: PolicyPrediction,
    ) -> None:
        # Ensures that the code still works even if a lazy programmer missed
        # to type `return []` at the end of an action or the run method
        # returns `None` for some other reason.
        if events is None:
            events = []

        action_was_rejected_manually = any(
            isinstance(event, ActionExecutionRejected) for event in events
        )
        if not action_was_rejected_manually:
            structlogger.debug(
                "processor.actions.policy_prediction",
                prediction_events=copy.deepcopy(prediction.events),
            )
            tracker.update_with_events(prediction.events, self.domain)

            # log the action and its produced events
            tracker.update(action.event_for_successful_execution(prediction))

        structlogger.debug(
            "processor.actions.log",
            action_name=action.name(),
            rasa_events=copy.deepcopy(events),
        )
        tracker.update_with_events(events, self.domain)

    def _has_session_expired(self, tracker: DialogueStateTracker) -> bool:
        """Determine whether the latest session in `tracker` has expired.

        Args:
            tracker: Tracker to inspect.

        Returns:
            `True` if the session in `tracker` has expired, `False` otherwise.
        """
        if not self.domain.session_config.are_sessions_enabled():
            # tracker has never expired if sessions are disabled
            return False

        user_uttered_event: Optional[UserUttered] = tracker.get_last_event_for(
            UserUttered
        )

        if not user_uttered_event:
            # there is no user event so far so the session should not be considered
            # expired
            return False

        time_delta_in_seconds = time.time() - user_uttered_event.timestamp
        has_expired = (
            time_delta_in_seconds / 60
            > self.domain.session_config.session_expiration_time
        )
        if has_expired:
            logger.debug(
                f"The latest session for conversation ID '{tracker.sender_id}' has "
                f"expired."
            )

        return has_expired

    async def save_tracker(self, tracker: DialogueStateTracker) -> None:
        """Save the given tracker to the tracker store.

        Args:
            tracker: Tracker to be saved.
        """
        await self.tracker_store.save(tracker)

    def _predict_next_with_tracker(
        self, tracker: DialogueStateTracker
    ) -> PolicyPrediction:
        """Collect predictions from ensemble and return action and predictions."""
        followup_action = tracker.followup_action
        if followup_action:
            tracker.clear_followup_action()
            if followup_action in self.domain.action_names_or_texts:
                prediction = PolicyPrediction.for_action_name(
                    self.domain, followup_action, FOLLOWUP_ACTION
                )
                return prediction

            logger.error(
                f"Trying to run unknown follow-up action '{followup_action}'. "
                "Instead of running that, Rasa Open Source will ignore the action "
                "and predict the next action."
            )

        target = self.model_metadata.core_target
        if not target:
            raise ValueError("Cannot predict next action if there is no core target.")

        results = self.graph_runner.run(
            inputs={PLACEHOLDER_TRACKER: tracker}, targets=[target]
        )
        policy_prediction = results[target]
        return policy_prediction
