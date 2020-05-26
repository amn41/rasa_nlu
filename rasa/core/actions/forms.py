from typing import Text, List, Optional, Union, Any, Dict, Tuple
import logging

from rasa.core.channels import OutputChannel
from rasa.core.domain import Domain

from rasa.core.actions import Action
from rasa.core.actions.action import ActionExecutionRejection, RemoteAction
from rasa.core.events import Event, SlotSet, Form
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

# this slot is used to store information needed
# to do the form handling
REQUESTED_SLOT = "requested_slot"


class FormAction(Action):
    def __init__(
        self, form_name: Text, action_endpoint: Optional[EndpointConfig]
    ) -> None:
        self._form_name = form_name
        self.action_endpoint = action_endpoint
        self._domain: Optional[Domain] = None

    def name(self) -> Text:
        return self._form_name

    def required_slots(self, tracker: DialogueStateTracker) -> List[Text]:
        """A list of required slots that the form has to fill.

        Use `tracker` to request different list of slots
        depending on the state of the dialogue
        """

        return list(self._config().keys())

    def from_entity(
        self,
        entity: Text,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Dict[Text, Any]:
        """A dictionary for slot mapping to extract slot value.

        From:
        - an extracted entity
        - conditioned on
            - intent if it is not None
            - not_intent if it is not None,
                meaning user intent should not be this intent
            - role if it is not None
            - group if it is not None
        """

        intent, not_intent = self._list_intents(intent, not_intent)

        return {
            "type": "from_entity",
            "entity": entity,
            "intent": intent,
            "not_intent": not_intent,
            "role": role,
            "group": group,
        }

    def from_trigger_intent(
        self,
        value: Any,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
    ) -> Dict[Text, Any]:
        """A dictionary for slot mapping to extract slot value.

        From:
        - trigger_intent: value pair
        - conditioned on
            - intent if it is not None
            - not_intent if it is not None,
                meaning user intent should not be this intent

        Only used on form activation.
        """

        intent, not_intent = self._list_intents(intent, not_intent)

        return {
            "type": "from_trigger_intent",
            "value": value,
            "intent": intent,
            "not_intent": not_intent,
        }

    def from_intent(
        self,
        value: Any,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
    ) -> Dict[Text, Any]:
        """A dictionary for slot mapping to extract slot value.

        From:
        - intent: value pair
        - conditioned on
            - intent if it is not None
            - not_intent if it is not None,
                meaning user intent should not be this intent
        """

        intent, not_intent = self._list_intents(intent, not_intent)

        return {
            "type": "from_intent",
            "value": value,
            "intent": intent,
            "not_intent": not_intent,
        }

    def from_text(
        self,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
    ) -> Dict[Text, Any]:
        """A dictionary for slot mapping to extract slot value.

        From:
        - a whole message
        - conditioned on
            - intent if it is not None
            - not_intent if it is not None,
                meaning user intent should not be this intent
        """

        intent, not_intent = self._list_intents(intent, not_intent)

        return {"type": "from_text", "intent": intent, "not_intent": not_intent}

    # noinspection PyMethodMayBeStatic
    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict[Text, Any]]]]:
        """A dictionary to map required slots.

        Options:
        - an extracted entity
        - intent: value pairs
        - trigger_intent: value pairs
        - a whole message
        or a list of them, where the first match will be picked

        Empty dict is converted to a mapping of
        the slot to the extracted entity with the same name
        """

        if not self._domain:
            return {}
        return self._config()

    def _config(self) -> Dict:
        return next(
            (
                form[self._form_name]
                for form in self._domain.forms
                if self._form_name in form.keys()
            ),
            {},
        )

    def get_mappings_for_slot(self, slot_to_fill: Text) -> List[Dict[Text, Any]]:
        """Get mappings for requested slot.

        If None, map requested slot to an entity with the same name
        """

        requested_slot_mappings = self._to_list(
            self.slot_mappings().get(slot_to_fill, self.from_entity(slot_to_fill))
        )
        # check provided slot mappings
        for requested_slot_mapping in requested_slot_mappings:
            if (
                not isinstance(requested_slot_mapping, dict)
                or requested_slot_mapping.get("type") is None
            ):
                raise TypeError("Provided incompatible slot mapping")

        return requested_slot_mappings

    @staticmethod
    def intent_is_desired(
        requested_slot_mapping: Dict[Text, Any], tracker: "DialogueStateTracker"
    ) -> bool:
        """Check whether user intent matches intent conditions"""

        mapping_intents = requested_slot_mapping.get("intent", [])
        mapping_not_intents = requested_slot_mapping.get("not_intent", [])

        intent = tracker.latest_message.intent.get("name")

        intent_not_blacklisted = (
            not mapping_intents and intent not in mapping_not_intents
        )

        return intent_not_blacklisted or intent in mapping_intents

    def entity_is_desired(
        self,
        requested_slot_mapping: Dict[Text, Any],
        slot: Text,
        tracker: "DialogueStateTracker",
    ) -> bool:
        """Check whether slot should be filled by an entity in the input or not.

        Args:
            requested_slot_mapping: Slot mapping.
            slot: The slot to be filled.
            tracker: The tracker.

        Returns:
            True, if slot should be filled, false otherwise.
        """

        # slot name is equal to the entity type
        slot_equals_entity = slot == requested_slot_mapping.get("entity")

        # use the custom slot mapping 'from_entity' defined by the user to check
        # whether we can fill a slot with an entity
        matching_values = self.get_entity_value(
            requested_slot_mapping.get("entity"),
            tracker,
            requested_slot_mapping.get("role"),
            requested_slot_mapping.get("group"),
        )
        slot_fulfils_entity_mapping = matching_values is not None

        return slot_equals_entity or slot_fulfils_entity_mapping

    @staticmethod
    def get_entity_value(
        name: Text,
        tracker: "DialogueStateTracker",
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Any:
        """Extract entities for given name and optional role and group.

        Args:
            name: entity type (name) of interest
            tracker: the tracker
            role: optional entity role of interest
            group: optional entity group of interest

        Returns:
            Value of entity.
        """
        # list is used to cover the case of list slot type
        value = list(
            tracker.get_latest_entity_values(name, entity_group=group, entity_role=role)
        )
        if len(value) == 0:
            value = None
        elif len(value) == 1:
            value = value[0]
        return value

    # noinspection PyUnusedLocal
    def extract_other_slots(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Dict[Text, Any]:
        """Extract the values of the other slots
            if they are set by corresponding entities from the user input
            else return None
        """
        slot_to_fill = tracker.get_slot(REQUESTED_SLOT)

        slot_values = {}
        for slot in self.required_slots(tracker):
            # look for other slots
            if slot != slot_to_fill:
                # list is used to cover the case of list slot type
                other_slot_mappings = self.get_mappings_for_slot(slot)

                for other_slot_mapping in other_slot_mappings:
                    # check whether the slot should be filled by an entity in the input
                    should_fill_entity_slot = (
                        other_slot_mapping["type"] == "from_entity"
                        and self.intent_is_desired(other_slot_mapping, tracker)
                        and self.entity_is_desired(other_slot_mapping, slot, tracker)
                    )
                    # check whether the slot should be
                    # filled from trigger intent mapping
                    should_fill_trigger_slot = (
                        tracker.active_form.get("name") != self.name()
                        and other_slot_mapping["type"] == "from_trigger_intent"
                        and self.intent_is_desired(other_slot_mapping, tracker)
                    )
                    if should_fill_entity_slot:
                        value = self.get_entity_value(
                            other_slot_mapping["entity"],
                            tracker,
                            other_slot_mapping.get("role"),
                            other_slot_mapping.get("group"),
                        )
                    elif should_fill_trigger_slot:
                        value = other_slot_mapping.get("value")
                    else:
                        value = None

                    if value is not None:
                        logger.debug(f"Extracted '{value}' for extra slot '{slot}'.")
                        slot_values[slot] = value
                        # this slot is done, check  next
                        break

        return slot_values

    # noinspection PyUnusedLocal
    def extract_requested_slot(
        self, tracker: "DialogueStateTracker", domain: Domain
    ) -> Dict[Text, Any]:
        """Extract the value of requested slot from a user input
            else return None
        """
        slot_to_fill = tracker.get_slot(REQUESTED_SLOT)
        logger.debug(f"Trying to extract requested slot '{slot_to_fill}' ...")

        # get mapping for requested slot
        requested_slot_mappings = self.get_mappings_for_slot(slot_to_fill)

        for requested_slot_mapping in requested_slot_mappings:
            logger.debug(f"Got mapping '{requested_slot_mapping}'")

            if self.intent_is_desired(requested_slot_mapping, tracker):
                mapping_type = requested_slot_mapping["type"]

                if mapping_type == "from_entity":
                    value = self.get_entity_value(
                        requested_slot_mapping.get("entity"),
                        tracker,
                        requested_slot_mapping.get("role"),
                        requested_slot_mapping.get("group"),
                    )
                elif mapping_type == "from_intent":
                    value = requested_slot_mapping.get("value")
                elif mapping_type == "from_trigger_intent":
                    # from_trigger_intent is only used on form activation
                    continue
                elif mapping_type == "from_text":
                    value = tracker.latest_message.text
                else:
                    raise ValueError("Provided slot mapping type is not supported")

                if value is not None:
                    logger.debug(
                        f"Successfully extracted '{value}' for requested slot "
                        f"'{slot_to_fill}'"
                    )
                    return {slot_to_fill: value}

        logger.debug(f"Failed to extract requested slot '{slot_to_fill}'")
        return {}

    async def validate_slots(
        self,
        slot_dict: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """Validate slots using helper validation functions.

        Call validate_{slot} function for each slot, value pair to be validated.
        If this function is not implemented, set the slot to the value.
        """

        validate_name = f"action_validate_{self._form_name}"
        form_events = [
            SlotSet(slot_name, value) for slot_name, value in slot_dict.items()
        ]

        if validate_name not in domain.action_names or not tracker.active_form.get(
            "validate"
        ):
            # TODO: Currently the remote action needs to return SlotSet events for every
            # slot. it could be easy to forget to set one of them. We could
            # only overwrite Slot events from `form_events` if they were returned by the
            # form
            return form_events

        tracker = DialogueStateTracker.from_events(
            tracker.sender_id, tracker.events_after_latest_restart() + form_events
        )

        action = RemoteAction(validate_name, self.action_endpoint)
        return await action.run(output_channel, nlg, tracker, domain)

    async def validate(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """Extract and validate value of requested slot.

        If nothing was extracted reject execution of the form action.
        Subclass this method to add custom validation and rejection logic
        """

        # extract other slots that were not requested
        # but set by corresponding entity or trigger intent mapping
        slot_values = self.extract_other_slots(tracker, domain)

        # extract requested slot
        slot_to_fill = tracker.get_slot(REQUESTED_SLOT)
        if slot_to_fill:
            slot_values.update(self.extract_requested_slot(tracker, domain))

            if not slot_values:
                # reject to execute the form action
                # if some slot was requested but nothing was extracted
                # it will allow other policies to predict another action
                raise ActionExecutionRejection(
                    self.name(),
                    f"Failed to extract slot {slot_to_fill} with action {self.name()}",
                )
        logger.debug(f"Validating extracted slots: {slot_values}")
        return await self.validate_slots(
            slot_values, tracker, domain, output_channel, nlg
        )

    # noinspection PyUnusedLocal
    def request_next_slot(
        self, tracker: "DialogueStateTracker", domain: Domain
    ) -> Optional[List[Event]]:
        """Request the next slot and utter template if needed,
            else return None"""

        for slot in self.required_slots(tracker):
            if self._should_request_slot(tracker, slot):
                logger.debug(f"Request next slot '{slot}'")
                return [SlotSet(REQUESTED_SLOT, slot)]

        # no more required slots to fill
        return None

    def deactivate(self) -> List[Event]:
        """Return `Form` event with `None` as name to deactivate the form
            and reset the requested slot"""

        logger.debug(f"Deactivating the form '{self.name()}'")
        return [Form(None), SlotSet(REQUESTED_SLOT, None)]

    # helpers
    @staticmethod
    def _to_list(x: Optional[Any]) -> List[Any]:
        """Convert object to a list if it is not a list,
            None converted to empty list
        """
        if x is None:
            x = []
        elif not isinstance(x, list):
            x = [x]

        return x

    def _list_intents(
        self,
        intent: Optional[Union[Text, List[Text]]] = None,
        not_intent: Optional[Union[Text, List[Text]]] = None,
    ) -> Tuple[List[Text], List[Text]]:
        """Check provided intent and not_intent"""
        if intent and not_intent:
            raise ValueError(
                f"Providing  both intent '{intent}' and not_intent '{not_intent}' "
                f"is not supported."
            )

        return self._to_list(intent), self._to_list(not_intent)

    def _log_form_slots(self, tracker: "DialogueStateTracker") -> None:
        """Logs the values of all required slots before submitting the form."""
        slot_values = "\n".join(
            [
                f"\t{slot}: {tracker.get_slot(slot)}"
                for slot in self.required_slots(tracker)
            ]
        )
        logger.debug(
            f"No slots left to request, all required slots are filled:\n{slot_values}"
        )

    async def _activate_if_required(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """Activate form if the form is called for the first time.

        If activating, validate any required slots that were filled before
        form activation and return `Form` event with the name of the form, as well
        as any `SlotSet` events from validation of pre-filled slots.
        """

        if tracker.active_form.get("name") is not None:
            logger.debug(f"The form '{tracker.active_form}' is active")
        else:
            logger.debug("There is no active form")

        if tracker.active_form.get("name") == self.name():
            return []
        else:
            logger.debug(f"Activated the form '{self.name()}'")
            events = [Form(self.name())]

            # collect values of required slots filled before activation
            prefilled_slots = {}

            for slot_name in self.required_slots(tracker):
                if not self._should_request_slot(tracker, slot_name):
                    prefilled_slots[slot_name] = tracker.get_slot(slot_name)

            if prefilled_slots:
                logger.debug(f"Validating pre-filled required slots: {prefilled_slots}")
                events.extend(
                    await self.validate_slots(
                        prefilled_slots, tracker, domain, output_channel, nlg
                    )
                )
            else:
                logger.debug("No pre-filled required slots to validate.")

            return events

    async def _validate_if_required(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
    ) -> List[Event]:
        """Return a list of events from `self.validate(...)`
            if validation is required:
            - the form is active
            - the form is called after `action_listen`
            - form validation was not cancelled
        """
        if tracker.latest_action_name == "action_listen" and tracker.active_form.get(
            "validate", True
        ):
            logger.debug(f"Validating user input '{tracker.latest_message}'")
            return await self.validate(tracker, domain, output_channel, nlg)
        else:
            logger.debug("Skipping validation")
            return []

    @staticmethod
    def _should_request_slot(tracker: "DialogueStateTracker", slot_name: Text) -> bool:
        """Check whether form action should request given slot"""

        return tracker.get_slot(slot_name) is None

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """Execute the side effects of this form.

        Steps:
        - activate if needed
        - validate user input if needed
        - set validated slots
        - utter_ask_{slot} template with the next required slot
        - submit the form if all required slots are set
        - deactivate the form
        """

        self._domain = domain

        # activate the form (we don't return these events in case form immediately
        # finishes)
        activation_events = await self._activate_if_required(
            tracker, domain, output_channel, nlg
        )
        # validate user input
        events = await self._validate_if_required(tracker, domain, output_channel, nlg)
        # check that the form wasn't deactivated in validation
        if Form(None) not in events:
            temp_tracker = DialogueStateTracker.from_events(
                tracker.sender_id, tracker.events_after_latest_restart() + events
            )

            next_slot_events = self.request_next_slot(temp_tracker, domain)

            if next_slot_events is not None:
                # request next slot
                events = activation_events + events + next_slot_events
            else:
                # there is nothing more to request, so we can submit
                self._log_form_slots(temp_tracker)
                # deactivate the form after submission
                events.extend(self.deactivate())  # type: ignore

        return events

    def __str__(self) -> Text:
        return f"FormAction('{self.name()}')"
