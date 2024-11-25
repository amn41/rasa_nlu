import logging
from typing import List, Optional, Dict, Text, Set, Any

import numpy as np
import scipy.sparse

from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.utils import bilou_utils
from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
from rasa.shared.core.domain import SubState, State, Domain
from rasa.shared.core.constants import PREVIOUS_ACTION, ACTIVE_LOOP, USER, SLOTS
from rasa.shared.core.trackers import is_prev_action_listen_in_state
from rasa.shared.nlu.constants import (
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    ACTION_TEXT,
    ACTION_NAME,
    INTENT,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_TAGS,
    TEXT,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.utils.tensorflow import model_data_utils

logger = logging.getLogger(__name__)


class SingleStateFeaturizer:
    """Base class to transform the dialogue state into an ML format.

    Subclasses of SingleStateFeaturizer will decide how a bot will
    transform the dialogue state into a dictionary mapping an attribute
    to its features. Possible attributes are: `INTENT`, `TEXT`, `ACTION_NAME`,
    `ACTION_TEXT`, `ENTITIES`, `SLOTS` and `ACTIVE_LOOP`. Each attribute will be
    featurized into a list of `rasa.utils.features.Features`.
    """

    def __init__(self) -> None:
        """Initialize the single state featurizer."""
        self._default_feature_states: Dict[Text, Any] = {}
        self.action_texts: List[Text] = []
        self.entity_tag_specs: List[EntityTagSpec] = []

    def _create_entity_tag_specs(
        self, bilou_tagging: bool = False
    ) -> List[EntityTagSpec]:
        """Returns the tag to index mapping for entities.

        Returns:
            Tag to index mapping.
        """
        if ENTITIES not in self._default_feature_states:
            return []

        if bilou_tagging:
            tag_id_index_mapping = {
                f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
                for tag, idx_1 in self._default_feature_states[ENTITIES].items()
                for idx_2, prefix in enumerate(BILOU_PREFIXES)
            }
        else:
            tag_id_index_mapping = {
                tag: idx + 1  # +1 to keep 0 for the NO_ENTITY_TAG
                for tag, idx in self._default_feature_states[ENTITIES].items()
            }

        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_index_mapping[NO_ENTITY_TAG] = 0

        # TODO
        #  The entity states used to create the tag-idx-mapping contains the
        #  entities and the concatenated entity and roles/groups. We do not
        #  distinguish between entities and roles/groups right now.
        #  we return a list to anticipate that
        return [
            EntityTagSpec(
                tag_name=ENTITY_ATTRIBUTE_TYPE,
                tags_to_ids=tag_id_index_mapping,
                ids_to_tags={value: key for key, value in tag_id_index_mapping.items()},
                num_tags=len(tag_id_index_mapping),
            )
        ]

    def prepare_for_training(self, domain: Domain, bilou_tagging: bool = False) -> None:
        """Gets necessary information for featurization from domain.

        Args:
            domain: An instance of :class:`rasa.shared.core.domain.Domain`.
            bilou_tagging: indicates whether BILOU tagging should be used or not
        """
        # store feature states for each attribute in order to create binary features
        def convert_to_dict(feature_states: List[Text]) -> Dict[Text, int]:
            return {
                feature_state: idx for idx, feature_state in enumerate(feature_states)
            }

        self._default_feature_states[INTENT] = convert_to_dict(domain.intents)
        self._default_feature_states[ACTION_NAME] = convert_to_dict(
            domain.action_names_or_texts
        )
        self._default_feature_states[ENTITIES] = convert_to_dict(domain.entity_states)
        self._default_feature_states[SLOTS] = convert_to_dict(domain.slot_states)
        self._default_feature_states[ACTIVE_LOOP] = convert_to_dict(domain.form_names)
        self.action_texts = domain.action_texts
        self.entity_tag_specs = self._create_entity_tag_specs(bilou_tagging)

    def _state_features_for_attribute(
        self, sub_state: SubState, attribute: Text
    ) -> Dict[Text, int]:
        # FIXME: the code below is not type-safe, but fixing it
        #        would require more refactoring, for instance using
        #        data classes in our states
        if attribute in {INTENT, ACTION_NAME}:
            return {sub_state[attribute]: 1}  # type: ignore[dict-item]
        elif attribute == ENTITIES:
            return {entity: 1 for entity in sub_state.get(ENTITIES, [])}  # type: ignore[misc]  # noqa: E501
        elif attribute == ACTIVE_LOOP:
            return {sub_state["name"]: 1}  # type: ignore[dict-item]
        elif attribute == SLOTS:
            return {
                f"{slot_name}_{i}": value  # type: ignore[misc]
                for slot_name, slot_as_feature in sub_state.items()
                for i, value in enumerate(slot_as_feature)
            }
        else:
            raise ValueError(
                f"Given attribute '{attribute}' is not supported. "
                f"It must be one of '{self._default_feature_states.keys()}'."
            )

    def _create_features(
        self, sub_state: SubState, attribute: Text, sparse: bool = False
    ) -> List[Features]:
        state_features = self._state_features_for_attribute(sub_state, attribute)

        features = np.zeros(len(self._default_feature_states[attribute]), np.float32)
        for state_feature, value in state_features.items():
            # check that the value is in default_feature_states to be able to assign
            # its value
            if state_feature in self._default_feature_states[attribute]:
                features[self._default_feature_states[attribute][state_feature]] = value
        features = np.expand_dims(features, 0)

        if sparse:
            features = scipy.sparse.coo_matrix(features)

        return [
            Features(
                features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
            )
        ]

    @staticmethod
    def _to_sparse_sentence_features(
        sparse_sequence_features: List[Features],
    ) -> List[Features]:
        return [
            Features(
                scipy.sparse.coo_matrix(feature.features.sum(0)),
                FEATURE_TYPE_SENTENCE,
                feature.attribute,
                feature.origin,
            )
            for feature in sparse_sequence_features
        ]

    @staticmethod
    def _get_name_attribute(attributes: Set[Text]) -> Optional[Text]:
        # there is always either INTENT or ACTION_NAME
        return next(
            (
                attribute
                for attribute in attributes
                if attribute in {INTENT, ACTION_NAME}
            ),
            None,
        )

    def _extract_state_features(
        self,
        sub_state: SubState,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        sparse: bool = False,
    ) -> Dict[Text, List[Features]]:

        # Remove entities from possible attributes
        attributes = set(
            attribute for attribute in sub_state.keys() if attribute != ENTITIES
        )

        if precomputations is not None:

            # Collect features for all those attributes
            attributes_to_features = precomputations.collect_features(
                sub_state, attributes=attributes
            )
            # if features for INTENT or ACTION_NAME exist,
            # they are always sparse sequence features;
            # transform them to sentence sparse features
            if attributes_to_features.get(INTENT):
                attributes_to_features[INTENT] = self._to_sparse_sentence_features(
                    attributes_to_features[INTENT]
                )
            if attributes_to_features.get(ACTION_NAME):
                attributes_to_features[ACTION_NAME] = self._to_sparse_sentence_features(
                    attributes_to_features[ACTION_NAME]
                )

            # Combine and sort the features:
            # Per attribute, combine features of same type and level into one Feature,
            # and (if there are any such features) store the results in a list where
            # - all the sparse features are listed first and a
            # - sequence feature is always listed before the sentence feature of the
            #   same type (sparse/not sparse).
            output = {
                attribute: Features.reduce(
                    features_list=features_list, expected_origins=None
                )
                for attribute, features_list in attributes_to_features.items()
                if len(features_list) > 0  # otherwise, following will fail
            }
        else:
            output = {}

        # Check that the name attribute has features
        name_attribute = self._get_name_attribute(attributes)
        if name_attribute and name_attribute not in output:
            # nlu pipeline didn't create features for user or action
            # this might happen, for example, when we have action_name in the state
            # but it did not get featurized because only character level
            # CountVectorsFeaturizer was included in the config.
            output[name_attribute] = self._create_features(
                sub_state, name_attribute, sparse
            )
        return output

    def encode_state(
        self,
        state: State,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Dict[Text, List[Features]]:
        """Encode the given state.

        Args:
            state: The state to encode
            precomputations: Contains precomputed features and attributes.

        Returns:
            A dictionary of state_type to list of features.
        """
        state_features = {}
        for state_type, sub_state in state.items():
            if state_type == PREVIOUS_ACTION:
                state_features.update(
                    self._extract_state_features(
                        sub_state, precomputations=precomputations, sparse=True
                    )
                )
            # featurize user only if it is "real" user input,
            # i.e. input from a turn after action_listen
            if state_type == USER and is_prev_action_listen_in_state(state):

                state_features.update(
                    self._extract_state_features(
                        sub_state, precomputations=precomputations, sparse=True
                    )
                )
                if sub_state.get(ENTITIES):
                    state_features[ENTITIES] = self._create_features(
                        sub_state, ENTITIES, sparse=True
                    )

            if state_type in {SLOTS, ACTIVE_LOOP}:
                state_features[state_type] = self._create_features(
                    sub_state, state_type, sparse=True
                )

        return state_features

    def encode_entities(
        self,
        entity_data: Dict[Text, Any],
        precomputations: Optional[MessageContainerForCoreFeaturization],
        bilou_tagging: bool = False,
    ) -> Dict[Text, List[Features]]:
        """Encode the given entity data.

        Produce numeric entity tags for tokens.

        Args:
            entity_data: The dict containing the text and entity labels and locations
            precomputations: Contains precomputed features and attributes.
            bilou_tagging: indicates whether BILOU tagging should be used or not

        Returns:
            A dictionary of entity type to list of features.
        """
        # TODO
        #  The entity states used to create the tag-idx-mapping contains the
        #  entities and the concatenated entity and roles/groups. We do not
        #  distinguish between entities and roles/groups right now.
        if (
            not entity_data
            or not self.entity_tag_specs
            or self.entity_tag_specs[0].num_tags < 2
        ):
            # we cannot build a classifier with fewer than 2 classes
            return {}
        if precomputations is None:
            message = None
        else:
            message = precomputations.lookup_message(user_text=entity_data[TEXT])
            message.data[ENTITIES] = entity_data[ENTITIES]

        if not message:
            return {}

        if bilou_tagging:
            bilou_utils.apply_bilou_schema_to_message(message)

        return {
            ENTITY_TAGS: [
                model_data_utils.get_tag_ids(
                    message, self.entity_tag_specs[0], bilou_tagging
                )
            ]
        }

    def _encode_action(
        self,
        action: Text,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Dict[Text, List[Features]]:
        if action in self.action_texts:
            action_as_sub_state = {ACTION_TEXT: action}
        else:
            action_as_sub_state = {ACTION_NAME: action}

        return self._extract_state_features(
            action_as_sub_state, precomputations=precomputations
        )

    def encode_all_labels(
        self,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> List[Dict[Text, List[Features]]]:
        """Encode all action from the domain.

        Args:
            domain: The domain that contains the actions.
            precomputations: Contains precomputed features and attributes.

        Returns:
            A list of encoded actions.
        """
        return [
            self._encode_action(action, precomputations)
            for action in domain.action_names_or_texts
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_texts": self.action_texts,
            "entity_tag_specs": self.entity_tag_specs,
            "feature_states": self._default_feature_states,
        }

    @classmethod
    def create_from_dict(
        cls, data: Dict[str, Any]
    ) -> Optional["SingleStateFeaturizer"]:
        if not data:
            return None

        featurizer = SingleStateFeaturizer()
        featurizer.action_texts = data["action_texts"]
        featurizer._default_feature_states = data["feature_states"]
        featurizer.entity_tag_specs = data["entity_tag_specs"]
        return featurizer


class IntentTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    """A SingleStateFeaturizer for use with policies that predict intent labels."""

    def _encode_intent(
        self,
        intent: Text,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Dict[Text, List[Features]]:
        """Extracts a numeric representation of an intent.

        Args:
            intent: Intent to be encoded.
            precomputations: Contains precomputed features and attributes.

        Returns:
            Encoded representation of intent.
        """
        intent_as_sub_state = {INTENT: intent}
        return self._extract_state_features(intent_as_sub_state, precomputations)

    def encode_all_labels(
        self,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> List[Dict[Text, List[Features]]]:
        """Encodes all relevant labels from the domain using the given precomputations.

        Args:
            domain: The domain that contains the labels.
            precomputations: Contains precomputed features and attributes.

        Returns:
            A list of encoded labels.
        """
        return [
            self._encode_intent(intent, precomputations) for intent in domain.intents
        ]
