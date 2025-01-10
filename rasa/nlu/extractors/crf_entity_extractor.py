from __future__ import annotations

import logging
import typing
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional, Text, Tuple, Callable, Type

import numpy as np

import rasa.nlu.utils.bilou_utils as bilou_utils
import rasa.shared.utils.io
import rasa.utils.train_utils
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.nlu.test import determine_token_labels
from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
    SPLIT_ENTITIES_BY_COMMA,
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.constants import BILOU_FLAG, FEATURIZERS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from sklearn_crfsuite import CRF


CONFIG_FEATURES = "features"


class CRFToken:
    def __init__(
        self,
        text: Text,
        pos_tag: Text,
        pattern: Dict[Text, Any],
        dense_features: np.ndarray,
        entity_tag: Text,
        entity_role_tag: Text,
        entity_group_tag: Text,
    ):
        self.text = text
        self.pos_tag = pos_tag
        self.pattern = pattern
        self.dense_features = dense_features
        self.entity_tag = entity_tag
        self.entity_role_tag = entity_role_tag
        self.entity_group_tag = entity_group_tag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "pos_tag": self.pos_tag,
            "pattern": self.pattern,
            "dense_features": [str(x) for x in list(self.dense_features)],
            "entity_tag": self.entity_tag,
            "entity_role_tag": self.entity_role_tag,
            "entity_group_tag": self.entity_group_tag,
        }

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "CRFToken":
        return cls(
            data["text"],
            data["pos_tag"],
            data["pattern"],
            np.array([float(x) for x in data["dense_features"]]),
            data["entity_tag"],
            data["entity_role_tag"],
            data["entity_group_tag"],
        )


class CRFEntityExtractorOptions(str, Enum):
    """Features that can be used for the 'CRFEntityExtractor'."""

    PATTERN = "pattern"
    LOW = "low"
    TITLE = "title"
    PREFIX5 = "prefix5"
    PREFIX2 = "prefix2"
    SUFFIX5 = "suffix5"
    SUFFIX3 = "suffix3"
    SUFFIX2 = "suffix2"
    SUFFIX1 = "suffix1"
    BIAS = "bias"
    POS = "pos"
    POS2 = "pos2"
    UPPER = "upper"
    DIGIT = "digit"
    TEXT_DENSE_FEATURES = "text_dense_features"
    ENTITY = "entity"


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=True
)
class CRFEntityExtractor(GraphComponent, EntityExtractorMixin):
    """Implements conditional random fields (CRF) to do named entity recognition."""

    CONFIG_FEATURES = "features"

    function_dict: Dict[Text, Callable[[CRFToken], Any]] = {
        CRFEntityExtractorOptions.LOW: lambda crf_token: crf_token.text.lower(),
        CRFEntityExtractorOptions.TITLE: lambda crf_token: crf_token.text.istitle(),
        CRFEntityExtractorOptions.PREFIX5: lambda crf_token: crf_token.text[:5],
        CRFEntityExtractorOptions.PREFIX2: lambda crf_token: crf_token.text[:2],
        CRFEntityExtractorOptions.SUFFIX5: lambda crf_token: crf_token.text[-5:],
        CRFEntityExtractorOptions.SUFFIX3: lambda crf_token: crf_token.text[-3:],
        CRFEntityExtractorOptions.SUFFIX2: lambda crf_token: crf_token.text[-2:],
        CRFEntityExtractorOptions.SUFFIX1: lambda crf_token: crf_token.text[-1:],
        CRFEntityExtractorOptions.BIAS: lambda _: "bias",
        CRFEntityExtractorOptions.POS: lambda crf_token: crf_token.pos_tag,
        CRFEntityExtractorOptions.POS2: lambda crf_token: crf_token.pos_tag[:2]
        if crf_token.pos_tag is not None
        else None,
        CRFEntityExtractorOptions.UPPER: lambda crf_token: crf_token.text.isupper(),
        CRFEntityExtractorOptions.DIGIT: lambda crf_token: crf_token.text.isdigit(),
        CRFEntityExtractorOptions.PATTERN: lambda crf_token: crf_token.pattern,
        CRFEntityExtractorOptions.TEXT_DENSE_FEATURES: (
            lambda crf_token: CRFEntityExtractor._convert_dense_features_for_crfsuite(  # noqa: E501
                crf_token
            )
        ),
        CRFEntityExtractorOptions.ENTITY: lambda crf_token: crf_token.entity_tag,
    }

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # BILOU_flag determines whether to use BILOU tagging or not.
            # More rigorous however requires more examples per entity
            # rule of thumb: use only if more than 100 egs. per entity
            BILOU_FLAG: True,
            # Split entities by comma, this makes sense e.g. for a list of ingredients
            # in a recipie, but it doesn't make sense for the parts of an address
            SPLIT_ENTITIES_BY_COMMA: True,
            # crf_features is [before, token, after] array with before, token,
            # after holding keys about which features to use for each token,
            # for example, 'title' in array before will have the feature
            # "is the preceding token in title case?"
            # POS features require SpacyTokenizer
            # pattern feature require RegexFeaturizer
            CONFIG_FEATURES: [
                [
                    CRFEntityExtractorOptions.LOW,
                    CRFEntityExtractorOptions.TITLE,
                    CRFEntityExtractorOptions.UPPER,
                ],
                [
                    CRFEntityExtractorOptions.LOW,
                    CRFEntityExtractorOptions.BIAS,
                    CRFEntityExtractorOptions.PREFIX5,
                    CRFEntityExtractorOptions.PREFIX2,
                    CRFEntityExtractorOptions.SUFFIX5,
                    CRFEntityExtractorOptions.SUFFIX3,
                    CRFEntityExtractorOptions.SUFFIX2,
                    CRFEntityExtractorOptions.UPPER,
                    CRFEntityExtractorOptions.TITLE,
                    CRFEntityExtractorOptions.DIGIT,
                    CRFEntityExtractorOptions.PATTERN,
                ],
                [
                    CRFEntityExtractorOptions.LOW,
                    CRFEntityExtractorOptions.TITLE,
                    CRFEntityExtractorOptions.UPPER,
                ],
            ],
            # The maximum number of iterations for optimization algorithms.
            "max_iterations": 50,
            # weight of the L1 regularization
            "L1_c": 0.1,
            # weight of the L2 regularization
            "L2_c": 0.1,
            # Name of dense featurizers to use.
            # If list is empty all available dense features are used.
            "featurizers": [],
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        entity_taggers: Optional[Dict[Text, "CRF"]] = None,
    ) -> None:
        """Creates an instance of entity extractor."""
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource

        self.entity_taggers = entity_taggers

        self.crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

        self._validate_configuration()

        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            config[SPLIT_ENTITIES_BY_COMMA], SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE
        )

    def _validate_configuration(self) -> None:
        if len(self.component_config.get(CONFIG_FEATURES, [])) % 2 != 1:
            raise ValueError(
                "Need an odd number of crf feature lists to have a center word."
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CRFEntityExtractor:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn_crfsuite", "sklearn"]

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the extractor on a data set."""
        # checks whether there is at least one
        # example with an entity annotation
        if not training_data.entity_examples:
            logger.debug(
                "No training examples with entities present. Skip training"
                "of 'CRFEntityExtractor'."
            )
            return self._resource

        self.check_correct_entity_annotations(training_data)

        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)

        # only keep the CRFs for tags we actually have training data for
        self._update_crf_order(training_data)

        # filter out pre-trained entity examples
        entity_examples = self.filter_trainable_entities(training_data.nlu_examples)
        entity_examples = [
            message
            for message in entity_examples
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]
        dataset = [self._convert_to_crf_tokens(example) for example in entity_examples]

        self.entity_taggers = self.train_model(
            dataset, self.component_config, self.crf_order
        )

        self.persist(dataset)

        return self._resource

    def _update_crf_order(self, training_data: TrainingData) -> None:
        """Train only CRFs we actually have training data for."""
        _crf_order = []

        for tag_name in self.crf_order:
            if tag_name == ENTITY_ATTRIBUTE_TYPE and training_data.entities:
                _crf_order.append(ENTITY_ATTRIBUTE_TYPE)
            elif tag_name == ENTITY_ATTRIBUTE_ROLE and training_data.entity_roles:
                _crf_order.append(ENTITY_ATTRIBUTE_ROLE)
            elif tag_name == ENTITY_ATTRIBUTE_GROUP and training_data.entity_groups:
                _crf_order.append(ENTITY_ATTRIBUTE_GROUP)

        self.crf_order = _crf_order

    def process(self, messages: List[Message]) -> List[Message]:
        """Augments messages with entities."""
        for message in messages:
            entities = self.extract_entities(message)
            entities = self.add_extractor_name(entities)
            message.set(
                ENTITIES, message.get(ENTITIES, []) + entities, add_to_output=True
            )

        return messages

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities from the given message using the trained model(s)."""
        if self.entity_taggers is None or not message.features_present(
            attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
        ):
            return []

        tokens = message.get(TOKENS_NAMES[TEXT])
        crf_tokens = self._convert_to_crf_tokens(message)

        predictions: Dict[Text, List[Dict[Text, float]]] = {}
        for tag_name, entity_tagger in self.entity_taggers.items():
            # use predicted entity tags as features for second level CRFs
            include_tag_features = tag_name != ENTITY_ATTRIBUTE_TYPE
            if include_tag_features:
                self._add_tag_to_crf_token(crf_tokens, predictions)

            features = self._crf_tokens_to_features(
                crf_tokens, self.component_config, include_tag_features
            )
            predictions[tag_name] = entity_tagger.predict_marginals_single(features)

        # convert predictions into a list of tags and a list of confidences
        tags, confidences = self._tag_confidences(tokens, predictions)

        return self.convert_predictions_into_entities(
            message.get(TEXT), tokens, tags, self.split_entities_config, confidences
        )

    def _add_tag_to_crf_token(
        self,
        crf_tokens: List[CRFToken],
        predictions: Dict[Text, List[Dict[Text, float]]],
    ) -> None:
        """Add predicted entity tags to CRF tokens."""
        if ENTITY_ATTRIBUTE_TYPE in predictions:
            _tags, _ = self._most_likely_tag(predictions[ENTITY_ATTRIBUTE_TYPE])
            for tag, token in zip(_tags, crf_tokens):
                token.entity_tag = tag

    def _most_likely_tag(
        self, predictions: List[Dict[Text, float]]
    ) -> Tuple[List[Text], List[float]]:
        """Get the entity tags with the highest confidence.

        Args:
            predictions: list of mappings from entity tag to confidence value

        Returns:
            List of entity tags and list of confidence values.
        """
        _tags = []
        _confidences = []

        for token_predictions in predictions:
            tag = max(token_predictions, key=lambda key: token_predictions[key])
            _tags.append(tag)

            if self.component_config[BILOU_FLAG]:
                # if we are using BILOU flags, we will sum up the prob
                # of the B, I, L and U tags for an entity
                _confidences.append(
                    sum(
                        _confidence
                        for _tag, _confidence in token_predictions.items()
                        if bilou_utils.tag_without_prefix(tag)
                        == bilou_utils.tag_without_prefix(_tag)
                    )
                )
            else:
                _confidences.append(token_predictions[tag])

        return _tags, _confidences

    def _tag_confidences(
        self, tokens: List[Token], predictions: Dict[Text, List[Dict[Text, float]]]
    ) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]:
        """Get most likely tag predictions with confidence values for tokens."""
        tags = {}
        confidences = {}

        for tag_name, predicted_tags in predictions.items():
            if len(tokens) != len(predicted_tags):
                raise Exception(
                    "Inconsistency in amount of tokens between crfsuite and message"
                )

            _tags, _confidences = self._most_likely_tag(predicted_tags)

            if self.component_config[BILOU_FLAG]:
                _tags, _confidences = bilou_utils.ensure_consistent_bilou_tagging(
                    _tags, _confidences
                )

            confidences[tag_name] = _confidences
            tags[tag_name] = _tags

        return tags, confidences

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> CRFEntityExtractor:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_dir:
                dataset = rasa.shared.utils.io.read_json_file(
                    model_dir / "crf_dataset.json"
                )
                crf_order = rasa.shared.utils.io.read_json_file(
                    model_dir / "crf_order.json"
                )

                dataset = [
                    [CRFToken.create_from_dict(token_data) for token_data in sub_list]
                    for sub_list in dataset
                ]

                entity_taggers = cls.train_model(dataset, config, crf_order)

                entity_extractor = cls(config, model_storage, resource, entity_taggers)
                entity_extractor.crf_order = crf_order
                return entity_extractor
        except ValueError:
            logger.warning(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource)

    def persist(self, dataset: List[List[CRFToken]]) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as model_dir:
            data_to_store = [
                [token.to_dict() for token in sub_list] for sub_list in dataset
            ]

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_dir / "crf_dataset.json", data_to_store
            )
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_dir / "crf_order.json", self.crf_order
            )

    @classmethod
    def _crf_tokens_to_features(
        cls,
        crf_tokens: List[CRFToken],
        config: Dict[str, Any],
        include_tag_features: bool = False,
    ) -> List[Dict[Text, Any]]:
        """Convert the list of tokens into discrete features."""
        configured_features = config[CONFIG_FEATURES]
        sentence_features = []

        for token_idx in range(len(crf_tokens)):
            # the features for the current token include features of the token
            # before and after the current features (if defined in the config)
            # token before (-1), current token (0), token after (+1)
            window_size = len(configured_features)
            half_window_size = window_size // 2
            window_range = range(-half_window_size, half_window_size + 1)

            token_features = cls._create_features_for_token(
                crf_tokens,
                token_idx,
                half_window_size,
                window_range,
                include_tag_features,
                config,
            )

            sentence_features.append(token_features)

        return sentence_features

    @classmethod
    def _create_features_for_token(
        cls,
        crf_tokens: List[CRFToken],
        token_idx: int,
        half_window_size: int,
        window_range: range,
        include_tag_features: bool,
        config: Dict[str, Any],
    ) -> Dict[Text, Any]:
        """Convert a token into discrete features including words before and after."""
        configured_features = config[CONFIG_FEATURES]
        prefixes = [str(i) for i in window_range]

        token_features = {}

        # iterate over the tokens in the window range (-1, 0, +1) to collect the
        # features for the token at token_idx
        for pointer_position in window_range:
            current_token_idx = token_idx + pointer_position

            if current_token_idx >= len(crf_tokens):
                # token is at the end of the sentence
                token_features["EOS"] = True
            elif current_token_idx < 0:
                # token is at the beginning of the sentence
                token_features["BOS"] = True
            else:
                token = crf_tokens[current_token_idx]

                # get the features to extract for the token we are currently looking at
                current_feature_idx = pointer_position + half_window_size
                features = configured_features[current_feature_idx]

                prefix = prefixes[current_feature_idx]

                # we add the 'entity' feature to include the entity type as features
                # for the role and group CRFs
                # (do not modify features, otherwise we will end up adding 'entity'
                # over and over again, making training very slow)
                additional_features = []
                if include_tag_features:
                    additional_features.append(CRFEntityExtractorOptions.ENTITY)

                for feature in features + additional_features:
                    if feature == CRFEntityExtractorOptions.PATTERN:
                        # add all regexes extracted from the 'RegexFeaturizer' as a
                        # feature: 'pattern_name' is the name of the pattern the user
                        # set in the training data, 'matched' is either 'True' or
                        # 'False' depending on whether the token actually matches the
                        # pattern or not
                        regex_patterns = cls.function_dict[feature](token)
                        for pattern_name, matched in regex_patterns.items():
                            token_features[
                                f"{prefix}:{feature}:{pattern_name}"
                            ] = matched
                    else:
                        value = cls.function_dict[feature](token)
                        token_features[f"{prefix}:{feature}"] = value

        return token_features

    @staticmethod
    def _crf_tokens_to_tags(crf_tokens: List[CRFToken], tag_name: Text) -> List[Text]:
        """Return the list of tags for the given tag name."""
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            return [crf_token.entity_role_tag for crf_token in crf_tokens]
        if tag_name == ENTITY_ATTRIBUTE_GROUP:
            return [crf_token.entity_group_tag for crf_token in crf_tokens]

        return [crf_token.entity_tag for crf_token in crf_tokens]

    @staticmethod
    def _pattern_of_token(message: Message, idx: int) -> Dict[Text, bool]:
        """Get the patterns of the token at the given index extracted by the
        'RegexFeaturizer'.

        The 'RegexFeaturizer' adds all patterns listed in the training data to the
        token. The pattern name is mapped to either 'True' (pattern applies to token) or
        'False' (pattern does not apply to token).

        Args:
            message: The message.
            idx: The token index.

        Returns:
            The pattern dict.
        """
        if message.get(TOKENS_NAMES[TEXT]) is not None:
            return message.get(TOKENS_NAMES[TEXT])[idx].get(
                CRFEntityExtractorOptions.PATTERN, {}
            )
        return {}

    def _get_dense_features(self, message: Message) -> Optional[np.ndarray]:
        """Convert dense features to python-crfsuite feature format."""
        features, _ = message.get_dense_features(
            TEXT, self.component_config["featurizers"]
        )

        if features is None:
            return None

        tokens = message.get(TOKENS_NAMES[TEXT])
        if len(tokens) != len(features.features):
            rasa.shared.utils.io.raise_warning(
                f"Number of dense features ({len(features.features)}) for attribute "
                f"'{TEXT}' does not match number of tokens ({len(tokens)}).",
                docs=DOCS_URL_COMPONENTS + "#crfentityextractor",
            )
            return None

        return features.features

    @staticmethod
    def _convert_dense_features_for_crfsuite(
        crf_token: CRFToken,
    ) -> Dict[Text, Dict[Text, float]]:
        """Converts dense features of CRFTokens to dicts for the crfsuite."""
        feature_dict = {
            str(index): token_features
            for index, token_features in enumerate(crf_token.dense_features)
        }
        converted = {"text_dense_features": feature_dict}
        return converted

    def _convert_to_crf_tokens(self, message: Message) -> List[CRFToken]:
        """Take a message and convert it to crfsuite format."""
        crf_format = []
        tokens = message.get(TOKENS_NAMES[TEXT])

        text_dense_features = self._get_dense_features(message)
        tags = self._get_tags(message)

        for i, token in enumerate(tokens):
            pattern = self._pattern_of_token(message, i)
            entity = self.get_tag_for(tags, ENTITY_ATTRIBUTE_TYPE, i)
            group = self.get_tag_for(tags, ENTITY_ATTRIBUTE_GROUP, i)
            role = self.get_tag_for(tags, ENTITY_ATTRIBUTE_ROLE, i)
            pos_tag = token.get(POS_TAG_KEY)
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )

            crf_format.append(
                CRFToken(
                    text=token.text,
                    pos_tag=pos_tag,
                    entity_tag=entity,
                    entity_group_tag=group,
                    entity_role_tag=role,
                    pattern=pattern,
                    dense_features=dense_features,
                )
            )

        return crf_format

    def _get_tags(self, message: Message) -> Dict[Text, List[Text]]:
        """Get assigned entity tags of message."""
        tokens = message.get(TOKENS_NAMES[TEXT])
        tags = {}

        for tag_name in self.crf_order:
            if self.component_config[BILOU_FLAG]:
                bilou_key = bilou_utils.get_bilou_key_for_tag(tag_name)
                if message.get(bilou_key):
                    _tags = message.get(bilou_key)
                else:
                    _tags = [NO_ENTITY_TAG for _ in tokens]
            else:
                _tags = [
                    determine_token_labels(
                        token, message.get(ENTITIES), attribute_key=tag_name
                    )
                    for token in tokens
                ]
            tags[tag_name] = _tags

        return tags

    @classmethod
    def train_model(
        cls,
        df_train: List[List[CRFToken]],
        config: Dict[str, Any],
        crf_order: List[str],
    ) -> OrderedDict[str, CRF]:
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        entity_taggers = OrderedDict()

        for tag_name in crf_order:
            logger.debug(f"Training CRF for '{tag_name}'.")

            # add entity tag features for second level CRFs
            include_tag_features = tag_name != ENTITY_ATTRIBUTE_TYPE
            X_train = (
                cls._crf_tokens_to_features(sentence, config, include_tag_features)
                for sentence in df_train
            )
            y_train = (
                cls._crf_tokens_to_tags(sentence, tag_name) for sentence in df_train
            )

            entity_tagger = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                # coefficient for L1 penalty
                c1=config["L1_c"],
                # coefficient for L2 penalty
                c2=config["L2_c"],
                # stop earlier
                max_iterations=config["max_iterations"],
                # include transitions that are possible, but not observed
                all_possible_transitions=True,
            )
            entity_tagger.fit(X_train, y_train)

            entity_taggers[tag_name] = entity_tagger

            logger.debug("Training finished.")

        return entity_taggers
