from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Text, Tuple, Type

import numpy as np
import scipy.sparse

from rasa.nlu.tokenizers.tokenizer import Tokenizer
import rasa.shared.utils.io
import rasa.utils.io
import rasa.nlu.utils.pattern_utils as pattern_utils
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.featurizers.sparse_featurizer.sparse_featurizer import SparseFeaturizer
from rasa.shared.nlu.constants import TEXT, RESPONSE, ACTION_TEXT
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class RegexFeaturizer(SparseFeaturizer, GraphComponent):
    """Adds message features based on regex expressions."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            **SparseFeaturizer.get_default_config(),
            # text will be processed with case sensitive as default
            "case_sensitive": True,
            # use lookup tables to generate features
            "use_lookup_tables": True,
            # use regexes to generate features
            "use_regexes": True,
            # use match word boundaries for lookup table
            "use_word_boundaries": True,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        known_patterns: Optional[List[Dict[Text, Text]]] = None,
    ) -> None:
        """Constructs new features for regexes and lookup table using regex expressions.

        Args:
            config: Configuration for the component.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            known_patterns: Regex Patterns the component should pre-load itself with.
        """
        super().__init__(execution_context.node_name, config)

        self._model_storage = model_storage
        self._resource = resource

        self.known_patterns = known_patterns if known_patterns else []
        self.case_sensitive = config["case_sensitive"]
        self.finetune_mode = execution_context.is_finetuning

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RegexFeaturizer:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    def _merge_new_patterns(self, new_patterns: List[Dict[Text, Text]]) -> None:
        """Updates already known patterns with new patterns extracted from data.

        New patterns should always be added to the end of the existing
        patterns and the order of the existing patterns should not be disturbed.

        Args:
            new_patterns: Patterns extracted from training data and to be merged with
                known patterns.
        """
        pattern_name_index_map = {
            pattern["name"]: index for index, pattern in enumerate(self.known_patterns)
        }
        for extra_pattern in new_patterns:
            new_pattern_name = extra_pattern["name"]

            # Some patterns may have just new examples added
            # to them. These do not count as additional pattern.
            if new_pattern_name in pattern_name_index_map:
                self.known_patterns[pattern_name_index_map[new_pattern_name]][
                    "pattern"
                ] = extra_pattern["pattern"]
            else:
                self.known_patterns.append(extra_pattern)

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the component with all patterns extracted from training data."""
        patterns_from_data = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self._config["use_lookup_tables"],
            use_regexes=self._config["use_regexes"],
            use_word_boundaries=self._config["use_word_boundaries"],
        )
        if self.finetune_mode:
            # Merge patterns extracted from data with known patterns
            self._merge_new_patterns(patterns_from_data)
        else:
            self.known_patterns = patterns_from_data

        self._persist()
        return self._resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples (see parent class for full docstring)."""
        for example in training_data.training_examples:
            for attribute in [TEXT, RESPONSE, ACTION_TEXT]:
                self._text_features_with_regex(example, attribute)

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Featurizes all given messages in-place.

        Returns:
          the given list of messages which have been modified in-place
        """
        for message in messages:
            self._text_features_with_regex(message, TEXT)

        return messages

    def _text_features_with_regex(self, message: Message, attribute: Text) -> None:
        """Helper method to extract features and set them appropriately in the message.

        Args:
            message: Message to be featurized.
            attribute: Attribute of message to be featurized.
        """
        if self.known_patterns:
            sequence_features, sentence_features = self._features_for_patterns(
                message, attribute
            )

            self.add_features_to_message(
                sequence_features, sentence_features, attribute, message
            )

    def _features_for_patterns(
        self, message: Message, attribute: Text
    ) -> Tuple[Optional[scipy.sparse.coo_matrix], Optional[scipy.sparse.coo_matrix]]:
        """Checks which known patterns match the message.

        Given a sentence, returns a vector of {1,0} values indicating which
        regexes did match. Furthermore, if the
        message is tokenized, the function will mark all tokens with a dict
        relating the name of the regex to whether it was matched.

        Args:
            message: Message to be featurized.
            attribute: Attribute of message to be featurized.

        Returns:
           Token and sentence level features of message attribute.
        """
        # Attribute not set (e.g. response not present)
        if not message.get(attribute):
            return None, None

        tokens = message.get(TOKENS_NAMES[attribute], [])

        if not tokens:
            # nothing to featurize
            return None, None

        flags = 0  # default flag
        if not self.case_sensitive:
            flags = re.IGNORECASE

        sequence_length = len(tokens)

        num_patterns = len(self.known_patterns)

        sequence_features = np.zeros([sequence_length, num_patterns])
        sentence_features = np.zeros([1, num_patterns])

        for pattern_index, pattern in enumerate(self.known_patterns):
            matches = list(
                re.finditer(pattern["pattern"], message.get(attribute), flags=flags)
            )

            for token_index, t in enumerate(tokens):
                patterns = t.get("pattern", default={})
                patterns[pattern["name"]] = False

                for match in matches:
                    if t.start < match.end() and t.end > match.start():
                        patterns[pattern["name"]] = True
                        sequence_features[token_index][pattern_index] = 1.0
                        if attribute in [RESPONSE, TEXT, ACTION_TEXT]:
                            # sentence vector should contain all patterns
                            sentence_features[0][pattern_index] = 1.0

                t.set("pattern", patterns)

        return (
            scipy.sparse.coo_matrix(sequence_features),
            scipy.sparse.coo_matrix(sentence_features),
        )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> RegexFeaturizer:
        """Loads trained component (see parent class for full docstring)."""
        known_patterns = None

        try:
            with model_storage.read_from(resource) as model_dir:
                patterns_file_name = model_dir / "patterns.json"
                known_patterns = rasa.shared.utils.io.read_json_file(patterns_file_name)
        except (ValueError, FileNotFoundError):
            logger.warning(
                f"Failed to load `{cls.__class__.__name__}` from model storage. "
                f"Resource '{resource.name}' doesn't exist."
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            known_patterns=known_patterns,
        )

    def _persist(self) -> None:
        with self._model_storage.write_to(self._resource) as model_dir:
            regex_file = model_dir / "patterns.json"
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                regex_file, self.known_patterns
            )

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
