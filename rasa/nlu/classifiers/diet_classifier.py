from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union, TypeVar, Type

import numpy as np
import scipy.sparse
import tensorflow as tf

from rasa.exceptions import ModelNotFound
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.nlu.classifiers.classifier import IntentClassifier
import rasa.shared.utils.io
import rasa.nlu.utils.bilou_utils as bilou_utils
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils
from rasa.utils.tensorflow import rasa_layers
from rasa.utils.tensorflow.feature_array import (
    FeatureArray,
    serialize_nested_feature_arrays,
    deserialize_nested_feature_arrays,
)
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
)
from rasa.nlu.constants import TOKENS_NAMES, DEFAULT_TRANSFORMER_SIZE
from rasa.shared.nlu.constants import (
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
    TEXT,
    INTENT,
    INTENT_RESPONSE_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
    SPLIT_ENTITIES_BY_COMMA,
)
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.tensorflow.constants import (
    DROP_SMALL_LAST_BATCH,
    LABEL,
    IDS,
    HIDDEN_LAYERS_SIZES,
    RENORMALIZE_CONFIDENCES,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    TENSORBOARD_LOG_DIR,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    CONNECTION_DENSITY,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    AUTO,
    BALANCED,
    CROSS_ENTROPY,
    TENSORBOARD_LOG_LEVEL,
    CONCAT_DIMENSION,
    FEATURIZERS,
    CHECKPOINT_MODEL,
    SEQUENCE,
    SENTENCE,
    SEQUENCE_LENGTH,
    DENSE_DIMENSION,
    MASK,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    SOFTMAX,
    RUN_EAGERLY,
)

logger = logging.getLogger(__name__)

SPARSE = "sparse"
DENSE = "dense"
LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS

POSSIBLE_TAGS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP]

DIETClassifierT = TypeVar("DIETClassifierT", bound="DIETClassifier")


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    ],
    is_trainable=True,
)
class DIETClassifier(GraphComponent, IntentClassifier, EntityExtractorMixin):
    """A multi-task model for intent classification and entity extraction.

    DIET is Dual Intent and Entity Transformer.
    The architecture is based on a transformer which is shared for both tasks.
    A sequence of entity labels is predicted through a Conditional Random Field (CRF)
    tagging layer on top of the transformer output sequence corresponding to the
    input sequence of tokens. The transformer output for the ``__CLS__`` token and
    intent labels are embedded into a single semantic vector space. We use the
    dot-product loss to maximize the similarity with the target label and minimize
    similarities with negative samples.
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Featurizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            # ## Architecture of the used neural network
            # Hidden layer sizes for layers before the embedding layers for user message
            # and labels.
            # The number of hidden layers is equal to the length of the corresponding
            # list.
            HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []},
            # Whether to share the hidden layer weights between user message and labels.
            SHARE_HIDDEN_LAYERS: False,
            # Number of units in transformer
            TRANSFORMER_SIZE: DEFAULT_TRANSFORMER_SIZE,
            # Number of transformer layers
            NUM_TRANSFORMER_LAYERS: 2,
            # Number of attention heads in transformer
            NUM_HEADS: 4,
            # If 'True' use key relative embeddings in attention
            KEY_RELATIVE_ATTENTION: False,
            # If 'True' use value relative embeddings in attention
            VALUE_RELATIVE_ATTENTION: False,
            # Max position for relative embeddings. Only in effect if key- or value
            # relative attention are turned on
            MAX_RELATIVE_POSITION: 5,
            # Use a unidirectional or bidirectional encoder.
            UNIDIRECTIONAL_ENCODER: False,
            # ## Training parameters
            # Initial and final batch sizes:
            # Batch size will be linearly increased for each epoch.
            BATCH_SIZES: [64, 256],
            # Strategy used when creating batches.
            # Can be either 'sequence' or 'balanced'.
            BATCH_STRATEGY: BALANCED,
            # Number of epochs to train
            EPOCHS: 300,
            # Set random seed to any 'int' to get reproducible results
            RANDOM_SEED: None,
            # Initial learning rate for the optimizer
            LEARNING_RATE: 0.001,
            # ## Parameters for embeddings
            # Dimension size of embedding vectors
            EMBEDDING_DIMENSION: 20,
            # Dense dimension to use for sparse features.
            DENSE_DIMENSION: {TEXT: 128, LABEL: 20},
            # Default dimension to use for concatenating sequence and sentence features.
            CONCAT_DIMENSION: {TEXT: 128, LABEL: 20},
            # The number of incorrect labels. The algorithm will minimize
            # their similarity to the user input during training.
            NUM_NEG: 20,
            # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
            SIMILARITY_TYPE: AUTO,
            # The type of the loss function, either 'cross_entropy' or 'margin'.
            LOSS_TYPE: CROSS_ENTROPY,
            # Number of top intents for which confidences should be reported.
            # Set to 0 if confidences for all intents should be reported.
            RANKING_LENGTH: LABEL_RANKING_LENGTH,
            # Indicates how similar the algorithm should try to make embedding vectors
            # for correct labels.
            # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
            MAX_POS_SIM: 0.8,
            # Maximum negative similarity for incorrect labels.
            # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
            MAX_NEG_SIM: -0.4,
            # If 'True' the algorithm only minimizes maximum similarity over
            # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
            USE_MAX_NEG_SIM: True,
            # If 'True' scale loss inverse proportionally to the confidence
            # of the correct prediction
            SCALE_LOSS: False,
            # ## Regularization parameters
            # The scale of regularization
            REGULARIZATION_CONSTANT: 0.002,
            # The scale of how important is to minimize the maximum similarity
            # between embeddings of different labels,
            # used only if 'loss_type' is set to 'margin'.
            NEGATIVE_MARGIN_SCALE: 0.8,
            # Dropout rate for encoder
            DROP_RATE: 0.2,
            # Dropout rate for attention
            DROP_RATE_ATTENTION: 0,
            # Fraction of trainable weights in internal layers.
            CONNECTION_DENSITY: 0.2,
            # If 'True' apply dropout to sparse input tensors
            SPARSE_INPUT_DROPOUT: True,
            # If 'True' apply dropout to dense input tensors
            DENSE_INPUT_DROPOUT: True,
            # ## Evaluation parameters
            # How often calculate validation accuracy.
            # Small values may hurt performance.
            EVAL_NUM_EPOCHS: 20,
            # How many examples to use for hold out validation set
            # Large values may hurt performance, e.g. model accuracy.
            # Set to 0 for no validation.
            EVAL_NUM_EXAMPLES: 0,
            # ## Model config
            # If 'True' intent classification is trained and intent predicted.
            INTENT_CLASSIFICATION: True,
            # If 'True' named entity recognition is trained and entities predicted.
            ENTITY_RECOGNITION: True,
            # If 'True' random tokens of the input message will be masked and the model
            # should predict those tokens.
            MASKED_LM: False,
            # 'BILOU_flag' determines whether to use BILOU tagging or not.
            # If set to 'True' labelling is more rigorous, however more
            # examples per entity are required.
            # Rule of thumb: you should have more than 100 examples per entity.
            BILOU_FLAG: True,
            # If you want to use tensorboard to visualize training and validation
            # metrics, set this option to a valid output directory.
            TENSORBOARD_LOG_DIR: None,
            # Define when training metrics for tensorboard should be logged.
            # Either after every epoch or for every training step.
            # Valid values: 'epoch' and 'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",
            # Perform model checkpointing
            CHECKPOINT_MODEL: False,
            # Specify what features to use as sequence and sentence features
            # By default all features in the pipeline are used.
            FEATURIZERS: [],
            # Split entities by comma, this makes sense e.g. for a list of ingredients
            # in a recipie, but it doesn't make sense for the parts of an address
            SPLIT_ENTITIES_BY_COMMA: True,
            # If 'True' applies sigmoid on all similarity terms and adds
            # it to the loss function to ensure that similarity values are
            # approximately bounded. Used inside cross-entropy loss only.
            CONSTRAIN_SIMILARITIES: False,
            # Model confidence to be returned during inference. Currently, the only
            # possible value is `softmax`.
            MODEL_CONFIDENCE: SOFTMAX,
            # Determines whether the confidences of the chosen top intents should be
            # renormalized so that they sum up to 1. By default, we do not renormalize
            # and return the confidences for the top intents as is.
            # Note that renormalization only makes sense if confidences are generated
            # via `softmax`.
            RENORMALIZE_CONFIDENCES: False,
            # Determines whether to construct the model graph or not.
            # This is advantageous when the model is only trained or inferred for
            # a few steps, as the compilation of the graph tends to take more time than
            # running it. It is recommended to not adjust the optimization parameter.
            RUN_EAGERLY: False,
            # Determines whether the last batch should be dropped if it contains fewer
            # than half a batch size of examples
            DROP_SMALL_LAST_BATCH: False,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        model: Optional[RasaModel] = None,
        sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None,
    ) -> None:
        """Declare instance variables with default values."""
        if EPOCHS not in config:
            rasa.shared.utils.io.raise_warning(
                f"Please configure the number of '{EPOCHS}' in your configuration file."
                f" We will change the default value of '{EPOCHS}' in the future to 1. "
            )

        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

        self._check_config_parameters()

        # transform numbers to labels
        self.index_label_id_mapping = index_label_id_mapping or {}

        self._entity_tag_specs = entity_tag_specs

        self.model = model

        self.tmp_checkpoint_dir = None
        if self.component_config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())

        self._label_data: Optional[RasaModelData] = None
        self._data_example: Optional[Dict[Text, Dict[Text, List[FeatureArray]]]] = None

        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            self.component_config[SPLIT_ENTITIES_BY_COMMA],
            SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
        )

        self.finetune_mode = self._execution_context.is_finetuning
        self._sparse_feature_sizes = sparse_feature_sizes

    # init helpers
    def _check_masked_lm(self) -> None:
        if (
            self.component_config[MASKED_LM]
            and self.component_config[NUM_TRANSFORMER_LAYERS] == 0
        ):
            raise ValueError(
                f"If number of transformer layers is 0, "
                f"'{MASKED_LM}' option should be 'False'."
            )

    def _check_share_hidden_layers_sizes(self) -> None:
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            first_hidden_layer_sizes = next(
                iter(self.component_config[HIDDEN_LAYERS_SIZES].values())
            )
            # check that all hidden layer sizes are the same
            identical_hidden_layer_sizes = all(
                current_hidden_layer_sizes == first_hidden_layer_sizes
                for current_hidden_layer_sizes in self.component_config[
                    HIDDEN_LAYERS_SIZES
                ].values()
            )
            if not identical_hidden_layer_sizes:
                raise ValueError(
                    f"If hidden layer weights are shared, "
                    f"{HIDDEN_LAYERS_SIZES} must coincide."
                )

    def _check_config_parameters(self) -> None:
        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )

        self._check_masked_lm()
        self._check_share_hidden_layers_sizes()

        self.component_config = train_utils.update_confidence_type(
            self.component_config
        )

        train_utils.validate_configuration_settings(self.component_config)

        self.component_config = train_utils.update_similarity_type(
            self.component_config
        )
        self.component_config = train_utils.update_evaluation_parameters(
            self.component_config
        )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DIETClassifier:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    @property
    def label_key(self) -> Optional[Text]:
        """Return key if intent classification is activated."""
        return LABEL_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @property
    def label_sub_key(self) -> Optional[Text]:
        """Return sub key if intent classification is activated."""
        return LABEL_SUB_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @staticmethod
    def model_class() -> Type[RasaModel]:
        return DIET

    # training data helpers:
    @staticmethod
    def _label_id_index_mapping(
        training_data: TrainingData, attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary."""
        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        return {value: key for key, value in mapping.items()}

    def _create_entity_tag_specs(
        self, training_data: TrainingData
    ) -> List[EntityTagSpec]:
        """Create entity tag specifications with their respective tag id mappings."""
        _tag_specs = []

        for tag_name in POSSIBLE_TAGS:
            if self.component_config[BILOU_FLAG]:
                tag_id_index_mapping = bilou_utils.build_tag_id_dict(
                    training_data, tag_name
                )
            else:
                tag_id_index_mapping = self._tag_id_index_mapping_for(
                    tag_name, training_data
                )

            if tag_id_index_mapping:
                _tag_specs.append(
                    EntityTagSpec(
                        tag_name=tag_name,
                        tags_to_ids=tag_id_index_mapping,
                        ids_to_tags=self._invert_mapping(tag_id_index_mapping),
                        num_tags=len(tag_id_index_mapping),
                    )
                )

        return _tag_specs

    @staticmethod
    def _tag_id_index_mapping_for(
        tag_name: Text, training_data: TrainingData
    ) -> Optional[Dict[Text, int]]:
        """Create mapping from tag name to id."""
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            distinct_tags = training_data.entity_roles
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            distinct_tags = training_data.entity_groups
        else:
            distinct_tags = training_data.entities

        distinct_tags = distinct_tags - {NO_ENTITY_TAG} - {None}

        if not distinct_tags:
            return None

        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tags), 1)
        }
        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_dict[NO_ENTITY_TAG] = 0

        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    def _check_labels_features_exist(
        self, labels_example: List[Message], attribute: Text
    ) -> bool:
        """Checks if all labels have features set."""
        return all(
            label_example.features_present(
                attribute, self.component_config[FEATURIZERS]
            )
            for label_example in labels_example
        )

    def _extract_features(
        self, message: Message, attribute: Text
    ) -> Dict[Text, Union[scipy.sparse.spmatrix, np.ndarray]]:

        (
            sparse_sequence_features,
            sparse_sentence_features,
        ) = message.get_sparse_features(attribute, self.component_config[FEATURIZERS])
        dense_sequence_features, dense_sentence_features = message.get_dense_features(
            attribute, self.component_config[FEATURIZERS]
        )

        if dense_sequence_features is not None and sparse_sequence_features is not None:
            if (
                dense_sequence_features.features.shape[0]
                != sparse_sequence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sequence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )
        if dense_sentence_features is not None and sparse_sentence_features is not None:
            if (
                dense_sentence_features.features.shape[0]
                != sparse_sentence_features.features.shape[0]
            ):
                raise ValueError(
                    f"Sequence dimensions for sparse and dense sentence features "
                    f"don't coincide in '{message.get(TEXT)}'"
                    f"for attribute '{attribute}'."
                )

        # If we don't use the transformer and we don't want to do entity recognition,
        # to speed up training take only the sentence features as feature vector.
        # We would not make use of the sequence anyway in this setup. Carrying over
        # those features to the actual training process takes quite some time.
        if (
            self.component_config[NUM_TRANSFORMER_LAYERS] == 0
            and not self.component_config[ENTITY_RECOGNITION]
            and attribute not in [INTENT, INTENT_RESPONSE_KEY]
        ):
            sparse_sequence_features = None
            dense_sequence_features = None

        out = {}

        if sparse_sentence_features is not None:
            out[f"{SPARSE}_{SENTENCE}"] = sparse_sentence_features.features
        if sparse_sequence_features is not None:
            out[f"{SPARSE}_{SEQUENCE}"] = sparse_sequence_features.features
        if dense_sentence_features is not None:
            out[f"{DENSE}_{SENTENCE}"] = dense_sentence_features.features
        if dense_sequence_features is not None:
            out[f"{DENSE}_{SEQUENCE}"] = dense_sequence_features.features

        return out

    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """Checks if features have same dimensionality if hidden layers are shared."""
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_sentence_features = model_data.number_of_units(TEXT, SENTENCE)
            num_label_sentence_features = model_data.number_of_units(LABEL, SENTENCE)
            num_text_sequence_features = model_data.number_of_units(TEXT, SEQUENCE)
            num_label_sequence_features = model_data.number_of_units(LABEL, SEQUENCE)

            if (0 < num_text_sentence_features != num_label_sentence_features > 0) or (
                0 < num_text_sequence_features != num_label_sequence_features > 0
            ):
                raise ValueError(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def _extract_labels_precomputed_features(
        self, label_examples: List[Message], attribute: Text = INTENT
    ) -> Tuple[List[FeatureArray], List[FeatureArray]]:
        """Collects precomputed encodings."""
        features = defaultdict(list)

        for e in label_examples:
            label_features = self._extract_features(e, attribute)
            for feature_key, feature_value in label_features.items():
                features[feature_key].append(feature_value)
        sequence_features = []
        sentence_features = []
        for feature_name, feature_value in features.items():
            if SEQUENCE in feature_name:
                sequence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
            else:
                sentence_features.append(
                    FeatureArray(np.array(feature_value), number_of_dimensions=3)
                )
        return sequence_features, sentence_features

    @staticmethod
    def _compute_default_label_features(
        labels_example: List[Message],
    ) -> List[FeatureArray]:
        """Computes one-hot representation for the labels."""
        logger.debug("No label features found. Computing default label features.")

        eye_matrix = np.eye(len(labels_example), dtype=np.float32)
        # add sequence dimension to one-hot labels
        return [
            FeatureArray(
                np.array([np.expand_dims(a, 0) for a in eye_matrix]),
                number_of_dimensions=3,
            )
        ]

    def _create_label_data(
        self,
        training_data: TrainingData,
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.

        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """
        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]
        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            (
                sequence_features,
                sentence_features,
            ) = self._extract_labels_precomputed_features(labels_example, attribute)
        else:
            sequence_features = None
            sentence_features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL, SEQUENCE, sequence_features)
        label_data.add_features(LABEL, SENTENCE, sentence_features)
        if label_data.does_feature_not_exist(
            LABEL, SENTENCE
        ) and label_data.does_feature_not_exist(LABEL, SEQUENCE):
            raise ValueError(
                "No label features are present. Please check your configuration file."
            )

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [
                FeatureArray(
                    np.expand_dims(label_ids, -1),
                    number_of_dimensions=2,
                )
            ],
        )

        label_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[FeatureArray]:
        if self._label_data is None:
            return []

        feature_arrays = self._label_data.get(LABEL, SENTENCE)
        all_label_features = feature_arrays[0]
        return [
            FeatureArray(
                np.array([all_label_features[label_id] for label_id in label_ids]),
                number_of_dimensions=all_label_features.number_of_dimensions,
            )
        ]

    def _create_model_data(
        self,
        training_data: List[Message],
        label_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = None,
        training: bool = True,
    ) -> RasaModelData:
        """Prepare data for training and create a RasaModelData object."""
        from rasa.utils.tensorflow import model_data_utils

        attributes_to_consider = [TEXT]
        if training and self.component_config[INTENT_CLASSIFICATION]:
            # we don't have any intent labels during prediction, just add them during
            # training
            attributes_to_consider.append(label_attribute)
        if (
            training
            and self.component_config[ENTITY_RECOGNITION]
            and self._entity_tag_specs
        ):
            # Add entities as labels only during training and only if there was
            # training data added for entities with DIET configured to predict entities.
            attributes_to_consider.append(ENTITIES)

        if training and label_attribute is not None:
            # only use those training examples that have the label_attribute set
            # during training
            training_data = [
                example for example in training_data if label_attribute in example.data
            ]

        training_data = [
            message
            for message in training_data
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]

        if not training_data:
            # no training data are present to train
            return RasaModelData()

        (
            features_for_examples,
            sparse_feature_sizes,
        ) = model_data_utils.featurize_training_examples(
            training_data,
            attributes_to_consider,
            entity_tag_specs=self._entity_tag_specs,
            featurizers=self.component_config[FEATURIZERS],
            bilou_tagging=self.component_config[BILOU_FLAG],
        )
        attribute_data, _ = model_data_utils.convert_to_data_format(
            features_for_examples, consider_dialogue_dimension=False
        )

        model_data = RasaModelData(
            label_key=self.label_key, label_sub_key=self.label_sub_key
        )
        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        # Current implementation doesn't yet account for updating sparse
        # feature sizes of label attributes. That's why we remove them.
        sparse_feature_sizes = self._remove_label_sparse_feature_sizes(
            sparse_feature_sizes=sparse_feature_sizes, label_attribute=label_attribute
        )
        model_data.add_sparse_feature_sizes(sparse_feature_sizes)

        self._add_label_features(
            model_data, training_data, label_attribute, label_id_dict, training
        )

        # make sure all keys are in the same order during training and prediction
        # as we rely on the order of key and sub-key when constructing the actual
        # tensors from the model data
        model_data.sort()

        return model_data

    @staticmethod
    def _remove_label_sparse_feature_sizes(
        sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        label_attribute: Optional[Text] = None,
    ) -> Dict[Text, Dict[Text, List[int]]]:

        if label_attribute in sparse_feature_sizes:
            del sparse_feature_sizes[label_attribute]
        return sparse_feature_sizes

    def _add_label_features(
        self,
        model_data: RasaModelData,
        training_data: List[Message],
        label_attribute: Text,
        label_id_dict: Dict[Text, int],
        training: bool = True,
    ) -> None:
        label_ids = []
        if training and self.component_config[INTENT_CLASSIFICATION]:
            for example in training_data:
                if example.get(label_attribute):
                    label_ids.append(label_id_dict[example.get(label_attribute)])
            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [
                    FeatureArray(
                        np.expand_dims(label_ids, -1),
                        number_of_dimensions=2,
                    )
                ],
            )

        if (
            label_attribute
            and model_data.does_feature_not_exist(label_attribute, SENTENCE)
            and model_data.does_feature_not_exist(label_attribute, SEQUENCE)
        ):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                LABEL, SENTENCE, self._use_default_label_features(np.array(label_ids))
            )

        # as label_attribute can have different values, e.g. INTENT or RESPONSE,
        # copy over the features to the LABEL key to make
        # it easier to access the label features inside the model itself
        model_data.update_key(label_attribute, SENTENCE, LABEL, SENTENCE)
        model_data.update_key(label_attribute, SEQUENCE, LABEL, SEQUENCE)
        model_data.update_key(label_attribute, MASK, LABEL, MASK)

        model_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """
        if (
            self.component_config[BILOU_FLAG]
            and self.component_config[ENTITY_RECOGNITION]
        ):
            bilou_utils.apply_bilou_schema(training_data)

        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=INTENT
        )

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT
        )

        self._entity_tag_specs = self._create_entity_tag_specs(training_data)

        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )
        model_data = self._create_model_data(
            training_data.nlu_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        return len(np.unique(model_data.get(LABEL_KEY, LABEL_SUB_KEY))) >= 2

    def train(self, training_data: TrainingData) -> Resource:
        """Train the embedding intent classifier on a data set."""
        model_data = self.preprocess_train_data(training_data)
        if model_data.is_empty():
            logger.debug(
                f"Cannot train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the classifier."
            )
            return self._resource

        if not self.model and self.finetune_mode:
            raise rasa.shared.exceptions.InvalidParameterException(
                f"{self.__class__.__name__} was instantiated "
                f"with `model=None` and `finetune_mode=True`. "
                f"This is not a valid combination as the component "
                f"needs an already instantiated and trained model "
                f"to continue training in finetune mode."
            )

        if self.component_config.get(INTENT_CLASSIFICATION):
            if not self._check_enough_labels(model_data):
                logger.error(
                    f"Cannot train '{self.__class__.__name__}'. "
                    f"Need at least 2 different intent classes. "
                    f"Skipping training of classifier."
                )
                return self._resource
        if self.component_config.get(ENTITY_RECOGNITION):
            self.check_correct_entity_annotations(training_data)

        # keep one example for persisting and loading
        self._data_example = model_data.first_data_example()

        if not self.finetune_mode:
            # No pre-trained model to load from. Create a new instance of the model.
            self.model = self._instantiate_model_class(model_data)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    self.component_config[LEARNING_RATE]
                ),
                run_eagerly=self.component_config[RUN_EAGERLY],
            )
        else:
            if self.model is None:
                raise ModelNotFound("Model could not be found. ")

            self.model.adjust_for_incremental_training(
                data_example=self._data_example,
                new_sparse_feature_sizes=model_data.get_sparse_feature_sizes(),
                old_sparse_feature_sizes=self._sparse_feature_sizes,
            )
        self._sparse_feature_sizes = model_data.get_sparse_feature_sizes()

        data_generator, validation_data_generator = train_utils.create_data_generators(
            model_data,
            self.component_config[BATCH_SIZES],
            self.component_config[EPOCHS],
            self.component_config[BATCH_STRATEGY],
            self.component_config[EVAL_NUM_EXAMPLES],
            self.component_config[RANDOM_SEED],
            drop_small_last_batch=self.component_config[DROP_SMALL_LAST_BATCH],
        )
        callbacks = train_utils.create_common_callbacks(
            self.component_config[EPOCHS],
            self.component_config[TENSORBOARD_LOG_DIR],
            self.component_config[TENSORBOARD_LOG_LEVEL],
            self.tmp_checkpoint_dir,
        )

        self.model.fit(
            data_generator,
            epochs=self.component_config[EPOCHS],
            validation_data=validation_data_generator,
            validation_freq=self.component_config[EVAL_NUM_EPOCHS],
            callbacks=callbacks,
            verbose=False,
            shuffle=False,  # we use custom shuffle inside data generator
        )

        self.persist()

        return self._resource

    # process helpers
    def _predict(
        self, message: Message
    ) -> Optional[Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]]:
        if self.model is None:
            logger.debug(
                f"There is no trained model for '{self.__class__.__name__}': The "
                f"component is either not trained or didn't receive enough training "
                f"data."
            )
            return None

        # create session data from message and convert it into a batch of 1
        model_data = self._create_model_data([message], training=False)
        if model_data.is_empty():
            return None
        return self.model.run_inference(model_data)

    def _predict_label(
        self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""
        label: Dict[Text, Any] = {"name": None, "confidence": 0.0}
        label_ranking: List[Dict[Text, Any]] = []

        if predict_out is None:
            return label, label_ranking

        message_sim = predict_out["i_scores"]
        message_sim = message_sim.flatten()  # sim is a matrix

        # if X contains all zeros do not predict some label
        if message_sim.size == 0:
            return label, label_ranking

        # rank the confidences
        ranking_length = self.component_config[RANKING_LENGTH]
        renormalize = (
            self.component_config[RENORMALIZE_CONFIDENCES]
            and self.component_config[MODEL_CONFIDENCE] == SOFTMAX
        )
        ranked_label_indices, message_sim = train_utils.rank_and_mask(
            message_sim, ranking_length=ranking_length, renormalize=renormalize
        )

        # construct the label and ranking
        casted_message_sim: List[float] = message_sim.tolist()  # np.float to float
        top_label_idx = ranked_label_indices[0]
        label = {
            "name": self.index_label_id_mapping[top_label_idx],
            "confidence": casted_message_sim[top_label_idx],
        }

        ranking = [(idx, casted_message_sim[idx]) for idx in ranked_label_indices]
        label_ranking = [
            {"name": self.index_label_id_mapping[label_idx], "confidence": score}
            for label_idx, score in ranking
        ]

        return label, label_ranking

    def _predict_entities(
        self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        if predict_out is None:
            return []

        predicted_tags, confidence_values = train_utils.entity_label_to_tags(
            predict_out, self._entity_tag_specs, self.component_config[BILOU_FLAG]
        )

        entities = self.convert_predictions_into_entities(
            message.get(TEXT),
            message.get(TOKENS_NAMES[TEXT], []),
            predicted_tags,
            self.split_entities_config,
            confidence_values,
        )

        entities = self.add_extractor_name(entities)
        entities = message.get(ENTITIES, []) + entities

        return entities

    def process(self, messages: List[Message]) -> List[Message]:
        """Augments the message with intents, entities, and diagnostic data."""
        for message in messages:
            out = self._predict(message)

            if self.component_config[INTENT_CLASSIFICATION]:
                label, label_ranking = self._predict_label(out)

                message.set(INTENT, label, add_to_output=True)
                message.set("intent_ranking", label_ranking, add_to_output=True)

            if self.component_config[ENTITY_RECOGNITION]:
                entities = self._predict_entities(out, message)

                message.set(ENTITIES, entities, add_to_output=True)

            if out and self._execution_context.should_add_diagnostic_data:
                message.add_diagnostic_data(
                    self._execution_context.node_name, out.get(DIAGNOSTIC_DATA)
                )

        return messages

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        if self.model is None:
            return None

        with self._model_storage.write_to(self._resource) as model_path:
            file_name = self.__class__.__name__
            tf_model_file = model_path / f"{file_name}.tf_model"

            rasa.shared.utils.io.create_directory_for_file(tf_model_file)

            if self.component_config[CHECKPOINT_MODEL] and self.tmp_checkpoint_dir:
                self.model.load_weights(self.tmp_checkpoint_dir / "checkpoint.tf_model")
                # Save an empty file to flag that this model has been
                # produced using checkpointing
                checkpoint_marker = model_path / f"{file_name}.from_checkpoint.pkl"
                checkpoint_marker.touch()

            self.model.save(str(tf_model_file))

            # save data example
            serialize_nested_feature_arrays(
                self._data_example,
                model_path / f"{file_name}.data_example.st",
                model_path / f"{file_name}.data_example_metadata.json",
            )
            # save label data
            serialize_nested_feature_arrays(
                dict(self._label_data.data) if self._label_data is not None else {},
                model_path / f"{file_name}.label_data.st",
                model_path / f"{file_name}.label_data_metadata.json",
            )

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.sparse_feature_sizes.json",
                self._sparse_feature_sizes,
            )
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.index_label_id_mapping.json",
                self.index_label_id_mapping,
            )

            entity_tag_specs = (
                [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
                if self._entity_tag_specs
                else []
            )
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.entity_tag_specs.json", entity_tag_specs
            )

    @classmethod
    def load(
        cls: Type[DIETClassifierT],
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> DIETClassifierT:
        """Loads a policy from the storage (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_path:
                return cls._load(
                    model_path, config, model_storage, resource, execution_context
                )
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource, execution_context)

    @classmethod
    def _load(
        cls: Type[DIETClassifierT],
        model_path: Path,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DIETClassifierT:
        """Loads the trained model from the provided directory."""
        (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            data_example,
            sparse_feature_sizes,
        ) = cls._load_from_files(model_path)

        config = train_utils.update_confidence_type(config)
        config = train_utils.update_similarity_type(config)

        model = cls._load_model(
            entity_tag_specs,
            label_data,
            config,
            data_example,
            model_path,
            finetune_mode=execution_context.is_finetuning,
        )

        return cls(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
            index_label_id_mapping=index_label_id_mapping,
            entity_tag_specs=entity_tag_specs,
            model=model,
            sparse_feature_sizes=sparse_feature_sizes,
        )

    @classmethod
    def _load_from_files(
        cls, model_path: Path
    ) -> Tuple[
        Dict[int, Text],
        List[EntityTagSpec],
        RasaModelData,
        Dict[Text, Dict[Text, List[FeatureArray]]],
        Dict[Text, Dict[Text, List[int]]],
    ]:
        file_name = cls.__name__

        # load data example
        data_example = deserialize_nested_feature_arrays(
            str(model_path / f"{file_name}.data_example.st"),
            str(model_path / f"{file_name}.data_example_metadata.json"),
        )
        # load label data
        loaded_label_data = deserialize_nested_feature_arrays(
            str(model_path / f"{file_name}.label_data.st"),
            str(model_path / f"{file_name}.label_data_metadata.json"),
        )
        label_data = RasaModelData(data=loaded_label_data)

        sparse_feature_sizes = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.sparse_feature_sizes.json"
        )
        index_label_id_mapping = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.index_label_id_mapping.json"
        )
        entity_tag_specs = rasa.shared.utils.io.read_json_file(
            model_path / f"{file_name}.entity_tag_specs.json"
        )
        entity_tag_specs = [
            EntityTagSpec(
                tag_name=tag_spec["tag_name"],
                ids_to_tags={
                    int(key): value for key, value in tag_spec["ids_to_tags"].items()
                },
                tags_to_ids={
                    key: int(value) for key, value in tag_spec["tags_to_ids"].items()
                },
                num_tags=tag_spec["num_tags"],
            )
            for tag_spec in entity_tag_specs
        ]

        index_label_id_mapping = {
            int(key): value for key, value in index_label_id_mapping.items()
        }

        return (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            data_example,
            sparse_feature_sizes,
        )

    @classmethod
    def _load_model(
        cls,
        entity_tag_specs: List[EntityTagSpec],
        label_data: RasaModelData,
        config: Dict[Text, Any],
        data_example: Dict[Text, Dict[Text, List[FeatureArray]]],
        model_path: Path,
        finetune_mode: bool = False,
    ) -> "RasaModel":
        file_name = cls.__name__
        tf_model_file = model_path / f"{file_name}.tf_model"

        label_key = LABEL_KEY if config[INTENT_CLASSIFICATION] else None
        label_sub_key = LABEL_SUB_KEY if config[INTENT_CLASSIFICATION] else None

        model_data_example = RasaModelData(
            label_key=label_key, label_sub_key=label_sub_key, data=data_example
        )

        model = cls._load_model_class(
            tf_model_file,
            model_data_example,
            label_data,
            entity_tag_specs,
            config,
            finetune_mode=finetune_mode,
        )

        return model

    @classmethod
    def _load_model_class(
        cls,
        tf_model_file: Text,
        model_data_example: RasaModelData,
        label_data: RasaModelData,
        entity_tag_specs: List[EntityTagSpec],
        config: Dict[Text, Any],
        finetune_mode: bool,
    ) -> "RasaModel":

        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )

        return cls.model_class().load(
            tf_model_file,
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            entity_tag_specs=entity_tag_specs,
            config=copy.deepcopy(config),
            finetune_mode=finetune_mode,
        )

    def _instantiate_model_class(self, model_data: RasaModelData) -> "RasaModel":
        return self.model_class()(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            entity_tag_specs=self._entity_tag_specs,
            config=self.component_config,
        )


class DIET(TransformerRasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        label_data: RasaModelData,
        entity_tag_specs: Optional[List[EntityTagSpec]],
        config: Dict[Text, Any],
    ) -> None:
        # create entity tag spec before calling super otherwise building the model
        # will fail
        super().__init__("DIET", config, data_signature, label_data)
        self._entity_tag_specs = self._ordered_tag_specs(entity_tag_specs)

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if TEXT in feature_name
        }

        # tf training
        self._create_metrics()
        self._update_metrics_to_log()

        # needed for efficient prediction
        self.all_labels_embed: Optional[tf.Tensor] = None

        self._prepare_layers()

    @staticmethod
    def _ordered_tag_specs(
        entity_tag_specs: Optional[List[EntityTagSpec]],
    ) -> List[EntityTagSpec]:
        """Ensure that order of entity tag specs matches CRF layer order."""
        if entity_tag_specs is None:
            return []

        crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

        ordered_tag_spec = []

        for tag_name in crf_order:
            for tag_spec in entity_tag_specs:
                if tag_name == tag_spec.tag_name:
                    ordered_tag_spec.append(tag_spec)

        return ordered_tag_spec

    def _check_data(self) -> None:
        if TEXT not in self.data_signature:
            raise InvalidConfigException(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL not in self.data_signature:
                raise InvalidConfigException(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )

            if self.config[SHARE_HIDDEN_LAYERS]:
                different_sentence_signatures = False
                different_sequence_signatures = False
                if (
                    SENTENCE in self.data_signature[TEXT]
                    and SENTENCE in self.data_signature[LABEL]
                ):
                    different_sentence_signatures = (
                        self.data_signature[TEXT][SENTENCE]
                        != self.data_signature[LABEL][SENTENCE]
                    )
                if (
                    SEQUENCE in self.data_signature[TEXT]
                    and SEQUENCE in self.data_signature[LABEL]
                ):
                    different_sequence_signatures = (
                        self.data_signature[TEXT][SEQUENCE]
                        != self.data_signature[LABEL][SEQUENCE]
                    )

                if different_sentence_signatures or different_sequence_signatures:
                    raise ValueError(
                        "If hidden layer weights are shared, data signatures "
                        "for text_features and label_features must coincide."
                    )

        if self.config[ENTITY_RECOGNITION] and (
            ENTITIES not in self.data_signature
            or ENTITY_ATTRIBUTE_TYPE not in self.data_signature[ENTITIES]
        ):
            logger.debug(
                f"You specified '{self.__class__.__name__}' to train entities, but "
                f"no entities are present in the training data. Skipping training of "
                f"entities."
            )
            self.config[ENTITY_RECOGNITION] = False

    def _create_metrics(self) -> None:
        # self.metrics will have the same order as they are created
        # so create loss metrics first to output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.entity_group_loss = tf.keras.metrics.Mean(name="g_loss")
        self.entity_role_loss = tf.keras.metrics.Mean(name="r_loss")
        # create accuracy metrics second to output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.intent_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")
        self.entity_group_f1 = tf.keras.metrics.Mean(name="g_f1")
        self.entity_role_f1 = tf.keras.metrics.Mean(name="r_f1")

    def _update_metrics_to_log(self) -> None:
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log.append("i_acc")
            if debug_log_level:
                self.metrics_to_log.append("i_loss")
        if self.config[ENTITY_RECOGNITION]:
            for tag_spec in self._entity_tag_specs:
                if tag_spec.num_tags != 0:
                    name = tag_spec.tag_name
                    self.metrics_to_log.append(f"{name[0]}_f1")
                    if debug_log_level:
                        self.metrics_to_log.append(f"{name[0]}_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {
            "t": "total",
            "i": "intent",
            "e": "entity",
            "m": "mask",
            "r": "role",
            "g": "group",
        }
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _prepare_layers(self) -> None:
        # For user text, prepare layers that combine different feature types, embed
        # everything using a transformer and optionally also do masked language
        # modeling.
        self.text_name = TEXT
        self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ] = rasa_layers.RasaSequenceLayer(
            self.text_name, self.data_signature[self.text_name], self.config
        )
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_loss(self.text_name)

        # Intent labels are treated similarly to user text but without the transformer,
        # without masked language modelling, and with no dropout applied to the
        # individual features, only to the overall label embedding after all label
        # features have been combined.
        if self.config[INTENT_CLASSIFICATION]:
            self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL

            # disable input dropout applied to sparse and dense label features
            label_config = self.config.copy()
            label_config.update(
                {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
            )

            self._tf_layers[
                f"feature_combining_layer.{self.label_name}"
            ] = rasa_layers.RasaFeatureCombiningLayer(
                self.label_name, self.label_signature[self.label_name], label_config
            )

            self._prepare_ffnn_layer(
                self.label_name,
                self.config[HIDDEN_LAYERS_SIZES][self.label_name],
                self.config[DROP_RATE],
            )

            self._prepare_label_classification_layers(predictor_attribute=TEXT)

        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_mask_lm_loss(self, name: Text) -> None:
        # for embedding predicted tokens at masked positions
        self._prepare_embed_layers(f"{name}_lm_mask")

        # for embedding the true tokens that got masked
        self._prepare_embed_layers(f"{name}_golden_token")

        # mask loss is additional loss
        # set scaling to False, so that it doesn't overpower other losses
        self._prepare_dot_product_loss(f"{name}_mask", scale_loss=False)

    def _create_bow(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sequence_feature_lengths: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:

        x, _ = self._tf_layers[f"feature_combining_layer.{name}"](
            (sequence_features, sentence_features, sequence_feature_lengths),
            training=self._training,
        )

        # convert to bag-of-words by summing along the sequence dimension
        x = tf.reduce_sum(x, axis=1)

        return self._tf_layers[f"ffnn.{name}"](x, self._training)

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, LABEL
        )

        x = self._create_bow(
            self.tf_label_data[LABEL][SEQUENCE],
            self.tf_label_data[LABEL][SENTENCE],
            sequence_feature_lengths,
            self.label_name,
        )
        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    def _mask_loss(
        self,
        outputs: tf.Tensor,
        inputs: tf.Tensor,
        seq_ids: tf.Tensor,
        mlm_mask_boolean: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:
        # make sure there is at least one element in the mask
        mlm_mask_boolean = tf.cond(
            tf.reduce_any(mlm_mask_boolean),
            lambda: mlm_mask_boolean,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(mlm_mask_boolean)),
        )

        mlm_mask_boolean = tf.squeeze(mlm_mask_boolean, -1)

        # Pick elements that were masked, throwing away the batch & sequence dimension
        # and effectively switching from shape (batch_size, sequence_length, units) to
        # (num_masked_elements, units).
        outputs = tf.boolean_mask(outputs, mlm_mask_boolean)
        inputs = tf.boolean_mask(inputs, mlm_mask_boolean)
        ids = tf.boolean_mask(seq_ids, mlm_mask_boolean)

        tokens_predicted_embed = self._tf_layers[f"embed.{name}_lm_mask"](outputs)
        tokens_true_embed = self._tf_layers[f"embed.{name}_golden_token"](inputs)

        # To limit the otherwise computationally expensive loss calculation, we
        # constrain the label space in MLM (i.e. token space) to only those tokens that
        # were masked in this batch. Hence the reduced list of token embeddings
        # (tokens_true_embed) and the reduced list of labels (ids) are passed as
        # all_labels_embed and all_labels, respectively. In the future, we could be less
        # restrictive and construct a slightly bigger label space which could include
        # tokens not masked in the current batch too.
        return self._tf_layers[f"loss.{name}_mask"](
            inputs_embed=tokens_predicted_embed,
            labels_embed=tokens_true_embed,
            labels=ids,
            all_labels_embed=tokens_true_embed,
            all_labels=ids,
        )

    def _calculate_label_loss(
        self, text_features: tf.Tensor, label_features: tf.Tensor, label_ids: tf.Tensor
    ) -> tf.Tensor:
        all_label_ids, all_labels_embed = self._create_all_labels()

        text_embed = self._tf_layers[f"embed.{TEXT}"](text_features)
        label_embed = self._tf_layers[f"embed.{LABEL}"](label_features)

        return self._tf_layers[f"loss.{LABEL}"](
            text_embed, label_embed, label_ids, all_labels_embed, all_label_ids
        )

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )

        (
            text_transformed,
            text_in,
            mask_combined_sequence_sentence,
            text_seq_ids,
            mlm_mask_boolean_text,
            _,
        ) = self._tf_layers[f"sequence_layer.{self.text_name}"](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        losses = []

        # Lengths of sequences in case of sentence-level features are always 1, but they
        # can effectively be 0 if sentence-level features aren't present.
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )

        combined_sequence_sentence_feature_lengths = (
            sequence_feature_lengths + sentence_feature_lengths
        )

        if self.config[MASKED_LM] and self._training:
            loss, acc = self._mask_loss(
                text_transformed, text_in, text_seq_ids, mlm_mask_boolean_text, TEXT
            )
            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        if self.config[INTENT_CLASSIFICATION]:
            loss = self._batch_loss_intent(
                combined_sequence_sentence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )
            losses.append(loss)

        if self.config[ENTITY_RECOGNITION]:
            losses += self._batch_loss_entities(
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )

        return tf.math.add_n(losses)

    def _batch_loss_intent(
        self,
        combined_sequence_sentence_feature_lengths_text: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> tf.Tensor:
        # get sentence features vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths_text
        )

        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        label = self._create_bow(
            tf_batch_data[LABEL][SEQUENCE],
            tf_batch_data[LABEL][SENTENCE],
            sequence_feature_lengths_label,
            self.label_name,
        )
        loss, acc = self._calculate_label_loss(sentence_vector, label, label_ids)

        self._update_label_metrics(loss, acc)

        return loss

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:

        self.intent_loss.update_state(loss)
        self.intent_acc.update_state(acc)

    def _batch_loss_entities(
        self,
        mask_combined_sequence_sentence: tf.Tensor,
        sequence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> List[tf.Tensor]:
        losses = []

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            if tag_spec.num_tags == 0:
                continue

            tag_ids = tf_batch_data[ENTITIES][tag_spec.tag_name][0]
            # add a zero (no entity) for the sentence features to match the shape of
            # inputs
            tag_ids = tf.pad(tag_ids, [[0, 0], [0, 1], [0, 0]])

            loss, f1, _logits = self._calculate_entity_loss(
                text_transformed,
                tag_ids,
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                tag_spec.tag_name,
                entity_tags,
            )

            if tag_spec.tag_name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(tag_ids[:, :, 0], tf.int32), depth=tag_spec.num_tags
                )

            self._update_entity_metrics(loss, f1, tag_spec.tag_name)

            losses.append(loss)

        return losses

    def _update_entity_metrics(
        self, loss: tf.Tensor, f1: tf.Tensor, tag_name: Text
    ) -> None:
        if tag_name == ENTITY_ATTRIBUTE_TYPE:
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            self.entity_group_loss.update_state(loss)
            self.entity_group_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_ROLE:
            self.entity_role_loss.update_state(loss)
            self.entity_role_f1.update_state(f1)

    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction."""
        if self.config[INTENT_CLASSIFICATION]:
            _, self.all_labels_embed = self._create_all_labels()

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )

        text_transformed, _, _, _, _, attention_weights = self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )
        predictions = {
            DIAGNOSTIC_DATA: {
                "attention_weights": attention_weights,
                "text_transformed": text_transformed,
            }
        }

        if self.config[INTENT_CLASSIFICATION]:
            predictions.update(
                self._batch_predict_intents(
                    sequence_feature_lengths + sentence_feature_lengths,
                    text_transformed,
                )
            )

        if self.config[ENTITY_RECOGNITION]:
            predictions.update(
                self._batch_predict_entities(sequence_feature_lengths, text_transformed)
            )

        return predictions

    def _batch_predict_entities(
        self, sequence_feature_lengths: tf.Tensor, text_transformed: tf.Tensor
    ) -> Dict[Text, tf.Tensor]:
        predictions: Dict[Text, tf.Tensor] = {}

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            # skip crf layer if it was not trained
            if tag_spec.num_tags == 0:
                continue

            name = tag_spec.tag_name
            _input = text_transformed

            if entity_tags is not None:
                _tags = self._tf_layers[f"embed.{name}.tags"](entity_tags)
                _input = tf.concat([_input, _tags], axis=-1)

            _logits = self._tf_layers[f"embed.{name}.logits"](_input)
            pred_ids, confidences = self._tf_layers[f"crf.{name}"](
                _logits, sequence_feature_lengths
            )

            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

            if name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(pred_ids, tf.int32), depth=tag_spec.num_tags
                )

        return predictions

    def _batch_predict_intents(
        self,
        combined_sequence_sentence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
    ) -> Dict[Text, tf.Tensor]:

        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        # get sentence feature vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths
        )
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )

        return {"i_scores": scores}
