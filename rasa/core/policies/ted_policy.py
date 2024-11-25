from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict
import contextlib
from typing import Any, List, Optional, Text, Dict, Tuple, Union, Type

import numpy as np
import tensorflow as tf

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.exceptions import ModelNotFound
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.extractors.extractor import EntityTagSpec, EntityExtractorMixin
import rasa.core.actions.action
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import (
    ACTION_TEXT,
    ACTION_NAME,
    INTENT,
    TEXT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_TAGS,
    EXTRACTOR,
    SPLIT_ENTITIES_BY_COMMA,
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
)
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData
from rasa.core.constants import (
    DIALOGUE,
    POLICY_MAX_HISTORY,
    DEFAULT_MAX_HISTORY,
    DEFAULT_POLICY_PRIORITY,
    POLICY_PRIORITY,
)
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.core.constants import ACTIVE_LOOP, SLOTS, ACTION_LISTEN_NAME
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.events import EntitiesAdded, Event
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import (
    Features,
    save_features,
    load_features,
)
import rasa.shared.utils.io
import rasa.utils.io
from rasa.utils import train_utils
from rasa.utils.tensorflow.feature_array import (
    FeatureArray,
    serialize_nested_feature_arrays,
    deserialize_nested_feature_arrays,
)
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow import rasa_layers
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature, Data
from rasa.utils.tensorflow.model_data_utils import convert_to_data_format
from rasa.utils.tensorflow.constants import (
    LABEL,
    IDS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    RANKING_LENGTH,
    RENORMALIZE_CONFIDENCES,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    DROP_RATE_DIALOGUE,
    DROP_RATE_LABEL,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    CONNECTION_DENSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    CROSS_ENTROPY,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CHECKPOINT_MODEL,
    ENCODING_DIMENSION,
    UNIDIRECTIONAL_ENCODER,
    SEQUENCE,
    SENTENCE,
    SEQUENCE_LENGTH,
    DENSE_DIMENSION,
    CONCAT_DIMENSION,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    MASK,
    HIDDEN_LAYERS_SIZES,
    FEATURIZERS,
    ENTITY_RECOGNITION,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    SOFTMAX,
    BILOU_FLAG,
    EPOCH_OVERRIDE,
    USE_GPU,
)

logger = logging.getLogger(__name__)

E2E_CONFIDENCE_THRESHOLD = "e2e_confidence_threshold"
LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS
LENGTH = "length"
INDICES = "indices"
SENTENCE_FEATURES_TO_ENCODE = [INTENT, TEXT, ACTION_NAME, ACTION_TEXT]
SEQUENCE_FEATURES_TO_ENCODE = [TEXT, ACTION_TEXT, f"{LABEL}_{ACTION_TEXT}"]
LABEL_FEATURES_TO_ENCODE = [
    f"{LABEL}_{ACTION_NAME}",
    f"{LABEL}_{ACTION_TEXT}",
    f"{LABEL}_{INTENT}",
]
STATE_LEVEL_FEATURES = [ENTITIES, SLOTS, ACTIVE_LOOP]
PREDICTION_FEATURES = STATE_LEVEL_FEATURES + SENTENCE_FEATURES_TO_ENCODE + [DIALOGUE]


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True
)
class TEDPolicy(Policy):
    """Transformer Embedding Dialogue (TED) Policy.

    The model architecture is described in
    detail in https://arxiv.org/abs/1910.00486.
    In summary, the architecture comprises of the
    following steps:
        - concatenate user input (user intent and entities), previous system actions,
          slots and active forms for each time step into an input vector to
          pre-transformer embedding layer;
        - feed it to transformer;
        - apply a dense layer to the output of the transformer to get embeddings of a
          dialogue for each time step;
        - apply a dense layer to create embeddings for system actions for each time
          step;
        - calculate the similarity between the dialogue embedding and embedded system
          actions. This step is based on the StarSpace
          (https://arxiv.org/abs/1709.03856) idea.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            # ## Architecture of the used neural network
            # Hidden layer sizes for layers before the embedding layers for user message
            # and labels.
            # The number of hidden layers is equal to the length of the corresponding
            # list.
            HIDDEN_LAYERS_SIZES: {
                TEXT: [],
                ACTION_TEXT: [],
                f"{LABEL}_{ACTION_TEXT}": [],
            },
            # Dense dimension to use for sparse features.
            DENSE_DIMENSION: {
                TEXT: 128,
                ACTION_TEXT: 128,
                f"{LABEL}_{ACTION_TEXT}": 128,
                INTENT: 20,
                ACTION_NAME: 20,
                f"{LABEL}_{ACTION_NAME}": 20,
                ENTITIES: 20,
                SLOTS: 20,
                ACTIVE_LOOP: 20,
            },
            # Default dimension to use for concatenating sequence and sentence features.
            CONCAT_DIMENSION: {
                TEXT: 128,
                ACTION_TEXT: 128,
                f"{LABEL}_{ACTION_TEXT}": 128,
            },
            # Dimension size of embedding vectors before the dialogue transformer
            # encoder.
            ENCODING_DIMENSION: 50,
            # Number of units in transformer encoders
            TRANSFORMER_SIZE: {
                TEXT: 128,
                ACTION_TEXT: 128,
                f"{LABEL}_{ACTION_TEXT}": 128,
                DIALOGUE: 128,
            },
            # Number of layers in transformer encoders
            NUM_TRANSFORMER_LAYERS: {
                TEXT: 1,
                ACTION_TEXT: 1,
                f"{LABEL}_{ACTION_TEXT}": 1,
                DIALOGUE: 1,
            },
            # Number of attention heads in transformer
            NUM_HEADS: 4,
            # If 'True' use key relative embeddings in attention
            KEY_RELATIVE_ATTENTION: False,
            # If 'True' use value relative embeddings in attention
            VALUE_RELATIVE_ATTENTION: False,
            # Max position for relative embeddings. Only in effect if key- or value
            # relative
            # attention are turned on
            MAX_RELATIVE_POSITION: 5,
            # Use a unidirectional or bidirectional encoder
            # for `text`, `action_text`, and `label_action_text`.
            UNIDIRECTIONAL_ENCODER: False,
            # ## Training parameters
            # Initial and final batch sizes:
            # Batch size will be linearly increased for each epoch.
            BATCH_SIZES: [64, 256],
            # Strategy used whenc creating batches.
            # Can be either 'sequence' or 'balanced'.
            BATCH_STRATEGY: BALANCED,
            # Number of epochs to train
            EPOCHS: 1,
            # Set random seed to any 'int' to get reproducible results
            RANDOM_SEED: None,
            # Initial learning rate for the optimizer
            LEARNING_RATE: 0.001,
            # ## Parameters for embeddings
            # Dimension size of embedding vectors
            EMBEDDING_DIMENSION: 20,
            # The number of incorrect labels. The algorithm will minimize
            # their similarity to the user input during training.
            NUM_NEG: 20,
            # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
            SIMILARITY_TYPE: AUTO,
            # The type of the loss function, either 'cross_entropy' or 'margin'.
            LOSS_TYPE: CROSS_ENTROPY,
            # Number of top actions for which confidences should be predicted.
            # The number of  Set to `0` if confidences for all actions should be
            # predicted. The confidences for all other actions will be set to 0.
            RANKING_LENGTH: 0,
            # Determines wether the confidences of the chosen top actions should be
            # renormalized so that they sum up to 1. By default, we do not renormalize
            # and return the confidences for the top actions as is.
            # Note that renormalization only makes sense if confidences are generated
            # via `softmax`.
            RENORMALIZE_CONFIDENCES: False,
            # Indicates how similar the algorithm should try to make embedding vectors
            # for correct labels.
            # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
            MAX_POS_SIM: 0.8,
            # Maximum negative similarity for incorrect labels.
            # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
            MAX_NEG_SIM: -0.2,
            # If 'True' the algorithm only minimizes maximum similarity over
            # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
            USE_MAX_NEG_SIM: True,
            # If 'True' scale loss inverse proportionally to the confidence
            # of the correct prediction
            SCALE_LOSS: True,
            # ## Regularization parameters
            # The scale of regularization
            REGULARIZATION_CONSTANT: 0.001,
            # The scale of how important is to minimize the maximum similarity
            # between embeddings of different labels,
            # used only if 'loss_type' is set to 'margin'.
            NEGATIVE_MARGIN_SCALE: 0.8,
            # Dropout rate for embedding layers of dialogue features.
            DROP_RATE_DIALOGUE: 0.1,
            # Dropout rate for embedding layers of utterance level features.
            DROP_RATE: 0.0,
            # Dropout rate for embedding layers of label, e.g. action, features.
            DROP_RATE_LABEL: 0.0,
            # Dropout rate for attention.
            DROP_RATE_ATTENTION: 0.0,
            # Fraction of trainable weights in internal layers.
            CONNECTION_DENSITY: 0.2,
            # If 'True' apply dropout to sparse input tensors
            SPARSE_INPUT_DROPOUT: True,
            # If 'True' apply dropout to dense input tensors
            DENSE_INPUT_DROPOUT: True,
            # If 'True' random tokens of the input message will be masked. Since there
            # is no related loss term used inside TED, the masking effectively becomes
            # just input dropout applied to the text of user utterances.
            MASKED_LM: False,
            # ## Evaluation parameters
            # How often calculate validation accuracy.
            # Small values may hurt performance.
            EVAL_NUM_EPOCHS: 20,
            # How many examples to use for hold out validation set
            # Large values may hurt performance, e.g. model accuracy.
            # Set to 0 for no validation.
            EVAL_NUM_EXAMPLES: 0,
            # If you want to use tensorboard to visualize training and validation
            # metrics, set this option to a valid output directory.
            TENSORBOARD_LOG_DIR: None,
            # Define when training metrics for tensorboard should be logged.
            # Either after every epoch or for every training step.
            # Valid values: 'epoch' and 'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",
            # Perform model checkpointing
            CHECKPOINT_MODEL: False,
            # Only pick e2e prediction if the policy is confident enough
            E2E_CONFIDENCE_THRESHOLD: 0.5,
            # Specify what features to use as sequence and sentence features.
            # By default all features in the pipeline are used.
            FEATURIZERS: [],
            # If set to true, entities are predicted in user utterances.
            ENTITY_RECOGNITION: True,
            # if 'True' applies sigmoid on all similarity terms and adds
            # it to the loss function to ensure that similarity values are
            # approximately bounded. Used inside cross-entropy loss only.
            CONSTRAIN_SIMILARITIES: False,
            # Model confidence to be returned during inference. Currently, the only
            # possible value is `softmax`.
            MODEL_CONFIDENCE: SOFTMAX,
            # 'BILOU_flag' determines whether to use BILOU tagging or not.
            # If set to 'True' labelling is more rigorous, however more
            # examples per entity are required.
            # Rule of thumb: you should have more than 100 examples per entity.
            BILOU_FLAG: True,
            # Split entities by comma, this makes sense e.g. for a list of
            # ingredients in a recipe, but it doesn't make sense for the parts of
            # an address
            SPLIT_ENTITIES_BY_COMMA: SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
            # Max history of the policy, unbounded by default
            POLICY_MAX_HISTORY: DEFAULT_MAX_HISTORY,
            # Determines the importance of policies, higher values take precedence
            POLICY_PRIORITY: DEFAULT_POLICY_PRIORITY,
            USE_GPU: True,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        model: Optional[RasaModel] = None,
        featurizer: Optional[TrackerFeaturizer] = None,
        fake_features: Optional[Dict[Text, List[Features]]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
    ) -> None:
        """Declares instance variables with default values."""
        super().__init__(
            config, model_storage, resource, execution_context, featurizer=featurizer
        )
        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            config[SPLIT_ENTITIES_BY_COMMA], SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE
        )
        self._load_params(config)

        self.model = model

        self._entity_tag_specs = entity_tag_specs

        self.fake_features = fake_features or defaultdict(list)
        # TED is only e2e if only text is present in fake features, which represent
        # all possible input features for current version of this trained ted
        self.only_e2e = TEXT in self.fake_features and INTENT not in self.fake_features

        self._label_data: Optional[RasaModelData] = None
        self.data_example: Optional[Dict[Text, Dict[Text, List[FeatureArray]]]] = None

        self.tmp_checkpoint_dir = None
        if self.config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())

    @staticmethod
    def model_class() -> Type[TED]:
        """Gets the class of the model architecture to be used by the policy.

        Returns:
            Required class.
        """
        return TED

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        return "ted_policy"

    def _load_params(self, config: Dict[Text, Any]) -> None:
        new_config = rasa.utils.train_utils.check_core_deprecated_options(config)
        self.config = new_config
        self._auto_update_configuration()

    def _auto_update_configuration(self) -> None:
        """Takes care of deprecations and compatibility of parameters."""
        self.config = rasa.utils.train_utils.update_confidence_type(self.config)
        rasa.utils.train_utils.validate_configuration_settings(self.config)
        self.config = rasa.utils.train_utils.update_similarity_type(self.config)
        self.config = rasa.utils.train_utils.update_evaluation_parameters(self.config)

    def _create_label_data(
        self,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Tuple[RasaModelData, List[Dict[Text, List[Features]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        encoded_all_labels = (
            state_featurizer.encode_all_labels(domain, precomputations)
            if state_featurizer is not None
            else []
        )

        attribute_data, _ = convert_to_data_format(
            encoded_all_labels, featurizers=self.config[FEATURIZERS]
        )

        label_data = self._assemble_label_data(attribute_data, domain)

        return label_data, encoded_all_labels

    def _assemble_label_data(
        self, attribute_data: Data, domain: Domain
    ) -> RasaModelData:
        """Constructs data regarding labels to be fed to the model.

        The resultant model data can possibly contain one or both of the
        keys - [`label_action_name`, `label_action_text`] but will definitely
        contain the `label` key.
        `label_action_*` will contain the sequence, sentence and mask features
        for corresponding labels and `label` will contain the numerical label ids.

        Args:
            attribute_data: Feature data for all labels.
            domain: Domain of the assistant.

        Returns:
            Features of labels ready to be fed to the model.
        """
        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")
        label_data.add_lengths(
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE_LENGTH,
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE,
        )
        label_ids = np.arange(domain.num_actions)
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
        return label_data

    @staticmethod
    def _should_extract_entities(
        entity_tags: List[List[Dict[Text, List[Features]]]]
    ) -> bool:
        for turns_tags in entity_tags:
            for turn_tags in turns_tags:
                # if turn_tags are empty or all entity tag indices are `0`
                # it means that all the inputs only contain NO_ENTITY_TAG
                if turn_tags and np.any(turn_tags[ENTITY_TAGS][0].features):
                    return True
        return False

    def _create_data_for_entities(
        self, entity_tags: Optional[List[List[Dict[Text, List[Features]]]]]
    ) -> Optional[Data]:
        if not self.config[ENTITY_RECOGNITION]:
            return None

        # check that there are real entity tags
        if entity_tags and self._should_extract_entities(entity_tags):
            entity_tags_data, _ = convert_to_data_format(entity_tags)
            return entity_tags_data

        # there are no "real" entity tags
        logger.debug(
            f"Entity recognition cannot be performed, "
            f"set '{ENTITY_RECOGNITION}' config parameter to 'False'."
        )
        self.config[ENTITY_RECOGNITION] = False

        return None

    def _create_model_data(
        self,
        tracker_state_features: List[List[Dict[Text, List[Features]]]],
        label_ids: Optional[np.ndarray] = None,
        entity_tags: Optional[List[List[Dict[Text, List[Features]]]]] = None,
        encoded_all_labels: Optional[List[Dict[Text, List[Features]]]] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData.

        Args:
            tracker_state_features: a dictionary of attributes
                (INTENT, TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, ACTIVE_LOOP)
                to a list of features for all dialogue turns in all training trackers
            label_ids: the label ids (e.g. action ids) for every dialogue turn in all
                training trackers
            entity_tags: a dictionary of entity type (ENTITY_TAGS) to a list of features
                containing entity tag ids for text user inputs otherwise empty dict
                for all dialogue turns in all training trackers
            encoded_all_labels: a list of dictionaries containing attribute features
                for label ids

        Returns:
            RasaModelData
        """
        model_data = RasaModelData(label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY)

        if label_ids is not None and encoded_all_labels is not None:
            label_ids = np.array(
                [np.expand_dims(seq_label_ids, -1) for seq_label_ids in label_ids]
            )
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [FeatureArray(label_ids, number_of_dimensions=3)],
            )

            attribute_data, self.fake_features = convert_to_data_format(
                tracker_state_features, featurizers=self.config[FEATURIZERS]
            )

            entity_tags_data = self._create_data_for_entities(entity_tags)
            if entity_tags_data is not None:
                model_data.add_data(entity_tags_data)
        else:
            # method is called during prediction
            attribute_data, _ = convert_to_data_format(
                tracker_state_features,
                self.fake_features,
                featurizers=self.config[FEATURIZERS],
            )

        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        model_data.add_lengths(ACTION_TEXT, SEQUENCE_LENGTH, ACTION_TEXT, SEQUENCE)

        # add the dialogue lengths
        attribute_present = next(iter(list(attribute_data.keys())))
        dialogue_lengths = np.array(
            [
                np.size(np.squeeze(f, -1))
                for f in model_data.data[attribute_present][MASK][0]
            ]
        )
        model_data.data[DIALOGUE][LENGTH] = [
            FeatureArray(dialogue_lengths, number_of_dimensions=1)
        ]

        # make sure all keys are in the same order during training and prediction
        model_data.sort()

        return model_data

    @staticmethod
    def _get_trackers_for_training(
        trackers: List[TrackerWithCachedStates],
    ) -> List[TrackerWithCachedStates]:
        """Filters out the list of trackers which should not be used for training.

        Args:
            trackers: All trackers available for training.

        Returns:
            Trackers which should be used for training.
        """
        # By default, we train on all available trackers.
        return trackers

    def _prepare_for_training(
        self,
        trackers: List[TrackerWithCachedStates],
        domain: Domain,
        precomputations: MessageContainerForCoreFeaturization,
        **kwargs: Any,
    ) -> Tuple[RasaModelData, np.ndarray]:
        """Prepares data to be fed into the model.

        Args:
            trackers: List of training trackers to be featurized.
            domain: Domain of the assistant.
            precomputations: Contains precomputed features and attributes.
            **kwargs: Any other arguments.

        Returns:
            Featurized data to be fed to the model and corresponding label ids.
        """
        training_trackers = self._get_trackers_for_training(trackers)
        # dealing with training data
        tracker_state_features, label_ids, entity_tags = self._featurize_for_training(
            training_trackers,
            domain,
            precomputations=precomputations,
            bilou_tagging=self.config[BILOU_FLAG],
            **kwargs,
        )

        if not tracker_state_features:
            return RasaModelData(), label_ids

        self._label_data, encoded_all_labels = self._create_label_data(
            domain, precomputations=precomputations
        )

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            tracker_state_features, label_ids, entity_tags, encoded_all_labels
        )

        if self.config[ENTITY_RECOGNITION]:
            self._entity_tag_specs = (
                self.featurizer.state_featurizer.entity_tag_specs
                if self.featurizer.state_featurizer is not None
                else []
            )

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        return model_data, label_ids

    def run_training(
        self, model_data: RasaModelData, label_ids: Optional[np.ndarray] = None
    ) -> None:
        """Feeds the featurized training data to the model.

        Args:
            model_data: Featurized training data.
            label_ids: Label ids corresponding to the data points in `model_data`.
                These may or may not be used by the function depending
                on how the policy is trained.
        """
        if not self.finetune_mode:
            # This means the model wasn't loaded from a
            # previously trained model and hence needs
            # to be instantiated.
            self.model = self.model_class()(
                model_data.get_signature(),
                self.config,
                isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
                self._label_data,
                self._entity_tag_specs,
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config[LEARNING_RATE])
            )
        (
            data_generator,
            validation_data_generator,
        ) = rasa.utils.train_utils.create_data_generators(
            model_data,
            self.config[BATCH_SIZES],
            self.config[EPOCHS],
            self.config[BATCH_STRATEGY],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[RANDOM_SEED],
        )
        callbacks = rasa.utils.train_utils.create_common_callbacks(
            self.config[EPOCHS],
            self.config[TENSORBOARD_LOG_DIR],
            self.config[TENSORBOARD_LOG_LEVEL],
            self.tmp_checkpoint_dir,
        )

        if self.model is None:
            raise ModelNotFound("No model was detected prior to training.")

        self.model.fit(
            data_generator,
            epochs=self.config[EPOCHS],
            validation_data=validation_data_generator,
            validation_freq=self.config[EVAL_NUM_EPOCHS],
            callbacks=callbacks,
            verbose=False,
            shuffle=False,  # we use custom shuffle inside data generator
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> Resource:
        """Trains the policy (see parent class for full docstring)."""
        if not training_trackers:
            rasa.shared.utils.io.raise_warning(
                f"Skipping training of `{self.__class__.__name__}` "
                f"as no data was provided. You can exclude this "
                f"policy in the configuration "
                f"file to avoid this warning.",
                category=UserWarning,
            )
            return self._resource

        training_trackers = SupportedData.trackers_for_supported_data(
            self.supported_data(), training_trackers
        )

        model_data, label_ids = self._prepare_for_training(
            training_trackers, domain, precomputations
        )

        if model_data.is_empty():
            rasa.shared.utils.io.raise_warning(
                f"Skipping training of `{self.__class__.__name__}` "
                f"as no data was provided. You can exclude this "
                f"policy in the configuration "
                f"file to avoid this warning.",
                category=UserWarning,
            )
            return self._resource

        with (
            contextlib.nullcontext() if self.config["use_gpu"] else tf.device("/cpu:0")
        ):
            self.run_training(model_data, label_ids)

        self.persist()

        return self._resource

    def _featurize_tracker(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        rule_only_data: Optional[Dict[Text, Any]],
    ) -> List[List[Dict[Text, List[Features]]]]:
        # construct two examples in the batch to be fed to the model -
        # one by featurizing last user text
        # and second - an optional one (see conditions below),
        # the first example in the constructed batch either does not contain user input
        # or uses intent or text based on whether TED is e2e only.
        tracker_state_features = self._featurize_for_prediction(
            tracker,
            domain,
            precomputations=precomputations,
            use_text_for_last_user_input=self.only_e2e,
            rule_only_data=rule_only_data,
        )
        # the second - text, but only after user utterance and if not only e2e
        if (
            tracker.latest_action_name == ACTION_LISTEN_NAME
            and TEXT in self.fake_features
            and not self.only_e2e
        ):
            tracker_state_features += self._featurize_for_prediction(
                tracker,
                domain,
                precomputations=precomputations,
                use_text_for_last_user_input=True,
                rule_only_data=rule_only_data,
            )
        return tracker_state_features

    def _pick_confidence(
        self, confidences: np.ndarray, similarities: np.ndarray, domain: Domain
    ) -> Tuple[np.ndarray, bool]:
        # the confidences and similarities have shape (batch-size x number of actions)
        # batch-size can only be 1 or 2;
        # in the case batch-size==2, the first example contain user intent as features,
        # the second - user text as features
        if confidences.shape[0] > 2:
            raise ValueError(
                "We cannot pick prediction from batches of size more than 2."
            )
        # we use heuristic to pick correct prediction
        if confidences.shape[0] == 2:
            # we use similarities to pick appropriate input,
            # since it seems to be more accurate measure,
            # policy is trained to maximize the similarity not the confidence
            non_e2e_action_name = domain.action_names_or_texts[
                np.argmax(confidences[0])
            ]
            logger.debug(f"User intent lead to '{non_e2e_action_name}'.")
            e2e_action_name = domain.action_names_or_texts[np.argmax(confidences[1])]
            logger.debug(f"User text lead to '{e2e_action_name}'.")
            if (
                np.max(confidences[1]) > self.config[E2E_CONFIDENCE_THRESHOLD]
                # TODO maybe compare confidences is better
                and np.max(similarities[1]) > np.max(similarities[0])
            ):
                logger.debug(f"TED predicted '{e2e_action_name}' based on user text.")
                return confidences[1], True

            logger.debug(f"TED predicted '{non_e2e_action_name}' based on user intent.")
            return confidences[0], False

        # by default the first example in a batch is the one to use for prediction
        predicted_action_name = domain.action_names_or_texts[np.argmax(confidences[0])]
        basis_for_prediction = "text" if self.only_e2e else "intent"
        logger.debug(
            f"TED predicted '{predicted_action_name}' "
            f"based on user {basis_for_prediction}."
        )
        return confidences[0], self.only_e2e

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action (see parent class for full docstring)."""
        if self.model is None:
            return self._prediction(self._default_predictions(domain))

        # create model data from tracker
        tracker_state_features = self._featurize_tracker(
            tracker, domain, precomputations, rule_only_data=rule_only_data
        )
        model_data = self._create_model_data(tracker_state_features)
        outputs = self.model.run_inference(model_data)

        if isinstance(outputs["similarities"], np.ndarray):
            # take the last prediction in the sequence
            similarities = outputs["similarities"][:, -1, :]
        else:
            raise TypeError(
                "model output for `similarities` " "should be a numpy array"
            )
        if isinstance(outputs["scores"], np.ndarray):
            confidences = outputs["scores"][:, -1, :]
        else:
            raise TypeError("model output for `scores` should be a numpy array")
        # take correct prediction from batch
        confidence, is_e2e_prediction = self._pick_confidence(
            confidences, similarities, domain
        )

        # rank and mask the confidence (if we need to)
        ranking_length = self.config[RANKING_LENGTH]
        if 0 < ranking_length < len(confidence):
            renormalize = (
                self.config[RENORMALIZE_CONFIDENCES]
                and self.config[MODEL_CONFIDENCE] == SOFTMAX
            )
            _, confidence = train_utils.rank_and_mask(
                confidence, ranking_length=ranking_length, renormalize=renormalize
            )

        optional_events = self._create_optional_event_for_entities(
            outputs, is_e2e_prediction, precomputations, tracker
        )

        return self._prediction(
            confidence.tolist(),
            is_end_to_end_prediction=is_e2e_prediction,
            optional_events=optional_events,
            diagnostic_data=outputs.get(DIAGNOSTIC_DATA),
        )

    def _create_optional_event_for_entities(
        self,
        prediction_output: Dict[Text, tf.Tensor],
        is_e2e_prediction: bool,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        tracker: DialogueStateTracker,
    ) -> Optional[List[Event]]:
        if tracker.latest_action_name != ACTION_LISTEN_NAME or not is_e2e_prediction:
            # entities belong only to the last user message
            # and only if user text was used for prediction,
            # a user message always comes after action listen
            return None

        if not self.config[ENTITY_RECOGNITION]:
            # entity recognition is not turned on, no entities can be predicted
            return None

        # The batch dimension of entity prediction is not the same as batch size,
        # rather it is the number of last (if max history featurizer else all)
        # text inputs in the batch
        # therefore, in order to pick entities from the latest user message
        # we need to pick entities from the last batch dimension of entity prediction
        predicted_tags, confidence_values = rasa.utils.train_utils.entity_label_to_tags(
            prediction_output,
            self._entity_tag_specs,
            self.config[BILOU_FLAG],
            prediction_index=-1,
        )

        if ENTITY_ATTRIBUTE_TYPE not in predicted_tags:
            # no entities detected
            return None

        # entities belong to the last message of the tracker
        # convert the predicted tags to actual entities
        text = tracker.latest_message.text if tracker.latest_message is not None else ""
        if precomputations is not None:
            parsed_message = precomputations.lookup_message(user_text=text)
        else:
            parsed_message = Message(data={TEXT: text})
        tokens = parsed_message.get(TOKENS_NAMES[TEXT])
        entities = EntityExtractorMixin.convert_predictions_into_entities(
            text,
            tokens,
            predicted_tags,
            self.split_entities_config,
            confidences=confidence_values,
        )

        # add the extractor name
        for entity in entities:
            entity[EXTRACTOR] = "TEDPolicy"

        return [EntitiesAdded(entities)]

    def persist(self) -> None:
        """Persists the policy to a storage."""
        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called without a trained model present. "
                "Nothing to persist then!"
            )
            return

        with self._model_storage.write_to(self._resource) as model_path:
            model_filename = self._metadata_filename()
            tf_model_file = model_path / f"{model_filename}.tf_model"

            rasa.shared.utils.io.create_directory_for_file(tf_model_file)

            self.featurizer.persist(model_path)

            if self.config[CHECKPOINT_MODEL] and self.tmp_checkpoint_dir:
                self.model.load_weights(self.tmp_checkpoint_dir / "checkpoint.tf_model")
                # Save an empty file to flag that this model has been
                # produced using checkpointing
                checkpoint_marker = model_path / f"{model_filename}.from_checkpoint.pkl"
                checkpoint_marker.touch()

            self.model.save(str(tf_model_file))

            self.persist_model_utilities(model_path)

    def persist_model_utilities(self, model_path: Path) -> None:
        """Persists model's utility attributes like model weights, etc.

        Args:
            model_path: Path where model is to be persisted
        """
        model_filename = self._metadata_filename()
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.priority.json", self.priority
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.meta.json", self.config
        )
        # save data example
        serialize_nested_feature_arrays(
            self.data_example,
            str(model_path / f"{model_filename}.data_example.st"),
            str(model_path / f"{model_filename}.data_example_metadata.json"),
        )
        # save label data
        serialize_nested_feature_arrays(
            dict(self._label_data.data) if self._label_data is not None else {},
            str(model_path / f"{model_filename}.label_data.st"),
            str(model_path / f"{model_filename}.label_data_metadata.json"),
        )
        # save fake features
        metadata = save_features(
            self.fake_features, str(model_path / f"{model_filename}.fake_features.st")
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.fake_features_metadata.json", metadata
        )

        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.entity_tag_specs.json", entity_tag_specs
        )

    @classmethod
    def _load_model_utilities(cls, model_path: Path) -> Dict[Text, Any]:
        """Loads model's utility attributes.

        Args:
            model_path: Path where model is to be persisted.
        """
        tf_model_file = model_path / f"{cls._metadata_filename()}.tf_model"

        # load data example
        loaded_data = deserialize_nested_feature_arrays(
            str(model_path / f"{cls._metadata_filename()}.data_example.st"),
            str(model_path / f"{cls._metadata_filename()}.data_example_metadata.json"),
        )
        # load label data
        loaded_label_data = deserialize_nested_feature_arrays(
            str(model_path / f"{cls._metadata_filename()}.label_data.st"),
            str(model_path / f"{cls._metadata_filename()}.label_data_metadata.json"),
        )
        label_data = RasaModelData(data=loaded_label_data)

        # load fake features
        metadata = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.fake_features_metadata.json"
        )
        fake_features = load_features(
            str(model_path / f"{cls._metadata_filename()}.fake_features.st"), metadata
        )

        priority = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.priority.json"
        )
        entity_tag_specs = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.entity_tag_specs.json"
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
        model_config = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.meta.json"
        )

        return {
            "tf_model_file": tf_model_file,
            "loaded_data": loaded_data,
            "fake_features": fake_features,
            "label_data": label_data,
            "priority": priority,
            "entity_tag_specs": entity_tag_specs,
            "model_config": model_config,
        }

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> TEDPolicy:
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
        cls,
        model_path: Path,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> TEDPolicy:
        featurizer = TrackerFeaturizer.load(model_path)

        if not (model_path / f"{cls._metadata_filename()}.data_example.st").is_file():
            return cls(
                config,
                model_storage,
                resource,
                execution_context,
                featurizer=featurizer,
            )

        model_utilities = cls._load_model_utilities(model_path)

        config = cls._update_loaded_params(config)
        if execution_context.is_finetuning and EPOCH_OVERRIDE in config:
            config[EPOCHS] = config.get(EPOCH_OVERRIDE)

        (
            model_data_example,
            predict_data_example,
        ) = cls._construct_model_initialization_data(model_utilities["loaded_data"])

        model = None

        with (contextlib.nullcontext() if config["use_gpu"] else tf.device("/cpu:0")):
            model = cls._load_tf_model(
                model_utilities,
                model_data_example,
                predict_data_example,
                featurizer,
                execution_context.is_finetuning,
            )

        return cls._load_policy_with_model(
            config,
            model_storage,
            resource,
            execution_context,
            featurizer=featurizer,
            model_utilities=model_utilities,
            model=model,
        )

    @classmethod
    def _load_policy_with_model(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: TrackerFeaturizer,
        model: TED,
        model_utilities: Dict[Text, Any],
    ) -> TEDPolicy:
        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            model=model,
            featurizer=featurizer,
            fake_features=model_utilities["fake_features"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
        )

    @classmethod
    def _load_tf_model(
        cls,
        model_utilities: Dict[Text, Any],
        model_data_example: RasaModelData,
        predict_data_example: RasaModelData,
        featurizer: TrackerFeaturizer,
        should_finetune: bool,
    ) -> TED:
        model = cls.model_class().load(
            str(model_utilities["tf_model_file"]),
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            config=model_utilities["model_config"],
            max_history_featurizer_is_used=isinstance(
                featurizer, MaxHistoryTrackerFeaturizer
            ),
            label_data=model_utilities["label_data"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
            finetune_mode=should_finetune,
        )
        return model

    @classmethod
    def _construct_model_initialization_data(
        cls, loaded_data: Dict[Text, Dict[Text, List[FeatureArray]]]
    ) -> Tuple[RasaModelData, RasaModelData]:
        model_data_example = RasaModelData(
            label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY, data=loaded_data
        )
        predict_data_example = RasaModelData(
            label_key=LABEL_KEY,
            label_sub_key=LABEL_SUB_KEY,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                # we need to remove label features for prediction if they are present
                if feature_name in PREDICTION_FEATURES
            },
        )
        return model_data_example, predict_data_example

    @classmethod
    def _update_loaded_params(cls, meta: Dict[Text, Any]) -> Dict[Text, Any]:
        meta = rasa.utils.train_utils.update_confidence_type(meta)
        meta = rasa.utils.train_utils.update_similarity_type(meta)

        return meta


class TED(TransformerRasaModel):
    """TED model architecture from https://arxiv.org/abs/1910.00486."""

    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        config: Dict[Text, Any],
        max_history_featurizer_is_used: bool,
        label_data: RasaModelData,
        entity_tag_specs: Optional[List[EntityTagSpec]],
    ) -> None:
        """Initializes the TED model.

        Args:
            data_signature: the data signature of the input data
            config: the model configuration
            max_history_featurizer_is_used: if 'True'
                only the last dialogue turn will be used
            label_data: the label data
            entity_tag_specs: the entity tag specifications
        """
        super().__init__("TED", config, data_signature, label_data)

        self.max_history_featurizer_is_used = max_history_featurizer_is_used

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if feature_name in PREDICTION_FEATURES
        }

        self._entity_tag_specs = entity_tag_specs

        # metrics
        self.action_loss = tf.keras.metrics.Mean(name="loss")
        self.action_acc = tf.keras.metrics.Mean(name="acc")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")
        self.metrics_to_log += ["loss", "acc"]
        if self.config[ENTITY_RECOGNITION]:
            self.metrics_to_log += ["e_loss", "e_f1"]

        # needed for efficient prediction
        self.all_labels_embed: Optional[tf.Tensor] = None

        self._prepare_layers()

    def _check_data(self) -> None:
        if not any(key in [INTENT, TEXT] for key in self.data_signature.keys()):
            raise RasaException(
                f"No user features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

        if not any(
            key in [ACTION_NAME, ACTION_TEXT] for key in self.data_signature.keys()
        ):
            raise ValueError(
                f"No action features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL not in self.data_signature:
            raise ValueError(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    # ---CREATING LAYERS HELPERS---

    def _prepare_layers(self) -> None:
        for name in self.data_signature.keys():
            self._prepare_input_layers(
                name, self.data_signature[name], is_label_attribute=False
            )
            self._prepare_encoding_layers(name)

        for name in self.label_signature.keys():
            self._prepare_input_layers(
                name, self.label_signature[name], is_label_attribute=True
            )
            self._prepare_encoding_layers(name)

        self._tf_layers[
            f"transformer.{DIALOGUE}"
        ] = rasa_layers.prepare_transformer_layer(
            attribute_name=DIALOGUE,
            config=self.config,
            num_layers=self.config[NUM_TRANSFORMER_LAYERS][DIALOGUE],
            units=self.config[TRANSFORMER_SIZE][DIALOGUE],
            drop_rate=self.config[DROP_RATE_DIALOGUE],
            # use bidirectional transformer, because
            # we will invert dialogue sequence so that the last turn is located
            # at the first position and would always have
            # exactly the same positional encoding
            unidirectional=not self.max_history_featurizer_is_used,
        )

        self._prepare_label_classification_layers(DIALOGUE)

        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_input_layers(
        self,
        attribute_name: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        is_label_attribute: bool = False,
    ) -> None:
        """Prepares feature processing layers for sentence/sequence-level features.

        Distinguishes between label features and other features, not applying input
        dropout to the label ones.
        """
        # Disable input dropout in the config to be used if this is a label attribute.
        if is_label_attribute:
            config_to_use = self.config.copy()
            config_to_use.update(
                {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
            )
        else:
            config_to_use = self.config
        # Attributes with sequence-level features also have sentence-level features,
        # all these need to be combined and further processed.
        if attribute_name in SEQUENCE_FEATURES_TO_ENCODE:
            self._tf_layers[
                f"sequence_layer.{attribute_name}"
            ] = rasa_layers.RasaSequenceLayer(
                attribute_name, attribute_signature, config_to_use
            )
        # Attributes without sequence-level features require some actual feature
        # processing only if they have sentence-level features. Attributes with no
        # sequence- and sentence-level features (dialogue, entity_tags, label) are
        # skipped here.
        elif SENTENCE in attribute_signature:
            self._tf_layers[
                f"sparse_dense_concat_layer.{attribute_name}"
            ] = rasa_layers.ConcatenateSparseDenseFeatures(
                attribute=attribute_name,
                feature_type=SENTENCE,
                feature_type_signature=attribute_signature[SENTENCE],
                config=config_to_use,
            )

    def _prepare_encoding_layers(self, name: Text) -> None:
        """Create Ffnn encoding layer used just before combining all dialogue features.

        Args:
            name: attribute name
        """
        # create encoding layers only for the features which should be encoded;
        if name not in SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            return
        # check that there are SENTENCE features for the attribute name in data
        if (
            name in SENTENCE_FEATURES_TO_ENCODE
            and FEATURE_TYPE_SENTENCE not in self.data_signature[name]
        ):
            return
        #  same for label_data
        if (
            name in LABEL_FEATURES_TO_ENCODE
            and FEATURE_TYPE_SENTENCE not in self.label_signature[name]
        ):
            return

        self._prepare_ffnn_layer(
            f"{name}",
            [self.config[ENCODING_DIMENSION]],
            self.config[DROP_RATE_DIALOGUE],
            prefix="encoding_layer",
        )

    # ---GRAPH BUILDING HELPERS---

    @staticmethod
    def _compute_dialogue_indices(
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> None:
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], dtype=tf.int32)
        # wrap in a list, because that's the structure of tf_batch_data
        tf_batch_data[DIALOGUE][INDICES] = [
            (
                tf.map_fn(
                    tf.range,
                    dialogue_lengths,
                    fn_output_signature=tf.RaggedTensorSpec(
                        shape=[None], dtype=tf.int32
                    ),
                )
            ).values
        ]

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]
        # labels cannot have all features "fake"
        all_labels_encoded = {}
        for key in self.tf_label_data.keys():
            if key != LABEL_KEY:
                attribute_features, _, _ = self._encode_real_features_per_attribute(
                    self.tf_label_data, key
                )
                all_labels_encoded[key] = attribute_features

        x = self._collect_label_attribute_encodings(all_labels_encoded)

        # additional sequence axis is artifact of our RasaModelData creation
        # TODO check whether this should be solved in data creation
        x = tf.squeeze(x, axis=1)

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    @staticmethod
    def _collect_label_attribute_encodings(
        all_labels_encoded: Dict[Text, tf.Tensor]
    ) -> tf.Tensor:
        # Initialize with at least one attribute first
        # so that the subsequent TF ops are simplified.
        all_attributes_present = list(all_labels_encoded.keys())
        x = all_labels_encoded.pop(all_attributes_present[0])

        # Add remaining attributes
        for attribute in all_labels_encoded:
            x += all_labels_encoded.get(attribute)
        return x

    def _embed_dialogue(
        self,
        dialogue_in: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """Creates dialogue level embedding and mask.

        Args:
            dialogue_in: The encoded dialogue.
            tf_batch_data: Batch in model data format.

        Returns:
            The dialogue embedding, the mask, and (for diagnostic purposes)
            also the attention weights.
        """
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)
        mask = rasa_layers.compute_mask(dialogue_lengths)

        if self.max_history_featurizer_is_used:
            # invert dialogue sequence so that the last turn would always have
            # exactly the same positional encoding
            dialogue_in = tf.reverse_sequence(dialogue_in, dialogue_lengths, seq_axis=1)

        dialogue_transformed, attention_weights = self._tf_layers[
            f"transformer.{DIALOGUE}"
        ](dialogue_in, 1 - mask, self._training)
        dialogue_transformed = tf.nn.gelu(dialogue_transformed)

        if self.max_history_featurizer_is_used:
            # pick last vector if max history featurizer is used, since we inverted
            # dialogue sequence, the last vector is actually the first one
            dialogue_transformed = dialogue_transformed[:, :1, :]
            mask = tf.expand_dims(self._last_token(mask, dialogue_lengths), 1)
        elif not self._training:
            # during prediction we don't care about previous dialogue turns,
            # so to save computation time, use only the last one
            dialogue_transformed = tf.expand_dims(
                self._last_token(dialogue_transformed, dialogue_lengths), 1
            )
            mask = tf.expand_dims(self._last_token(mask, dialogue_lengths), 1)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        return dialogue_embed, mask, dialogue_transformed, attention_weights

    def _encode_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # The input is a representation of 4d tensor of
        # shape (batch-size x dialogue-len x sequence-len x units) in 3d of shape
        # (sum of dialogue history length for all tensors in the batch x
        # max sequence length x number of features).

        # However, some dialogue turns contain non existent state features,
        # e.g. `intent` and `text` features are mutually exclusive,
        # as well as `action_name` and `action_text` are mutually exclusive,
        # or some dialogue turns don't contain any `slots`.
        # In order to create 4d full tensors, we created "fake" zero features for
        # these non existent state features. And filtered them during batch generation.
        # Therefore the first dimensions for different attributes are different.
        # It could happen that some batches don't contain "real" features at all,
        # e.g. large number of stories don't contain any `slots`.
        # Therefore actual input tensors will be empty.
        # Since we need actual numbers to create dialogue turn features, we create
        # zero tensors in `_encode_fake_features_per_attribute` for these attributes.
        return tf.cond(
            tf.shape(tf_batch_data[attribute][SENTENCE][0])[0] > 0,
            lambda: self._encode_real_features_per_attribute(tf_batch_data, attribute),
            lambda: self._encode_fake_features_per_attribute(tf_batch_data, attribute),
        )

    def _encode_fake_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Returns dummy outputs for fake features of a given attribute.

        Needs to match the outputs of `_encode_real_features_per_attribute` in shape
        but these outputs will be filled with zeros.

        Args:
            tf_batch_data: Maps each attribute to its features and masks.
            attribute: The attribute whose fake features will be "processed", e.g.
                `ACTION_NAME`, `INTENT`.

        Returns:
            attribute_features: A tensor of shape `(batch_size, dialogue_length, units)`
                filled with zeros.
            text_output: Only for `TEXT` attribute (otherwise an empty tensor): A tensor
                of shape `(combined batch_size & dialogue_length, max seq length,
                units)` filled with zeros.
            text_sequence_lengths: Only for `TEXT` attribute, otherwise an empty tensor:
                Of hape `(combined batch_size & dialogue_length, 1)`, filled with zeros.
        """
        # we need to create real zero tensors with appropriate batch and dialogue dim
        # because they are passed to dialogue transformer
        attribute_mask = tf_batch_data[attribute][MASK][0]

        # determine all dimensions so that fake features of the correct shape can be
        # created
        batch_dim = tf.shape(attribute_mask)[0]
        dialogue_dim = tf.shape(attribute_mask)[1]
        if attribute in set(SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE):
            units = self.config[ENCODING_DIMENSION]
        else:
            # state-level attributes don't use an encoding layer, hence their size is
            # just the output size of the corresponding sparse+dense feature combining
            # layer
            units = self._tf_layers[
                f"sparse_dense_concat_layer.{attribute}"
            ].output_units

        attribute_features = tf.zeros(
            (batch_dim, dialogue_dim, units), dtype=tf.float32
        )

        # Only for user text, the transformer output and sequence lengths also have to
        # be created (here using fake features) to enable entity recognition training
        # and prediction.
        if attribute == TEXT:
            # we just need to get the correct last dimension size from the prepared
            # transformer
            text_units = self._tf_layers[f"sequence_layer.{attribute}"].output_units
            text_output = tf.zeros((0, 0, text_units), dtype=tf.float32)
            text_sequence_lengths = tf.zeros((0,), dtype=tf.int32)
        else:
            # simulate None with empty tensor of zeros
            text_output = tf.zeros((0,))
            text_sequence_lengths = tf.zeros((0,))

        return attribute_features, text_output, text_sequence_lengths

    @staticmethod
    def _create_last_dialogue_turns_mask(
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> tf.Tensor:
        # Since max_history_featurizer_is_used is True,
        # we need to find the locations of last dialogue turns in
        # (combined batch dimension and dialogue length,) dimension,
        # so that we can use `_sequence_lengths` as a boolean  mask to pick
        # which ones are "real" textual input in these last dialogue turns.

        # In order to do that we can use given `dialogue_lengths`.
        # For example:
        # If we have `dialogue_lengths = [2, 1, 3]`, than
        # `dialogue_indices = [0, 1, 0, 0, 1, 2]` here we can spot that `0`
        # always indicates the first dialogue turn,
        # which means that previous dialogue turn is the last dialogue turn.
        # Combining this with the fact that the last element in
        # `dialogue_indices` is always the last dialogue turn, we can add
        # a `0` to the end, getting
        # `_dialogue_indices = [0, 1, 0, 0, 1, 2, 0]`.
        # Then removing the first element
        # `_last_dialogue_turn_inverse_indicator = [1, 0, 0, 1, 2, 0]`
        # we see that `0` points to the last dialogue turn.
        # We convert all positive numbers to `True` and take
        # the inverse mask to get
        # `last_dialogue_mask = [0, 1, 1, 0, 0, 1],
        # which precisely corresponds to the fact that first dialogue is of
        # length 2, the second 1 and the third 3.
        last_dialogue_turn_mask = tf.math.logical_not(
            tf.cast(
                tf.concat(
                    [
                        tf_batch_data[DIALOGUE][INDICES][0],
                        tf.zeros((1,), dtype=tf.int32),
                    ],
                    axis=0,
                )[1:],
                dtype=tf.bool,
            )
        )
        # get only the indices of real inputs
        return tf.boolean_mask(
            last_dialogue_turn_mask,
            tf.reshape(tf_batch_data[attribute][SEQUENCE_LENGTH][0], (-1,)),
        )

    def _encode_real_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Encodes features for a given attribute.

        Args:
            tf_batch_data: Maps each attribute to its features and masks.
            attribute: the attribute we will encode features for
                (e.g., ACTION_NAME, INTENT)

        Returns:
            attribute_features: A tensor of shape `(batch_size, dialogue_length, units)`
                with all features for `attribute` processed and combined. If sequence-
                level features are present, the sequence dimension is eliminated using
                a transformer.
            text_output: Only for `TEXT` attribute (otherwise an empty tensor): A tensor
                of shape `(combined batch_size & dialogue_length, max seq length,
                units)` containing token-level embeddings further used for entity
                extraction from user text. Similar to `attribute_features` but returned
                for all tokens, not just for the last one.
            text_sequence_lengths: Only for `TEXT` attribute, otherwise an empty tensor:
                Shape `(combined batch_size & dialogue_length, 1)`, containing the
                sequence length for user text examples in `text_output`. The sequence
                length is effectively the number of tokens + 1 (to account also for
                sentence-level features). Needed for entity extraction from user text.
        """
        # simulate None with empty tensor of zeros
        text_output = tf.zeros((0,))
        text_sequence_lengths = tf.zeros((0,))

        if attribute in SEQUENCE_FEATURES_TO_ENCODE:
            # get lengths of real token sequences as a 3D tensor
            sequence_feature_lengths = self._get_sequence_feature_lengths(
                tf_batch_data, attribute
            )

            # sequence_feature_lengths contain `0` for "fake" features, while
            # tf_batch_data[attribute] contains only "real" features. Hence, we need to
            # get rid of the lengths that are 0. This step produces a 1D tensor.
            sequence_feature_lengths = tf.boolean_mask(
                sequence_feature_lengths, sequence_feature_lengths
            )

            attribute_features, _, _, _, _, _ = self._tf_layers[
                f"sequence_layer.{attribute}"
            ](
                (
                    tf_batch_data[attribute][SEQUENCE],
                    tf_batch_data[attribute][SENTENCE],
                    sequence_feature_lengths,
                ),
                training=self._training,
            )

            combined_sentence_sequence_feature_lengths = sequence_feature_lengths + 1

            # Only for user text, the transformer output and sequence lengths also have
            # to be returned to enable entity recognition training and prediction.
            if attribute == TEXT:
                text_output = attribute_features
                text_sequence_lengths = combined_sentence_sequence_feature_lengths

                if self.max_history_featurizer_is_used:
                    # get the location of all last dialogue inputs
                    last_dialogue_turns_mask = self._create_last_dialogue_turns_mask(
                        tf_batch_data, attribute
                    )
                    # pick outputs that correspond to the last dialogue turns
                    text_output = tf.boolean_mask(text_output, last_dialogue_turns_mask)
                    text_sequence_lengths = tf.boolean_mask(
                        text_sequence_lengths, last_dialogue_turns_mask
                    )

            # resulting attribute features will have shape
            # combined batch dimension and dialogue length x 1 x units
            attribute_features = tf.expand_dims(
                self._last_token(
                    attribute_features, combined_sentence_sequence_feature_lengths
                ),
                axis=1,
            )

        # for attributes without sequence-level features, all we need is to combine the
        # sparse and dense sentence-level features into one
        else:
            # resulting attribute features will have shape
            # combined batch dimension and dialogue length x 1 x units
            attribute_features = self._tf_layers[
                f"sparse_dense_concat_layer.{attribute}"
            ]((tf_batch_data[attribute][SENTENCE],), training=self._training)

        if attribute in SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            attribute_features = self._tf_layers[f"encoding_layer.{attribute}"](
                attribute_features, self._training
            )

        # attribute features have shape
        # (combined batch dimension and dialogue length x 1 x units)
        # convert them back to their original shape of
        # batch size x dialogue length x units
        attribute_features = self._convert_to_original_shape(
            attribute_features, tf_batch_data, attribute
        )

        return attribute_features, text_output, text_sequence_lengths

    @staticmethod
    def _convert_to_original_shape(
        attribute_features: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        attribute: Text,
    ) -> tf.Tensor:
        """Transform attribute features back to original shape.

        Given shape: (combined batch and dialogue dimension x 1 x units)
        Original shape: (batch x dialogue length x units)

        Args:
            attribute_features: the "real" features to convert
            tf_batch_data: dictionary mapping every attribute to its features and masks
            attribute: the attribute we will encode features for
                (e.g., ACTION_NAME, INTENT)

        Returns:
            The converted attribute features
        """
        # in order to convert the attribute features with shape
        # (combined batch-size and dialogue length x 1 x units)
        # to a shape of (batch-size x dialogue length x units)
        # we use tf.scatter_nd. Therefore, we need the target shape and the indices
        # mapping the values of attribute features to the position in the resulting
        # tensor.

        # attribute_mask has shape batch x dialogue_len x 1
        attribute_mask = tf_batch_data[attribute][MASK][0]

        if attribute in SENTENCE_FEATURES_TO_ENCODE + STATE_LEVEL_FEATURES:
            dialogue_lengths = tf.cast(
                tf_batch_data[DIALOGUE][LENGTH][0], dtype=tf.int32
            )
            dialogue_indices = tf_batch_data[DIALOGUE][INDICES][0]
        else:
            # for labels, dialogue length is a fake dim and equal to 1
            dialogue_lengths = tf.ones((tf.shape(attribute_mask)[0],), dtype=tf.int32)
            dialogue_indices = tf.zeros((tf.shape(attribute_mask)[0],), dtype=tf.int32)

        batch_dim = tf.shape(attribute_mask)[0]
        dialogue_dim = tf.shape(attribute_mask)[1]
        units = attribute_features.shape[-1]

        # attribute_mask has shape (batch x dialogue_len x 1), remove last dimension
        attribute_mask = tf.cast(tf.squeeze(attribute_mask, axis=-1), dtype=tf.int32)
        # sum of attribute mask contains number of dialogue turns with "real" features
        non_fake_dialogue_lengths = tf.reduce_sum(attribute_mask, axis=-1)
        # create the batch indices
        batch_indices = tf.repeat(tf.range(batch_dim), non_fake_dialogue_lengths)

        # attribute_mask has shape (batch x dialogue_len x 1), while
        # dialogue_indices has shape (combined_dialogue_len,)
        # in order to find positions of real input we need to flatten
        # attribute mask to (combined_dialogue_len,)
        dialogue_indices_mask = tf.boolean_mask(
            attribute_mask, tf.sequence_mask(dialogue_lengths, dtype=tf.int32)
        )
        # pick only those indices that contain "real" input
        dialogue_indices = tf.boolean_mask(dialogue_indices, dialogue_indices_mask)

        indices = tf.stack([batch_indices, dialogue_indices], axis=1)

        shape = tf.convert_to_tensor([batch_dim, dialogue_dim, units])
        attribute_features = tf.squeeze(attribute_features, axis=1)

        return tf.scatter_nd(indices, attribute_features, shape)

    def _process_batch_data(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Encodes batch data.

        Combines intent and text and action name and action text if both are present.

        Args:
            tf_batch_data: dictionary mapping every attribute to its features and masks

        Returns:
             Tensor: encoding of all features in the batch, combined;
        """
        # encode each attribute present in tf_batch_data
        text_output = None
        text_sequence_lengths = None
        batch_encoded = {}
        for attribute in tf_batch_data.keys():
            if attribute in SENTENCE_FEATURES_TO_ENCODE + STATE_LEVEL_FEATURES:
                (
                    attribute_features,
                    _text_output,
                    _text_sequence_lengths,
                ) = self._encode_features_per_attribute(tf_batch_data, attribute)

                batch_encoded[attribute] = attribute_features
                if attribute == TEXT:
                    text_output = _text_output
                    text_sequence_lengths = _text_sequence_lengths

        # if both action text and action name are present, combine them; otherwise,
        # return the one which is present

        if (
            batch_encoded.get(ACTION_TEXT) is not None
            and batch_encoded.get(ACTION_NAME) is not None
        ):
            batch_action = batch_encoded.pop(ACTION_TEXT) + batch_encoded.pop(
                ACTION_NAME
            )
        elif batch_encoded.get(ACTION_TEXT) is not None:
            batch_action = batch_encoded.pop(ACTION_TEXT)
        else:
            batch_action = batch_encoded.pop(ACTION_NAME)
        # same for user input
        if (
            batch_encoded.get(INTENT) is not None
            and batch_encoded.get(TEXT) is not None
        ):
            batch_user = batch_encoded.pop(INTENT) + batch_encoded.pop(TEXT)
        elif batch_encoded.get(TEXT) is not None:
            batch_user = batch_encoded.pop(TEXT)
        else:
            batch_user = batch_encoded.pop(INTENT)

        batch_features = [batch_user, batch_action]
        # once we have user input and previous action,
        # add all other attributes (SLOTS, ACTIVE_LOOP, etc.) to batch_features;
        for key in batch_encoded.keys():
            batch_features.append(batch_encoded.get(key))

        batch_features = tf.concat(batch_features, axis=-1)

        return batch_features, text_output, text_sequence_lengths

    def _reshape_for_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # The first dim of the output of the text sequence transformer is the same
        # as number of "real" features for `text` at the last dialogue turns
        # (let's call it `N`),
        # which corresponds to the first dim of the tag ids tensor.
        # To calculate the loss for entities we need the output of the text
        # sequence transformer (shape: N x sequence length x units),
        # the output of the dialogue transformer
        # (shape: batch size x dialogue length x units) and the tag ids for the
        # entities (shape: N x sequence length - 1 x units)
        # In order to process the tensors, they need to have the same shape.
        # Convert the output of the dialogue transformer to shape
        # (N x 1 x units).

        # Note: The CRF layer cannot handle 4D tensors. E.g. we cannot use the shape
        # batch size x dialogue length x sequence length x units

        # convert the output of the dialogue transformer
        # to shape (real entity dim x 1 x units)
        attribute_mask = tf_batch_data[TEXT][MASK][0]
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)

        if self.max_history_featurizer_is_used:
            # pick outputs that correspond to the last dialogue turns
            attribute_mask = tf.expand_dims(
                self._last_token(attribute_mask, dialogue_lengths), axis=1
            )
        dialogue_transformer_output = tf.boolean_mask(
            dialogue_transformer_output, tf.squeeze(attribute_mask, axis=-1)
        )

        # boolean mask removed axis=1, add it back
        dialogue_transformer_output = tf.expand_dims(
            dialogue_transformer_output, axis=1
        )

        # broadcast the dialogue transformer output sequence-length-times to get the
        # same shape as the text sequence transformer output
        dialogue_transformer_output = tf.tile(
            dialogue_transformer_output, (1, tf.shape(text_output)[1], 1)
        )

        # concat the output of the dialogue transformer to the output of the text
        # sequence transformer (adding context)
        # resulting shape (N x sequence length x 2 units)
        # N = number of "real" features for `text` at the last dialogue turns
        text_transformed = tf.concat(
            [text_output, dialogue_transformer_output], axis=-1
        )
        text_mask = rasa_layers.compute_mask(text_sequence_lengths)

        # add zeros to match the shape of text_transformed, because
        # max sequence length might differ, since it is calculated dynamically
        # based on a subset of sequence lengths
        sequence_diff = tf.shape(text_transformed)[1] - tf.shape(text_mask)[1]
        text_mask = tf.pad(text_mask, [[0, 0], [0, sequence_diff], [0, 0]])

        # remove additional dims and sentence features
        text_sequence_lengths = tf.reshape(text_sequence_lengths, (-1,)) - 1

        return text_transformed, text_mask, text_sequence_lengths

    # ---TRAINING---

    def _batch_loss_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> tf.Tensor:
        # It could happen that some batches don't contain "real" features for `text`,
        # e.g. large number of stories are intent only.
        # Therefore actual `text_output` will be empty.
        # We cannot create a loss with empty tensors.
        # Since we need actual numbers to create a full loss, we output
        # zero in this case.
        return tf.cond(
            tf.shape(text_output)[0] > 0,
            lambda: self._real_batch_loss_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_output,
                text_sequence_lengths,
            ),
            lambda: tf.constant(0.0),
        )

    def _real_batch_loss_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> tf.Tensor:

        text_transformed, text_mask, text_sequence_lengths = self._reshape_for_entities(
            tf_batch_data,
            dialogue_transformer_output,
            text_output,
            text_sequence_lengths,
        )

        tag_ids = tf_batch_data[ENTITY_TAGS][IDS][0]
        # add a zero (no entity) for the sentence features to match the shape of inputs
        sequence_diff = tf.shape(text_transformed)[1] - tf.shape(tag_ids)[1]
        tag_ids = tf.pad(tag_ids, [[0, 0], [0, sequence_diff], [0, 0]])

        loss, f1, _ = self._calculate_entity_loss(
            text_transformed,
            tag_ids,
            text_mask,
            text_sequence_lengths,
            ENTITY_ATTRIBUTE_TYPE,
        )

        self.entity_loss.update_state(loss)
        self.entity_f1.update_state(f1)

        return loss

    @staticmethod
    def _get_labels_embed(
        label_ids: tf.Tensor, all_labels_embed: tf.Tensor
    ) -> tf.Tensor:
        # instead of processing labels again, gather embeddings from
        # all_labels_embed using label ids

        indices = tf.cast(label_ids[:, :, 0], tf.int32)
        labels_embed = tf.gather(all_labels_embed, indices)

        return labels_embed

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
        self._compute_dialogue_indices(tf_batch_data)

        all_label_ids, all_labels_embed = self._create_all_labels_embed()

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        labels_embed = self._get_labels_embed(label_ids, all_labels_embed)

        dialogue_in, text_output, text_sequence_lengths = self._process_batch_data(
            tf_batch_data
        )
        (
            dialogue_embed,
            dialogue_mask,
            dialogue_transformer_output,
            _,
        ) = self._embed_dialogue(dialogue_in, tf_batch_data)
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        losses = []

        loss, acc = self._tf_layers[f"loss.{LABEL}"](
            dialogue_embed,
            labels_embed,
            label_ids,
            all_labels_embed,
            all_label_ids,
            dialogue_mask,
        )
        losses.append(loss)

        if (
            self.config[ENTITY_RECOGNITION]
            and text_output is not None
            and text_sequence_lengths is not None
        ):
            losses.append(
                self._batch_loss_entities(
                    tf_batch_data,
                    dialogue_transformer_output,
                    text_output,
                    text_sequence_lengths,
                )
            )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return tf.math.add_n(losses)

    # ---PREDICTION---
    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction."""
        _, self.all_labels_embed = self._create_all_labels_embed()

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )
        self._compute_dialogue_indices(tf_batch_data)

        dialogue_in, text_output, text_sequence_lengths = self._process_batch_data(
            tf_batch_data
        )
        (
            dialogue_embed,
            dialogue_mask,
            dialogue_transformer_output,
            attention_weights,
        ) = self._embed_dialogue(dialogue_in, tf_batch_data)
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        sim_all, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            dialogue_mask,
        )

        predictions = {
            "scores": scores,
            "similarities": sim_all,
            DIAGNOSTIC_DATA: {"attention_weights": attention_weights},
        }

        if (
            self.config[ENTITY_RECOGNITION]
            and text_output is not None
            and text_sequence_lengths is not None
        ):
            pred_ids, confidences = self._batch_predict_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_output,
                text_sequence_lengths,
            )
            name = ENTITY_ATTRIBUTE_TYPE
            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

        return predictions

    def _batch_predict_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # It could happen that current prediction turn don't contain
        # "real" features for `text`,
        # Therefore actual `text_output` will be empty.
        # We cannot predict entities with empty tensors.
        # Since we need to output some tensors of the same shape, we output
        # zero tensors.
        return tf.cond(
            tf.shape(text_output)[0] > 0,
            lambda: self._real_batch_predict_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_output,
                text_sequence_lengths,
            ),
            lambda: (
                # the output is of shape (batch_size, max_seq_len)
                tf.zeros(tf.shape(text_output)[:2], dtype=tf.int32),
                tf.zeros(tf.shape(text_output)[:2], dtype=tf.float32),
            ),
        )

    def _real_batch_predict_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        text_transformed, _, text_sequence_lengths = self._reshape_for_entities(
            tf_batch_data,
            dialogue_transformer_output,
            text_output,
            text_sequence_lengths,
        )

        name = ENTITY_ATTRIBUTE_TYPE

        _logits = self._tf_layers[f"embed.{name}.logits"](text_transformed)

        return self._tf_layers[f"crf.{name}"](_logits, text_sequence_lengths)
