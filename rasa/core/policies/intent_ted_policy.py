import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Any, List, Optional, Text, Dict, Type, Union, TYPE_CHECKING, Tuple
from collections import defaultdict

from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP, ACTION_UNLIKELY_INTENT_NAME
from rasa.shared.core.events import UserUttered
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    ENTITIES,
    ACTION_NAME,
)
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    IntentMaxHistoryTrackerFeaturizer,
)
from rasa.core.featurizers.single_state_featurizer import (
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.core.constants import UNLIKELY_INTENT_POLICY_PRIORITY, DIALOGUE
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.policies.ted_policy import (
    LABEL_KEY,
    LABEL_SUB_KEY,
    TEDPolicy,
    TED,
    SEQUENCE_LENGTH,
    SEQUENCE,
    PREDICTION_FEATURES,
)
from rasa.utils import train_utils
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.tensorflow.constants import (
    LABEL,
    DENSE_DIMENSION,
    ENCODING_DIMENSION,
    UNIDIRECTIONAL_ENCODER,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    EMBEDDING_DIMENSION,
    DROP_RATE_DIALOGUE,
    DROP_RATE_LABEL,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    CONNECTION_DENSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    INNER,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CHECKPOINT_MODEL,
    FEATURIZERS,
    ENTITY_RECOGNITION,
    IGNORE_INTENTS_LIST,
    BILOU_FLAG,
    LEARNING_RATE,
    CROSS_ENTROPY,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    HIDDEN_LAYERS_SIZES,
    CONCAT_DIMENSION,
    TOLERANCE,
    LABEL_PAD_ID,
)
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureArray,
    Data,
)

import rasa.utils.io as io_utils
from rasa.core.exceptions import RasaCoreException

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


logger = logging.getLogger(__name__)

SAVE_MODEL_FILE_NAME = "intent_ted_policy"


class IntentTEDPolicy(TEDPolicy):
    """`IntentTEDPolicy` has the same model architecture as `TEDPolicy`.

    The difference is at a task level.
    Instead of predicting the next probable action, this policy
    predicts whether the last predicted intent is a likely intent
    according to the training stories and conversation context.
    """

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding list.
        HIDDEN_LAYERS_SIZES: {TEXT: []},
        # Dense dimension to use for sparse features.
        DENSE_DIMENSION: {
            TEXT: 128,
            INTENT: 20,
            ACTION_NAME: 20,
            ENTITIES: 20,
            SLOTS: 20,
            ACTIVE_LOOP: 20,
            f"{LABEL}_{INTENT}": 20,
        },
        # Default dimension to use for concatenating sequence and sentence features.
        CONCAT_DIMENSION: {TEXT: 128},
        # Dimension size of embedding vectors before the dialogue transformer encoder.
        ENCODING_DIMENSION: 50,
        # Number of units in transformer encoders
        TRANSFORMER_SIZE: {TEXT: 128, DIALOGUE: 128,},
        # Number of layers in transformer encoders
        NUM_TRANSFORMER_LAYERS: {TEXT: 1, DIALOGUE: 1,},
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # Use a unidirectional or bidirectional encoder
        # for `text`, `action_text`, and `label_action_text`.
        UNIDIRECTIONAL_ENCODER: False,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used when creating batches.
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
        # Number of intents to store in predicted action metadata.
        RANKING_LENGTH: 10,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: True,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.001,
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
        # If 'True' random tokens of the input message will be masked. Since there is no
        # related loss term used inside TED, the masking effectively becomes just input
        # dropout applied to the text of user utterances.
        MASKED_LM: False,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
        # If you want to use tensorboard to visualize training and validation metrics,
        # set this option to a valid output directory.
        TENSORBOARD_LOG_DIR: None,
        # Define when training metrics for tensorboard should be logged.
        # Either after every epoch or for every training step.
        # Valid values: 'epoch' and 'batch'
        TENSORBOARD_LOG_LEVEL: "epoch",
        # Perform model checkpointing
        CHECKPOINT_MODEL: False,
        # Specify what features to use as sequence and sentence features.
        # By default all features in the pipeline are used.
        FEATURIZERS: [],
        # List of intents to ignore for `action_unlikely_intent` prediction.
        IGNORE_INTENTS_LIST: [],
        # Tolerance for prediction of action_unlikely_intent.
        # This is specified as a ratio of negative trackers for
        # which the predicted similarity score of the
        # corresponding correct label is above a threshold.
        # Hence, any tracker at inference time
        # should result in a predicted similarity score equal
        # to or above the threshold in order to avoid triggering
        # `action_unlikely_intent`. Higher values of `tolerance`
        # means the policy is more "tolerant" to surprising paths
        # in conversations and hence will result in lesser number
        # of `action_unlikely_intent` triggers. Acceptable values
        # are between 0.0 and 1.0 .
        TOLERANCE: 0.0,
    }

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = UNLIKELY_INTENT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        fake_features: Optional[Dict[Text, List["Features"]]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        should_finetune: bool = False,
        label_quantiles: Optional[Dict[int, List[float]]] = None,
        **kwargs: Any,
    ) -> None:
        """Declares instance variables with default values."""
        super().__init__(
            featurizer,
            priority,
            max_history,
            model,
            fake_features,
            entity_tag_specs,
            should_finetune,
            **kwargs,
        )

        self.label_quantiles = label_quantiles or {}
        self.label_thresholds = (
            self._pick_thresholds(self.label_quantiles, self.config[TOLERANCE])
            if self.label_quantiles
            else {}
        )
        self.ignore_intent_list = self.config[IGNORE_INTENTS_LIST]

        # Set all invalid / non configurable parameters
        self.config[ENTITY_RECOGNITION] = False
        self.config[BILOU_FLAG] = False
        self.config[SIMILARITY_TYPE] = INNER
        self.config[LOSS_TYPE] = CROSS_ENTROPY

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return IntentMaxHistoryTrackerFeaturizer(
            IntentTokenizerSingleStateFeaturizer(), max_history=max_history
        )

    @staticmethod
    def model_class() -> Type["IntentTED"]:
        """Gets the class of the model architecture to be used by the policy.

        Returns:
            Required class.
        """
        return IntentTED

    def _auto_update_configuration(self) -> None:
        self.config = train_utils.update_evaluation_parameters(self.config)
        self.config = train_utils.update_deprecated_sparsity_to_density(self.config)

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        return SAVE_MODEL_FILE_NAME

    def _assemble_label_data(
        self, attribute_data: Data, domain: Domain
    ) -> RasaModelData:
        """Constructs data regarding labels to be fed to the model.

        The resultant model data should contain the keys `label_intent`, `label`.
        `label_intent` will contain the sequence, sentence and mask features
        for all intent labels and `label` will contain the numerical label ids.

        Args:
            attribute_data: Feature data for all intent labels.
            domain: Domain of the assistant.

        Returns:
            Features of labels ready to be fed to the model.
        """
        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")
        label_data.add_lengths(
            f"{LABEL}_{INTENT}", SEQUENCE_LENGTH, f"{LABEL}_{INTENT}", SEQUENCE,
        )
        label_ids = np.arange(len(domain.intents))
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )
        return label_data

    @staticmethod
    def _prepare_data_for_prediction(model_data: RasaModelData) -> RasaModelData:
        """Transforms training model data to data usable for making model predictions.

        Transformation involves filtering out all features which
        are not useful at prediction time. This is important
        because the prediction signature will not contain these
        attributes and hence prediction will break.

        Args:
            model_data: Data used during model training.

        Returns:
            Transformed data usable for making predictions.
        """
        filtered_data: Dict[Text, Dict[Text, Any]] = {
            key: features
            for key, features in model_data.data.items()
            if key in PREDICTION_FEATURES
        }
        return RasaModelData(data=filtered_data)

    def compute_label_quantiles_post_training(
        self, model_data: RasaModelData, label_ids: np.ndarray
    ) -> None:
        """Computes quantile scores for prediction of `action_unlikely_intent`.

        Multiple quantiles are computed for each label
        so that an appropriate threshold can be picked at
        inference time according to the `tolerance` value specified.

        Args:
            model_data: Data used for training the model.
            label_ids: Numerical IDs of labels for each data point used during training.
        """
        # `model_data` contains data attributes like `label` which were
        # used during training. These attributes are not present in
        # the `predict_data_signature`. Prediction through the model
        # will break if `model_data` is passed as it is through the model.
        # Hence, we first filter out the attributes inside `model_data`
        # to keep only those which should be present during prediction.
        model_prediction_data = self._prepare_data_for_prediction(model_data)
        prediction_scores = self.model.run_bulk_inference(model_prediction_data)
        label_id_scores = self._collect_label_id_grouped_scores(
            prediction_scores, label_ids
        )
        # For each label id, compute multiple quantile scores.
        # These quantile scores can be looked up during inference
        # to select a specific threshold according to the `tolerance`
        # value specified in the configuration.
        self.label_quantiles = self._compute_label_quantiles(label_id_scores)

    def run_training(
        self, model_data: RasaModelData, label_ids: Optional[np.ndarray] = None
    ) -> None:
        """Feeds the featurized training data to the model.

        Args:
            model_data: Featurized training data.
            label_ids: Label ids corresponding to the data points in `model_data`.

        Raises:
            `RasaCoreException` if `label_ids` is None as it's needed for
                running post training procedures.
        """
        if label_ids is None:
            raise RasaCoreException(
                f"Incorrect usage of `run_training` "
                f"method of `{self.__class__.__name__}`."
                f"`label_ids` cannot be left to `None`."
            )
        super().run_training(model_data, label_ids)
        self.compute_label_quantiles_post_training(model_data, label_ids)

    def _collect_action_metadata(
        self, domain: Domain, similarities: np.array
    ) -> Dict[Text, Dict[Text, float]]:
        """Adds any metadata to be attached to the predicted action.

        Similarities for all intents and their thresholds are attached as metadata.

        Args:
            domain: Domain of the assistant.
            similarities: Predicted similarities for each intent.

        Returns:
            Metadata to be attached.
        """
        metadata = {}
        for intent_index, intent in enumerate(domain.intents):
            if intent_index in self.label_thresholds:
                metadata[intent] = {
                    "score": similarities[0][intent_index],
                    "threshold": self.label_thresholds[intent_index],
                }

        return metadata

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: Tracker containing past conversation events.
            domain: Domain of the assistant.
            interpreter: Interpreter which may be used by the policies to create
                additional features.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        if self.model is None:
            return self._prediction(self._default_predictions(domain))

        # Prediction through the policy is skipped if:
        # 1. Last event in the tracker was not of type `UserUttered`.
        # This is to prevent the ensemble of policies from being stuck
        # in a loop.
        # 2. If the tracker does not contain any event of type `UserUttered` till now.
        if not tracker.get_last_event_for(UserUttered) or (
            tracker.events and not isinstance(tracker.events[-1], UserUttered)
        ):
            logger.debug(
                f"Skipping predictions for {self.__class__.__name__} "
                f"as the last event in tracker is not of type `UserUttered`."
            )
            return self._prediction(self._default_predictions(domain))

        # create model data from tracker
        tracker_state_features = self._featurize_for_prediction(
            tracker, domain, interpreter
        )

        model_data = self._create_model_data(tracker_state_features)
        output = self.model.run_inference(model_data)

        # take the last prediction in the sequence
        similarities = output["similarities"][:, -1, :]

        # Check for unlikely intent
        query_intent = tracker.get_last_event_for(UserUttered).intent_name
        is_unlikely_intent = self._check_unlikely_intent(
            domain, similarities, query_intent
        )

        confidences = list(np.zeros(domain.num_actions))

        if is_unlikely_intent:
            confidences[domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME)] = 1.0

        return self._prediction(
            confidences,
            action_metadata=self._collect_action_metadata(domain, similarities),
        )

    def _should_check_for_intent(self, intent: Text, domain: Domain) -> bool:
        """Checks if the intent should raise `action_unlikely_intent`.

        Args:
            intent: Intent to be queried.
            domain: Domain of the assistant.

        Returns:
            Whether intent should raise `action_unlikely_intent` or not.
        """
        if domain.intents.index(intent) not in self.label_thresholds:
            # This means the intent was never present in a story
            logger.debug(
                f"Query intent index {domain.intents.index(intent)} not "
                f"found in label thresholds - {self.label_thresholds}."
                f"Check for `action_unlikely_intent` prediction will be skipped."
            )
            return False
        if intent in self.config[IGNORE_INTENTS_LIST]:
            logger.debug(
                f"Query intent {intent} found in {IGNORE_INTENTS_LIST}. "
                f"Check for `action_unlikely_intent` prediction will be skipped."
            )
            return False

        return True

    def _check_unlikely_intent(
        self, domain: Domain, similarities: np.array, query_intent: Text
    ) -> bool:
        """Checks if the query intent is probable according to model's predictions.

        If the similarity prediction for the intent of
        is lower than the threshold calculated for that
        intent during training, the corresponding user
        intent is unlikely.

        Args:
            domain: Domain of the assistant.
            similarities: Predicted similarities for all intents.
            query_intent: Intent to be queried.

        Returns:
            Whether query intent is likely or not.
        """
        logger.debug(f"Querying for intent {query_intent}")

        if not self._should_check_for_intent(query_intent, domain):
            return False

        predicted_intent_scores = {
            index: similarities[0][index] for index, intent in enumerate(domain.intents)
        }
        sorted_intent_scores = sorted(
            [
                (intent_label, score)
                for intent_label, score in predicted_intent_scores.items()
            ],
            key=lambda x: x[1],
        )
        query_intent_id = domain.intents.index(query_intent)
        query_intent_similarity = similarities[0][query_intent_id]

        logger.debug(
            f"Score for intent `{query_intent}` is "
            f"{query_intent_similarity}, while "
            f"threshold is {self.label_thresholds[query_intent_id]}"
        )
        logger.debug(
            f"Top 5 intents(in ascending order) that "
            f"are likely here are: {sorted_intent_scores[-5:]}"
        )

        # If score for query intent is below threshold and
        # the query intent is not the top likely intent
        if (
            query_intent_similarity < self.label_thresholds[query_intent_id]
            and query_intent_id != sorted_intent_scores[-1][0]
        ):
            logger.debug(
                f"Intent {query_intent}-{query_intent_id} unlikely to occur here."
            )
            return True

        return False

    @staticmethod
    def _collect_label_id_grouped_scores(
        output_scores: Dict[Text, Union[np.ndarray, Dict[Text, Any]]],
        label_ids: np.ndarray,
    ) -> Dict[int, Tuple[List[float], List[float]]]:
        """Collects similarities predicted for each label id.

        For each `label_id`, we collect similarity scores across
        all trackers and categorize them into two buckets:
            1. Similarity scores when `label_id` is the correct label.
            2. Similarity scores when `label_id` is the wrong label.

        Args:
            output_scores: Model's predictions for each data point.
            label_ids: Numerical IDs of labels for each data point.

        Returns:
            Both buckets of similarity scores grouped by each unique label id.
        """
        label_id_scores: Dict[int, List[List[float]]] = defaultdict(list)
        unique_label_ids = np.unique(label_ids).tolist()
        if LABEL_PAD_ID in unique_label_ids:
            unique_label_ids.remove(LABEL_PAD_ID)

        for label_id in unique_label_ids:
            label_id_scores[label_id] = [[], []]

        for index, all_pos_labels in enumerate(label_ids):

            for candidate_label_id in unique_label_ids:
                if candidate_label_id in all_pos_labels:
                    label_id_scores[candidate_label_id][0].append(
                        output_scores["similarities"][index, 0, candidate_label_id]
                    )
                else:
                    label_id_scores[candidate_label_id][1].append(
                        output_scores["similarities"][index, 0, candidate_label_id]
                    )

        # Get only unique scores so that duplicate
        # trackers created because of permutations are pruned out.
        unique_label_id_scores = {
            label_id: (list(set(scores[0])), list(set(scores[1])))
            for label_id, scores in label_id_scores.items()
        }
        return unique_label_id_scores

    @staticmethod
    def _compute_label_quantiles(
        label_id_scores: Dict[int, Tuple[List[float], List[float]]]
    ) -> Dict[int, List[float]]:
        """Computes multiple quantiles for each label id.

        The quantiles are computed over the negative scores
        collected for each label id. However, no quantile score
        can be greater than the minimum positive score collected
        for the corresponding label id.

        Args:
            label_id_scores: Scores collected for each label id
                over positive and negative trackers.

        Returns:
            Computed quantiles for each label id.
        """
        label_quantiles = {}

        for label_id, (positive_scores, negative_scores) in label_id_scores.items():
            label_quantiles[label_id] = [
                min(
                    min(positive_scores),
                    np.quantile(
                        negative_scores,
                        1 - tolerance_value / 100.0,
                        interpolation="lower",
                    ),
                )
                if negative_scores
                else min(positive_scores)
                for tolerance_value in range(0, 100, 5)
            ]

        return label_quantiles

    @staticmethod
    def _pick_thresholds(
        label_quantiles: Dict[int, List[float]], tolerance: float
    ) -> Dict[int, float]:
        """Compute a threshold for each label id.

        Uses tolerance which is the percentage of negative
        trackers for which predicted score should be equal
        to or above the threshold.

        Args:
            label_quantiles: Quantiles computed for each label id
            tolerance: Specified tolerance value from the configuration.

        Returns:
            Computed thresholds
        """
        label_thresholds = {}
        for label_id in label_quantiles:
            num_thresholds = len(label_quantiles[label_id])
            label_thresholds[label_id] = label_quantiles[label_id][
                min(int(tolerance * num_thresholds), num_thresholds - 1)
            ]
        return label_thresholds

    def persist_model_utilities(self, model_path: Path) -> None:
        """Persists model's utility attributes like model weights, etc.

        Args:
            model_path: Path where model is to be persisted
        """
        super().persist_model_utilities(model_path)
        io_utils.pickle_dump(
            model_path / f"{self._metadata_filename()}.label_quantiles.pkl",
            self.label_quantiles,
        )

    @classmethod
    def _load_model_utilities(cls, model_path: Path) -> Dict[Text, Any]:
        """Loads model's utility attributes.

        Args:
            model_path: Path where model is to be persisted.
        """
        model_utilties = super()._load_model_utilities(model_path)
        label_quantiles = io_utils.pickle_load(
            model_path / f"{cls._metadata_filename()}.label_quantiles.pkl"
        )
        model_utilties.update({"label_quantiles": label_quantiles})
        return model_utilties

    @classmethod
    def _update_loaded_params(cls, meta: Dict[Text, Any]) -> Dict[Text, Any]:
        meta = train_utils.override_defaults(cls.defaults, meta)
        return meta

    @classmethod
    def _load_policy_with_model(
        cls,
        model: "IntentTED",
        featurizer: TrackerFeaturizer,
        model_utilities: Dict[Text, Any],
        should_finetune: bool,
    ) -> "IntentTEDPolicy":
        return cls(
            featurizer=featurizer,
            priority=model_utilities["priority"],
            model=model,
            fake_features=model_utilities["fake_features"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
            should_finetune=should_finetune,
            label_quantiles=model_utilities["label_quantiles"],
            **model_utilities["meta"],
        )


class IntentTED(TED):
    """Follows TED's model architecture from https://arxiv.org/abs/1910.00486.

    However, it has been re-purposed to predict multiple
    labels (intents) instead of a single label (action).
    """

    def _prepare_label_classification_layers(self, predictor_attribute: Text) -> None:
        """Prepares layers & loss for the final label prediction step."""
        self._prepare_embed_layers(predictor_attribute)
        self._prepare_embed_layers(LABEL)
        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _prepare_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss",
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = self.dot_product_loss_layer(
            self.config[NUM_NEG],
            scale_loss,
            similarity_type=self.config[SIMILARITY_TYPE],
        )

    @property
    def dot_product_loss_layer(self) -> tf.keras.layers.Layer:
        """Returns the dot-product loss layer to use.

        Multiple intents can be valid simultaneously, so `IntentTED` uses the
        `MultiLabelDotProductLoss`.

        Returns:
            The loss layer that is used by `_prepare_dot_product_loss`.
        """
        return layers.MultiLabelDotProductLoss

    @staticmethod
    def _get_labels_embed(
        label_ids: tf.Tensor, all_labels_embed: tf.Tensor
    ) -> tf.Tensor:
        # instead of processing labels again, gather embeddings from
        # all_labels_embed using label ids

        indices = tf.cast(label_ids[:, :, 0], tf.int32)

        # Find padding indices. They should have a value -1
        padding_indices = tf.where(tf.equal(indices, -1))

        # Create a tensor of ones which will serve as updates to original `indices`
        updates_to_indices = tf.ones((tf.shape(padding_indices)[0]), dtype=tf.int32)

        # Add the tensor of 1s to indices with padding.
        # So, effectively -1s become 0. This is fine because
        # we don't change the original label indices but only
        # make them 'compatible' for the `tf.gather` op below.
        indices_to_gather = tf.cast(
            tf.tensor_scatter_nd_add(indices, padding_indices, updates_to_indices),
            tf.int32,
        )

        labels_embed = tf.gather(all_labels_embed, indices_to_gather)

        return labels_embed

    def run_bulk_inference(
        self, model_data: RasaModelData
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """Computes model's predictions for input data.

        Args:
            model_data: Data to be passed as input

        Returns:
            Predictions for the input data.
        """
        self._training = False

        batch_size = (
            self.config[BATCH_SIZES]
            if isinstance(self.config[BATCH_SIZES], int)
            else self.config[BATCH_SIZES][0]
        )

        return self.run_inference(
            model_data, batch_size=batch_size, output_keys_expected=["similarities"]
        )
