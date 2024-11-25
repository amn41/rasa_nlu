import dataclasses
import logging
from pathlib import Path
from typing import Any, List, Optional, Text, Dict, Type, Union

import numpy as np
import tensorflow as tf

import rasa.utils.common
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP, ACTION_UNLIKELY_INTENT_NAME
from rasa.shared.core.events import UserUttered, ActionExecuted
import rasa.shared.utils.io
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    ENTITIES,
    ACTION_NAME,
    SPLIT_ENTITIES_BY_COMMA,
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
)
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import IntentMaxHistoryTrackerFeaturizer
from rasa.core.featurizers.single_state_featurizer import (
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import (
    DIALOGUE,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
    UNLIKELY_INTENT_POLICY_PRIORITY,
)
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
    POSITIVE_SCORES_KEY,
    NEGATIVE_SCORES_KEY,
    USE_GPU,
)
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureArray, Data
from rasa.core.exceptions import RasaCoreException
from rasa.shared.utils import common


@dataclasses.dataclass
class RankingCandidateMetadata:
    """Dataclass to represent metada for a candidate intent."""

    name: Text
    score: float
    threshold: Optional[float]
    severity: Optional[float]


@dataclasses.dataclass
class UnexpecTEDIntentPolicyMetadata:
    """Dataclass to represent policy metadata."""

    query_intent: RankingCandidateMetadata
    ranking: List[RankingCandidateMetadata]


logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True
)
class UnexpecTEDIntentPolicy(TEDPolicy):
    """`UnexpecTEDIntentPolicy` has the same model architecture as `TEDPolicy`.

    The difference is at a task level.
    Instead of predicting the next probable action, this policy
    predicts whether the last predicted intent is a likely intent
    according to the training stories and conversation context.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        return {
            # ## Architecture of the used neural network
            # Hidden layer sizes for layers before the embedding layers for user message
            # and labels.
            # The number of hidden layers is equal to the length
            # of the corresponding list.
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
            # Dimension size of embedding vectors before
            # the dialogue transformer encoder.
            ENCODING_DIMENSION: 50,
            # Number of units in transformer encoders
            TRANSFORMER_SIZE: {TEXT: 128, DIALOGUE: 128},
            # Number of layers in transformer encoders
            NUM_TRANSFORMER_LAYERS: {TEXT: 1, DIALOGUE: 1},
            # Number of attention heads in transformer
            NUM_HEADS: 4,
            # If 'True' use key relative embeddings in attention
            KEY_RELATIVE_ATTENTION: False,
            # If 'True' use value relative embeddings in attention
            VALUE_RELATIVE_ATTENTION: False,
            # Max position for relative embeddings. Only in effect
            # if key- or value relative attention are turned on
            MAX_RELATIVE_POSITION: 5,
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
            # Number of intents to store in ranking key of predicted action metadata.
            # Set this to `0` to include all intents.
            RANKING_LENGTH: LABEL_RANKING_LENGTH,
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
            # If 'True' random tokens of the input message will be masked.
            # Since there is no related loss term used inside TED, the masking
            # effectively becomes just input dropout applied to the text of user
            # utterances.
            MASKED_LM: False,
            # ## Evaluation parameters
            # How often calculate validation accuracy.
            # Small values may hurt performance, e.g. model accuracy.
            EVAL_NUM_EPOCHS: 20,
            # How many examples to use for hold out validation set
            # Large values may hurt performance, e.g. model accuracy.
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
            # Specify what features to use as sequence and sentence features.
            # By default all features in the pipeline are used.
            FEATURIZERS: [],
            # List of intents to ignore for `action_unlikely_intent` prediction.
            IGNORE_INTENTS_LIST: [],
            # Tolerance for prediction of `action_unlikely_intent`.
            # For each intent, the tolerance is the percentage of
            # negative training instances (trackers for which
            # the corresponding intent is not the correct label) that
            # would be ignored by `UnexpecTEDIntentPolicy`. This is converted
            # into a similarity threshold by identifying the similarity
            # score for the (1 - tolerance) percentile of negative
            # examples. Any tracker with a similarity score below this
            # threshold will trigger an `action_unlikely_intent`.
            # Higher values of `tolerance` means the policy is more
            # "tolerant" to surprising paths in conversations and
            # hence will result in lesser number of `action_unlikely_intent`
            # triggers. Acceptable values are between 0.0 and 1.0 (inclusive).
            TOLERANCE: 0.0,
            # Split entities by comma, this makes sense e.g. for a list of
            # ingredients in a recipe, but it doesn't make sense for the parts of
            # an address
            SPLIT_ENTITIES_BY_COMMA: SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
            # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
            SIMILARITY_TYPE: INNER,
            # If set to true, entities are predicted in user utterances.
            ENTITY_RECOGNITION: False,
            # 'BILOU_flag' determines whether to use BILOU tagging or not.
            # If set to 'True' labelling is more rigorous, however more
            # examples per entity are required.
            # Rule of thumb: you should have more than 100 examples per entity.
            BILOU_FLAG: False,
            # The type of the loss function, either 'cross_entropy' or 'margin'.
            LOSS_TYPE: CROSS_ENTROPY,
            # Determines the importance of policies, higher values take precedence
            POLICY_PRIORITY: UNLIKELY_INTENT_POLICY_PRIORITY,
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
        label_quantiles: Optional[Dict[int, List[float]]] = None,
    ):
        """Declares instance variables with default values."""
        # Set all invalid / non configurable parameters
        config[ENTITY_RECOGNITION] = False
        config[BILOU_FLAG] = False
        config[SIMILARITY_TYPE] = INNER
        config[LOSS_TYPE] = CROSS_ENTROPY
        self.config = config

        super().__init__(
            self.config,
            model_storage,
            resource,
            execution_context,
            model,
            featurizer,
            fake_features,
            entity_tag_specs,
        )

        self.label_quantiles = label_quantiles or {}
        self.label_thresholds = (
            self._pick_thresholds(self.label_quantiles, self.config[TOLERANCE])
            if self.label_quantiles
            else {}
        )
        self.ignore_intent_list = self.config[IGNORE_INTENTS_LIST]

        common.mark_as_experimental_feature("UnexpecTED Intent Policy")

    def _standard_featurizer(self) -> IntentMaxHistoryTrackerFeaturizer:
        return IntentMaxHistoryTrackerFeaturizer(
            IntentTokenizerSingleStateFeaturizer(),
            max_history=self.config.get(POLICY_MAX_HISTORY),
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

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        return "unexpected_intent_policy"

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
            f"{LABEL}_{INTENT}", SEQUENCE_LENGTH, f"{LABEL}_{INTENT}", SEQUENCE
        )
        label_ids = np.arange(len(domain.intents))
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
        prediction_scores = (
            self.model.run_bulk_inference(model_prediction_data)
            if self.model is not None
            else {}
        )
        label_id_scores = self._collect_label_id_grouped_scores(
            prediction_scores, label_ids
        )
        # For each label id, compute multiple quantile scores.
        # These quantile scores can be looked up during inference
        # to select a specific threshold according to the `tolerance`
        # value specified in the configuration.
        self.label_quantiles = self._compute_label_quantiles(label_id_scores)

    @staticmethod
    def _get_trackers_for_training(
        trackers: List[TrackerWithCachedStates],
    ) -> List[TrackerWithCachedStates]:
        """Filters out the list of trackers which should not be used for training.

        `UnexpecTEDIntentPolicy` cannot be trained on trackers with:
        1. `UserUttered` events with no intent.
        2. `ActionExecuted` events with no action_name.

        Trackers with such events are filtered out.

        Args:
            trackers: All trackers available for training.

        Returns:
            Trackers which should be used for training.
        """
        trackers_for_training = []
        for tracker in trackers:
            tracker_compatible = True
            for event in tracker.applied_events():
                if (isinstance(event, UserUttered) and event.intent_name is None) or (
                    isinstance(event, ActionExecuted) and event.action_name is None
                ):
                    tracker_compatible = False
                    break
            if tracker_compatible:
                trackers_for_training.append(tracker)
        return trackers_for_training

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
        self, domain: Domain, similarities: np.ndarray, query_intent: Text
    ) -> UnexpecTEDIntentPolicyMetadata:
        """Collects metadata to be attached to the predicted action.

        Metadata schema looks like this:

        {
            "query_intent": <metadata of intent that was queried>,
            "ranking": <sorted list of metadata corresponding to all intents
                        (truncated by `ranking_length` parameter)
                        It also includes the `query_intent`.
                        Sorting is based on predicted similarities.>
        }

        Each metadata dictionary looks like this:

        {
            "name": <name of intent>,
            "score": <predicted similarity score>,
            "threshold": <threshold used for intent>,
            "severity": <numerical difference between threshold and score>
        }

        Args:
            domain: Domain of the assistant.
            similarities: Predicted similarities for each intent.
            query_intent: Name of intent queried in this round of inference.

        Returns:
            Metadata to be attached.
        """
        query_intent_index = domain.intents.index(query_intent)

        def _compile_metadata_for_label(
            label_name: Text, similarity_score: float, threshold: Optional[float]
        ) -> RankingCandidateMetadata:
            severity = float(threshold - similarity_score) if threshold else None
            return RankingCandidateMetadata(
                label_name,
                float(similarity_score),
                float(threshold) if threshold else None,
                severity,
            )

        query_intent_metadata = _compile_metadata_for_label(
            query_intent,
            similarities[0][domain.intents.index(query_intent)],
            self.label_thresholds.get(query_intent_index),
        )

        # Ranking in descending order of predicted similarities
        sorted_similarities = sorted(
            [(index, similarity) for index, similarity in enumerate(similarities[0])],
            key=lambda x: -x[1],
        )

        if self.config[RANKING_LENGTH] > 0:
            sorted_similarities = sorted_similarities[: self.config[RANKING_LENGTH]]

        ranking_metadata = [
            _compile_metadata_for_label(
                domain.intents[intent_index],
                similarity,
                self.label_thresholds.get(intent_index),
            )
            for intent_index, similarity in sorted_similarities
        ]

        return UnexpecTEDIntentPolicyMetadata(query_intent_metadata, ranking_metadata)

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: Tracker containing past conversation events.
            domain: Domain of the assistant.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            precomputations: Contains precomputed features and attributes.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        if self.model is None:
            return self._prediction(self._default_predictions(domain))

        # Prediction through the policy is skipped if:
        # 1. If the tracker does not contain any event of type `UserUttered`
        #    till now or the intent of such event is not in domain.
        # 2. There is at least one event of type `ActionExecuted`
        #    after the last `UserUttered` event.
        if self._should_skip_prediction(tracker, domain):
            logger.debug(
                f"Skipping predictions for {self.__class__.__name__} "
                f"as either there is no event of type `UserUttered`, "
                f"event's intent is new and not in domain or "
                f"there is an event of type `ActionExecuted` after "
                f"the last `UserUttered`."
            )
            return self._prediction(self._default_predictions(domain))

        # create model data from tracker
        tracker_state_features = self._featurize_for_prediction(
            tracker, domain, precomputations, rule_only_data=rule_only_data
        )

        model_data = self._create_model_data(tracker_state_features)
        output = self.model.run_inference(model_data)

        # take the last prediction in the sequence
        if isinstance(output["similarities"], np.ndarray):
            sequence_similarities = output["similarities"][:, -1, :]
        else:
            raise TypeError(
                "model output for `similarities` " "should be a numpy array"
            )

        # Check for unlikely intent
        last_user_uttered_event = tracker.get_last_event_for(UserUttered)
        query_intent = (
            last_user_uttered_event.intent_name
            if last_user_uttered_event is not None
            else ""
        )
        is_unlikely_intent = self._check_unlikely_intent(
            domain, sequence_similarities, query_intent
        )

        confidences = list(np.zeros(domain.num_actions))

        if is_unlikely_intent:
            confidences[domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME)] = 1.0

        return self._prediction(
            confidences,
            action_metadata=dataclasses.asdict(
                self._collect_action_metadata(
                    domain, sequence_similarities, query_intent
                )
            ),
        )

    @staticmethod
    def _should_skip_prediction(tracker: DialogueStateTracker, domain: Domain) -> bool:
        """Checks if the policy should skip making a prediction.

        A prediction can be skipped if:
            1. There is no event of type `UserUttered` in the tracker.
            2. If the `UserUttered` event's intent is new and not in domain
                (a new intent can be created from rasa interactive and not placed in
                domain yet)
            3. There is an event of type `ActionExecuted` after the last
                `UserUttered` event. This is to prevent the dialogue manager
                from getting stuck in a prediction loop.
                For example, if the last `ActionExecuted` event
                contained `action_unlikely_intent` predicted by
                `UnexpecTEDIntentPolicy` and
                if `UnexpecTEDIntentPolicy` runs inference
                on the same tracker, it will predict `action_unlikely_intent`
                again which would make the dialogue manager get stuck in a
                prediction loop.

        Returns:
            Whether prediction should be skipped.
        """
        applied_events = tracker.applied_events()

        for event in reversed(applied_events):
            if isinstance(event, ActionExecuted):
                return True
            elif isinstance(event, UserUttered):
                if event.intent_name not in domain.intents:
                    return True
                return False
        # No event of type `ActionExecuted` and `UserUttered` means
        # that there is nothing for `UnexpecTEDIntentPolicy` to predict on.
        return True

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
                f"found in label thresholds - {self.label_thresholds}. "
                f"Check for `{ACTION_UNLIKELY_INTENT_NAME}` prediction will be skipped."
            )
            return False
        if intent in self.config[IGNORE_INTENTS_LIST]:
            logger.debug(
                f"Query intent `{intent}` found in "
                f"`{IGNORE_INTENTS_LIST}={self.config[IGNORE_INTENTS_LIST]}`. "
                f"Check for `{ACTION_UNLIKELY_INTENT_NAME}` prediction will be skipped."
            )
            return False

        return True

    def _check_unlikely_intent(
        self, domain: Domain, similarities: np.ndarray, query_intent: Text
    ) -> bool:
        """Checks if the query intent is probable according to model's predictions.

        If the similarity prediction for the intent
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
        logger.debug(f"Querying for intent `{query_intent}`.")

        if not self._should_check_for_intent(query_intent, domain):
            return False

        predicted_intent_scores = {
            index: similarities[0][index] for index, intent in enumerate(domain.intents)
        }
        sorted_intent_scores = sorted(
            [
                (domain.intents[label_index], score)
                for label_index, score in predicted_intent_scores.items()
            ],
            key=lambda x: x[1],
        )
        query_intent_id = domain.intents.index(query_intent)
        query_intent_similarity = similarities[0][query_intent_id]
        highest_likely_intent_id = domain.intents.index(sorted_intent_scores[-1][0])

        logger.debug(
            f"Score for intent `{query_intent}` is "
            f"`{query_intent_similarity}`, while "
            f"threshold is `{self.label_thresholds[query_intent_id]}`."
        )
        logger.debug(
            f"Top 5 intents (in ascending order) that "
            f"are likely here are: `{sorted_intent_scores[-5:]}`."
        )

        # If score for query intent is below threshold and
        # the query intent is not the top likely intent
        if (
            query_intent_similarity < self.label_thresholds[query_intent_id]
            and query_intent_id != highest_likely_intent_id
        ):
            logger.debug(
                f"Intent `{query_intent}-{query_intent_id}` unlikely to occur here."
            )
            return True

        return False

    @staticmethod
    def _collect_label_id_grouped_scores(
        output_scores: Dict[Text, np.ndarray], label_ids: np.ndarray
    ) -> Dict[int, Dict[Text, List[float]]]:
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
        unique_label_ids = np.unique(label_ids).tolist()
        if LABEL_PAD_ID in unique_label_ids:
            unique_label_ids.remove(LABEL_PAD_ID)

        label_id_scores: Dict[int, Dict[Text, List[float]]] = {
            label_id: {POSITIVE_SCORES_KEY: [], NEGATIVE_SCORES_KEY: []}
            for label_id in unique_label_ids
        }

        for index, all_pos_labels in enumerate(label_ids):

            for candidate_label_id in unique_label_ids:
                if candidate_label_id in all_pos_labels:
                    label_id_scores[candidate_label_id][POSITIVE_SCORES_KEY].append(
                        output_scores["similarities"][index, 0, candidate_label_id]
                    )
                else:
                    label_id_scores[candidate_label_id][NEGATIVE_SCORES_KEY].append(
                        output_scores["similarities"][index, 0, candidate_label_id]
                    )

        return label_id_scores

    @staticmethod
    def _compute_label_quantiles(
        label_id_scores: Dict[int, Dict[Text, List[float]]]
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

        quantile_indices = [
            1 - tolerance_value / 100.0 for tolerance_value in range(0, 100, 5)
        ]
        for label_id, prediction_scores in label_id_scores.items():
            positive_scores, negative_scores = (
                prediction_scores[POSITIVE_SCORES_KEY],
                prediction_scores[NEGATIVE_SCORES_KEY],
            )
            minimum_positive_score = min(positive_scores)
            if negative_scores:
                quantile_values = np.quantile(  # type: ignore[call-overload]
                    negative_scores, quantile_indices, interpolation="lower"
                )
                label_quantiles[label_id] = [
                    min(minimum_positive_score, value) for value in quantile_values
                ]
            else:
                label_quantiles[label_id] = [minimum_positive_score] * len(
                    quantile_indices
                )

        return label_quantiles

    @staticmethod
    def _pick_thresholds(
        label_quantiles: Dict[int, List[float]], tolerance: float
    ) -> Dict[int, float]:
        """Computes a threshold for each label id.

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

        from safetensors.numpy import save_file

        save_file(
            {str(k): np.array(v) for k, v in self.label_quantiles.items()},
            model_path / f"{self._metadata_filename()}.label_quantiles.st",
        )

    @classmethod
    def _load_model_utilities(cls, model_path: Path) -> Dict[Text, Any]:
        """Loads model's utility attributes.

        Args:
            model_path: Path where model is to be persisted.
        """
        model_utilties = super()._load_model_utilities(model_path)

        from safetensors.numpy import load_file

        loaded_label_quantiles = load_file(
            model_path / f"{cls._metadata_filename()}.label_quantiles.st"
        )
        label_quantiles = {int(k): list(v) for k, v in loaded_label_quantiles.items()}

        model_utilties.update({"label_quantiles": label_quantiles})
        return model_utilties

    @classmethod
    def _update_loaded_params(cls, meta: Dict[Text, Any]) -> Dict[Text, Any]:
        meta = rasa.utils.common.override_defaults(cls.get_default_config(), meta)
        return meta

    @classmethod
    def _load_policy_with_model(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: TrackerFeaturizer,
        model: "IntentTED",
        model_utilities: Dict[Text, Any],
    ) -> "UnexpecTEDIntentPolicy":
        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            model=model,
            featurizer=featurizer,
            fake_features=model_utilities["fake_features"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
            label_quantiles=model_utilities["label_quantiles"],
        )


class IntentTED(TED):
    """Follows TED's model architecture from https://arxiv.org/abs/1910.00486.

    However, it has been re-purposed to predict multiple
    labels (intents) instead of a single label (action).
    """

    def _prepare_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss"
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

        # Find padding indices. They should have a value equal to `LABEL_PAD_ID`
        padding_indices = tf.where(tf.equal(indices, LABEL_PAD_ID))

        # Create a tensor of values with sign opposite to `LABEL_PAD_ID` which
        # will serve as updates to original `indices`
        updates_to_indices = (
            tf.ones((tf.shape(padding_indices)[0]), dtype=tf.int32) * -1 * LABEL_PAD_ID
        )

        # Add the updates tensor to indices with padding.
        # So, effectively all indices with `LABEL_PAD_ID=-1`
        # become 0 because updates contain 1s.
        # This is fine because we don't change the original non-padding label
        # indices but only make the padding indices 'compatible'
        # for the `tf.gather` op below.
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
