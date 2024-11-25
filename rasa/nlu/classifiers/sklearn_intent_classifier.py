from __future__ import annotations

import logging
import typing
import warnings
from typing import Any, Dict, List, Optional, Text, Tuple, Type

import numpy as np

import rasa.shared.utils.io
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.shared.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.constants import FEATURIZERS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class SklearnIntentClassifier(GraphComponent, IntentClassifier):
    """Intent classifier using the sklearn framework."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [DenseFeaturizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # C parameter of the svm - cross validation will select the best value
            "C": [1, 2, 5, 10, 20, 100],
            # gamma parameter of the svm
            "gamma": [0.1],
            # the kernels to use for the svm training - cross validation will
            # decide which one of them performs best
            "kernels": ["linear"],
            # We try to find a good number of cross folds to use during
            # intent training, this specifies the max number of folds
            "max_cross_validation_folds": 5,
            # Scoring function used for evaluating the hyper parameters
            # This can be a name or a function (cfr GridSearchCV doc for more info)
            "scoring_function": "f1_weighted",
            "num_threads": 1,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        clf: Optional["sklearn.model_selection.GridSearchCV"] = None,
        le: Optional["sklearn.preprocessing.LabelEncoder"] = None,
    ) -> None:
        """Construct a new intent classifier using the sklearn framework."""
        from sklearn.preprocessing import LabelEncoder

        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()
        self.clf = clf

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SklearnIntentClassifier:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn"]

    def transform_labels_str2num(self, labels: List[Text]) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation
        """
        return self.le.fit_transform(labels)

    def transform_labels_num2str(self, y: np.ndarray) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.

        :param y: List of labels to convert to numeric representation"""

        return self.le.inverse_transform(y)

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        num_threads = self.component_config["num_threads"]

        labels = [e.get("intent") for e in training_data.intent_examples]

        if len(set(labels)) < 2:
            rasa.shared.utils.io.raise_warning(
                "Can not train an intent classifier as there are not "
                "enough intents. Need at least 2 different intents. "
                "Skipping training of intent classifier.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return self._resource

        y = self.transform_labels_str2num(labels)
        training_examples = [
            message
            for message in training_data.intent_examples
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]
        X = np.stack(
            [self._get_sentence_features(example) for example in training_examples]
        )
        # reduce dimensionality
        X = np.reshape(X, (len(X), -1))

        self.clf = self._create_classifier(num_threads, y)

        with warnings.catch_warnings():
            # sklearn raises lots of
            # "UndefinedMetricWarning: F - score is ill - defined"
            # if there are few intent examples, this is needed to prevent it
            warnings.simplefilter("ignore")
            self.clf.fit(X, y)

        self.persist()
        return self._resource

    @staticmethod
    def _get_sentence_features(message: Message) -> np.ndarray:
        _, sentence_features = message.get_dense_features(TEXT)
        if sentence_features is not None:
            return sentence_features.features[0]

        raise ValueError(
            "No sentence features present. Not able to train sklearn policy."
        )

    def _num_cv_splits(self, y: np.ndarray) -> int:
        folds = self.component_config["max_cross_validation_folds"]
        return max(2, min(folds, np.min(np.bincount(y)) // 5))

    def _create_classifier(
        self, num_threads: int, y: np.ndarray
    ) -> "sklearn.model_selection.GridSearchCV":
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC

        C = self.component_config["C"]
        kernels = self.component_config["kernels"]
        gamma = self.component_config["gamma"]
        # dirty str fix because sklearn is expecting
        # str not instance of basestr...
        tuned_parameters = [
            {"C": C, "gamma": gamma, "kernel": [str(k) for k in kernels]}
        ]

        # aim for 5 examples in each fold

        cv_splits = self._num_cv_splits(y)

        return GridSearchCV(
            SVC(C=1, probability=True, class_weight="balanced"),
            param_grid=tuned_parameters,
            n_jobs=num_threads,
            cv=cv_splits,
            scoring=self.component_config["scoring_function"],
            verbose=1,
        )

    def process(self, messages: List[Message]) -> List[Message]:
        """Return the most likely intent and its probability for a message."""
        for message in messages:
            if self.clf is None or not message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            ):
                # component is either not trained or didn't
                # receive enough training data or the input doesn't
                # have required features.
                intent = None
                intent_ranking = []
            else:
                X = self._get_sentence_features(message).reshape(1, -1)

                intent_ids, probabilities = self.predict(X)
                intents = self.transform_labels_num2str(np.ravel(intent_ids))
                # `predict` returns a matrix as it is supposed
                # to work for multiple examples as well, hence we need to flatten
                probabilities = probabilities.flatten()

                if intents.size > 0 and probabilities.size > 0:
                    ranking = list(zip(list(intents), list(probabilities)))[
                        :LABEL_RANKING_LENGTH
                    ]

                    intent = {"name": intents[0], "confidence": probabilities[0]}

                    intent_ranking = [
                        {"name": intent_name, "confidence": score}
                        for intent_name, score in ranking
                    ]
                else:
                    intent = {"name": None, "confidence": 0.0}
                    intent_ranking = []

            message.set("intent", intent, add_to_output=True)
            message.set("intent_ranking", intent_ranking, add_to_output=True)

        return messages

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """Given a bow vector of an input text, predict the intent label.

        Return probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label.
        """
        if self.clf is None:
            raise RasaException(
                "Sklearn intent classifier has not been initialised and trained."
            )

        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given a bow vector of an input text, predict most probable label.

        Return only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second,
                 its probability.
        """
        pred_result = self.predict_prob(X)
        # sort the probabilities retrieving the indices of
        # the elements in sorted order

        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        import skops.io as sio

        with self._model_storage.write_to(self._resource) as model_dir:
            file_name = self.__class__.__name__
            classifier_file_name = model_dir / f"{file_name}_classifier.skops"
            encoder_file_name = model_dir / f"{file_name}_encoder.json"

            if self.clf and self.le:
                # convert self.le.classes_ (numpy array of strings) to a list in order
                # to use json dump
                rasa.shared.utils.io.dump_obj_as_json_to_file(
                    encoder_file_name, list(self.le.classes_)
                )
                sio.dump(self.clf.best_estimator_, classifier_file_name)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> SklearnIntentClassifier:
        """Loads trained component (see parent class for full docstring)."""
        from sklearn.preprocessing import LabelEncoder
        import skops.io as sio

        try:
            with model_storage.read_from(resource) as model_dir:
                file_name = cls.__name__
                classifier_file = model_dir / f"{file_name}_classifier.skops"

                if classifier_file.exists():
                    unknown_types = sio.get_untrusted_types(file=classifier_file)

                    if unknown_types:
                        logger.error(
                            f"Untrusted types ({unknown_types}) found when "
                            f"loading {classifier_file}!"
                        )
                        raise ValueError()
                    else:
                        classifier = sio.load(classifier_file, trusted=unknown_types)

                    encoder_file = model_dir / f"{file_name}_encoder.json"
                    classes = rasa.shared.utils.io.read_json_file(encoder_file)

                    encoder = LabelEncoder()
                    intent_classifier = cls(
                        config, model_storage, resource, classifier, encoder
                    )
                    # convert list of strings (class labels) back to numpy array of
                    # strings
                    intent_classifier.transform_labels_str2num(classes)
                    return intent_classifier
        except ValueError:
            logger.debug(
                f"Failed to load '{cls.__name__}' from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
        return cls(config, model_storage, resource)
