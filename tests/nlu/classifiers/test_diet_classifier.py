import copy
from pathlib import Path

import numpy as np
import pytest
from typing import Callable, List, Optional, Text, Dict, Any, Tuple

import rasa.utils.common
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.constants import BILOU_ENTITIES
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    PREDICTED_CONFIDENCE_KEY,
    INTENT_NAME_KEY,
)
from rasa.utils import train_utils
from rasa.utils.tensorflow.constants import (
    LOSS_TYPE,
    RANDOM_SEED,
    RANKING_LENGTH,
    EPOCHS,
    MASKED_LM,
    RENORMALIZE_CONFIDENCES,
    TENSORBOARD_LOG_LEVEL,
    TENSORBOARD_LOG_DIR,
    EVAL_NUM_EPOCHS,
    EVAL_NUM_EXAMPLES,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    BILOU_FLAG,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    MODEL_CONFIDENCE,
    HIDDEN_LAYERS_SIZES,
    RUN_EAGERLY,
)
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data.loading import load_data
from rasa.utils.tensorflow.model_data_utils import FeatureArray


@pytest.fixture()
def default_diet_resource() -> Resource:
    return Resource("DIET")


@pytest.fixture()
def create_diet(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    default_diet_resource: Resource,
) -> Callable[..., DIETClassifier]:
    def inner(
        config: Dict[Text, Any], load: bool = False, finetune: bool = False
    ) -> DIETClassifier:
        if load:
            constructor = DIETClassifier.load
        else:
            constructor = DIETClassifier.create

        default_execution_context.is_finetuning = finetune
        return constructor(
            config=rasa.utils.common.override_defaults(
                DIETClassifier.get_default_config(), config
            ),
            model_storage=default_model_storage,
            execution_context=default_execution_context,
            resource=default_diet_resource,
        )

    return inner


@pytest.fixture()
def create_train_load_and_process_diet(
    nlu_data_path: Text,
    create_diet: Callable[..., DIETClassifier],
    train_load_and_process_diet: Callable[..., Message],
) -> Callable[..., Message]:
    def inner(
        diet_config: Dict[Text, Any],
        pipeline: Optional[List[Dict[Text, Any]]] = None,
        training_data: str = nlu_data_path,
        message_text: Text = "Rasa is great!",
        expect_intent: bool = True,
    ) -> Message:
        diet = create_diet(diet_config)
        return train_load_and_process_diet(
            diet=diet,
            pipeline=pipeline,
            training_data=training_data,
            message_text=message_text,
            expect_intent=expect_intent,
        )

    return inner


@pytest.fixture()
def train_load_and_process_diet(
    nlu_data_path: Text,
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
    default_model_storage: ModelStorage,
) -> Callable[..., Message]:
    def inner(
        diet: DIETClassifier,
        pipeline: Optional[List[Dict[Text, Any]]] = None,
        training_data: str = nlu_data_path,
        message_text: Text = "Rasa is great!",
        expect_intent: bool = True,
    ) -> Message:

        if not pipeline:
            pipeline = [
                {"component": WhitespaceTokenizer},
                {"component": CountVectorsFeaturizer},
            ]

        training_data, loaded_pipeline = train_and_preprocess(pipeline, training_data)

        diet.train(training_data=training_data)

        message = Message(data={TEXT: message_text})
        message = process_message(loaded_pipeline, message)

        message2 = copy.deepcopy(message)

        classified_message = diet.process([message])[0]

        if expect_intent:
            assert classified_message.data["intent"]["name"]

        loaded_diet = create_diet(diet.component_config, load=True)

        classified_message2 = loaded_diet.process([message2])[0]

        assert classified_message2.fingerprint() == classified_message.fingerprint()

        return loaded_diet, classified_message

    return inner


def test_compute_default_label_features():
    label_features = [
        Message(data={TEXT: "test a"}),
        Message(data={TEXT: "test b"}),
        Message(data={TEXT: "test c"}),
        Message(data={TEXT: "test d"}),
    ]

    output = DIETClassifier._compute_default_label_features(label_features)

    output = output[0]

    for i, o in enumerate(output):
        assert isinstance(o, np.ndarray)
        assert o[0][i] == 1
        assert o.shape == (1, len(label_features))


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
                Message(
                    data={TEXT: "test b"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
            ],
            True,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, INTENT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, INTENT, "test"),
                    ],
                )
            ],
            False,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(2), FEATURE_TYPE_SEQUENCE, INTENT, "test")
                    ],
                )
            ],
            False,
        ),
    ],
)
def test_check_labels_features_exist(
    messages: List[Message], expected: bool, create_diet: Callable[..., DIETClassifier]
):
    attribute = TEXT
    classifier = create_diet({})
    assert classifier._check_labels_features_exist(messages, attribute) == expected


@pytest.mark.parametrize(
    "messages, entity_expected",
    [
        (
            [
                Message(
                    data={
                        TEXT: "test a",
                        INTENT: "intent a",
                        ENTITIES: [
                            {"start": 0, "end": 4, "value": "test", "entity": "test"}
                        ],
                    }
                ),
                Message(
                    data={
                        TEXT: "test b",
                        INTENT: "intent b",
                        ENTITIES: [
                            {"start": 0, "end": 4, "value": "test", "entity": "test"}
                        ],
                    }
                ),
            ],
            True,
        ),
        (
            [
                Message(data={TEXT: "test a", INTENT: "intent a"}),
                Message(data={TEXT: "test b", INTENT: "intent b"}),
            ],
            False,
        ),
    ],
)
def test_model_data_signature_with_entities(
    messages: List[Message],
    entity_expected: bool,
    create_diet: Callable[..., DIETClassifier],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
):
    classifier = create_diet({"BILOU_flag": False})
    training_data = TrainingData(messages)

    # create tokens and features for entity parsing inside DIET
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    training_data, loaded_pipeline = train_and_preprocess(pipeline, training_data)
    model_data = classifier.preprocess_train_data(training_data)
    entity_exists = "entities" in model_data.get_signature().keys()

    assert entity_exists == entity_expected


@pytest.mark.skip_on_windows
@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_different_settings_non_windows(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    pipeline = [
        {
            "component": WhitespaceTokenizer,
            "intent_tokenization_flag": True,
            "intent_split_symbol": "+",
        },
        {"component": CountVectorsFeaturizer},
    ]
    config = {MASKED_LM: True, EPOCHS: 1, RUN_EAGERLY: True}
    create_train_load_and_process_diet(config, pipeline)
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_different_settings(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    config = {LOSS_TYPE: "margin", EPOCHS: 1, RUN_EAGERLY: True}
    create_train_load_and_process_diet(config)
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_nested_dict_config(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    config = {
        HIDDEN_LAYERS_SIZES: {"text": [256, 512]},
        ENTITY_RECOGNITION: False,
        EPOCHS: 1,
        RUN_EAGERLY: True,
    }
    create_train_load_and_process_diet(config)
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_masked_lm_and_eval(
    nlu_data_path: Text,
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    # need at least number of intents as eval num examples...
    # reading the used data here so that the test doesn't break if data is changed
    importer = RasaFileImporter(training_data_paths=[nlu_data_path])
    training_data = importer.get_nlu_data()
    config = {
        MASKED_LM: True,
        EVAL_NUM_EXAMPLES: len(training_data.intents),
        EPOCHS: 10,
        RUN_EAGERLY: True,
    }
    create_train_load_and_process_diet(config)
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(210, func_only=True)
async def test_train_persist_load_with_only_entity_recognition(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    config = {
        ENTITY_RECOGNITION: True,
        INTENT_CLASSIFICATION: False,
        EPOCHS: 1,
        RUN_EAGERLY: True,
    }
    create_train_load_and_process_diet(
        config,
        training_data="data/examples/rasa/demo-rasa-multi-intent.yml",
        expect_intent=False,
    )
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_only_intent_classification(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    create_train_load_and_process_diet(
        {
            ENTITY_RECOGNITION: False,
            INTENT_CLASSIFICATION: True,
            EPOCHS: 1,
            RUN_EAGERLY: True,
        }
    )
    create_diet(
        {MASKED_LM: True, EPOCHS: 1, RUN_EAGERLY: True}, load=True, finetune=True
    )


@pytest.mark.parametrize(
    "classifier_params, data_path, output_length, output_should_sum_to_1",
    [
        (
            {},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            False,
        ),  # (num_intents > default ranking_length)
        (
            {RENORMALIZE_CONFIDENCES: True},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            True,
        ),  # (num_intents > default ranking_length) + renormalize
        (
            {RANKING_LENGTH: 0},
            "data/test/many_intents.yml",
            16,
            True,
        ),  # (ranking_length := num_intents)
        (
            {RANKING_LENGTH: 0, RENORMALIZE_CONFIDENCES: True},
            "data/test/many_intents.yml",
            16,
            True,
        ),  # (ranking_length := num_intents) + (unnecessary) renormalize
        (
            {RANKING_LENGTH: LABEL_RANKING_LENGTH + 1},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH + 1,
            False,
        ),  # (num_intents > specified ranking_length)
        (
            {RANKING_LENGTH: LABEL_RANKING_LENGTH + 1, RENORMALIZE_CONFIDENCES: True},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH + 1,
            True,
        ),  # (num_intents > specified ranking_length) + renormalize
        (
            {},
            "data/test_moodbot/data/nlu.yml",
            7,
            True,
        ),  # (num_intents < default ranking_length)
    ],
)
async def test_softmax_normalization(
    classifier_params,
    data_path: Text,
    output_length,
    output_should_sum_to_1,
    create_train_load_and_process_diet: Callable[..., Message],
):
    classifier_params[RANDOM_SEED] = 42
    classifier_params[EPOCHS] = 1
    classifier_params[RUN_EAGERLY] = True
    classifier_params[EVAL_NUM_EPOCHS] = 1

    _, parsed_message = create_train_load_and_process_diet(
        classifier_params, training_data=data_path
    )
    parse_data = parsed_message.data
    intent_ranking = parse_data.get("intent_ranking")
    # check that the output was correctly truncated after normalization
    assert len(intent_ranking) == output_length

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in intent_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1 == output_should_sum_to_1

    # check whether the normalization of rankings is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


async def test_margin_loss_is_not_normalized(
    create_train_load_and_process_diet: Callable[..., Message]
):
    _, parsed_message = create_train_load_and_process_diet(
        {
            LOSS_TYPE: "margin",
            RANDOM_SEED: 42,
            EPOCHS: 1,
            EVAL_NUM_EPOCHS: 1,
            RUN_EAGERLY: True,
        },
        training_data="data/test/many_intents.yml",
    )
    parse_data = parsed_message.data
    intent_ranking = parse_data.get("intent_ranking")

    # check that the output was correctly truncated
    assert len(intent_ranking) == LABEL_RANKING_LENGTH

    # check that output was not normalized
    assert [item["confidence"] for item in intent_ranking] != pytest.approx(1)

    # make sure top ranking is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.timeout(120, func_only=True)
async def test_set_random_seed(
    create_train_load_and_process_diet: Callable[..., Message]
):
    """test if train result is the same for two runs of tf embedding"""

    _, parsed_message1 = create_train_load_and_process_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 1, EPOCHS: 1, RUN_EAGERLY: True}
    )

    _, parsed_message2 = create_train_load_and_process_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 1, EPOCHS: 1, RUN_EAGERLY: True}
    )

    # Different random seed
    _, parsed_message3 = create_train_load_and_process_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 2, EPOCHS: 1, RUN_EAGERLY: True}
    )

    assert (
        parsed_message1.data["intent"]["confidence"]
        == parsed_message2.data["intent"]["confidence"]
    )
    assert (
        parsed_message2.data["intent"]["confidence"]
        != parsed_message3.data["intent"]["confidence"]
    )


@pytest.mark.parametrize("log_level", ["epoch", "batch"])
async def test_train_tensorboard_logging(
    log_level: Text,
    tmpdir: Path,
    create_train_load_and_process_diet: Callable[..., Message],
):
    tensorboard_log_dir = Path(tmpdir / "tensorboard")

    assert not tensorboard_log_dir.exists()

    pipeline = [
        {"component": WhitespaceTokenizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 3,
            "max_ngram": 17,
            "max_features": 10,
            "min_df": 5,
        },
    ]

    create_train_load_and_process_diet(
        {
            EPOCHS: 1,
            TENSORBOARD_LOG_LEVEL: log_level,
            TENSORBOARD_LOG_DIR: str(tensorboard_log_dir),
            MODEL_CONFIDENCE: "softmax",
            CONSTRAIN_SIMILARITIES: True,
            EVAL_NUM_EXAMPLES: 15,
            EVAL_NUM_EPOCHS: 1,
            RUN_EAGERLY: True,
        },
        pipeline,
    )

    assert tensorboard_log_dir.exists()

    all_files = list(tensorboard_log_dir.rglob("*.*"))
    assert len(all_files) == 2


@pytest.mark.flaky
async def test_train_model_checkpointing(
    default_model_storage: ModelStorage,
    default_diet_resource: Resource,
    create_train_load_and_process_diet: Callable[..., Message],
):
    create_train_load_and_process_diet(
        {
            EPOCHS: 2,
            EVAL_NUM_EPOCHS: 1,
            EVAL_NUM_EXAMPLES: 10,
            CHECKPOINT_MODEL: True,
            RUN_EAGERLY: True,
        }
    )
    with default_model_storage.read_from(default_diet_resource) as model_dir:
        all_files = list(model_dir.rglob("*.*"))
        assert any(["from_checkpoint" in str(filename) for filename in all_files])


async def test_process_unfeaturized_input(
    create_train_load_and_process_diet: Callable[..., Message],
):
    classifier, _ = create_train_load_and_process_diet(
        diet_config={EPOCHS: 1, RUN_EAGERLY: True}
    )
    message_text = "message text"
    unfeaturized_message = Message(data={TEXT: message_text})
    classified_message = classifier.process([unfeaturized_message])[0]

    assert classified_message.get(TEXT) == message_text
    assert not classified_message.get(INTENT)[INTENT_NAME_KEY]
    assert classified_message.get(INTENT)[PREDICTED_CONFIDENCE_KEY] == 0.0
    assert not classified_message.get(ENTITIES)


async def test_train_model_not_checkpointing(
    default_model_storage: ModelStorage,
    default_diet_resource: Resource,
    create_train_load_and_process_diet: Callable[..., Message],
):
    create_train_load_and_process_diet(
        {EPOCHS: 1, CHECKPOINT_MODEL: False, RUN_EAGERLY: True}
    )

    with default_model_storage.read_from(default_diet_resource) as model_dir:
        all_files = list(model_dir.rglob("*.*"))
        assert not any(["from_checkpoint" in str(filename) for filename in all_files])


async def test_train_fails_with_zero_eval_num_epochs(
    create_diet: Callable[..., DIETClassifier]
):
    with pytest.raises(InvalidConfigException):
        with pytest.warns(UserWarning) as warning:
            create_diet(
                {
                    EPOCHS: 1,
                    CHECKPOINT_MODEL: True,
                    EVAL_NUM_EPOCHS: 0,
                    EVAL_NUM_EXAMPLES: 10,
                }
            )

    warn_text = (
        f"You have opted to save the best model, but the value of '{EVAL_NUM_EPOCHS}' "
        f"is not -1 or greater than 0. Training will fail."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1


async def test_doesnt_checkpoint_with_zero_eval_num_examples(
    create_diet: Callable[..., DIETClassifier],
    default_model_storage: ModelStorage,
    default_diet_resource: Resource,
    train_load_and_process_diet: Callable[..., Message],
):
    with pytest.warns(UserWarning) as warning:
        classifier = create_diet(
            {
                EPOCHS: 2,
                CHECKPOINT_MODEL: True,
                EVAL_NUM_EXAMPLES: 0,
                EVAL_NUM_EPOCHS: 1,
                RUN_EAGERLY: True,
            }
        )

    warn_text = (
        f"You have opted to save the best model, but the value of "
        f"'{EVAL_NUM_EXAMPLES}' is not greater than 0. No checkpoint model "
        f"will be saved."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1

    train_load_and_process_diet(classifier)

    with default_model_storage.read_from(default_diet_resource) as model_dir:
        all_files = list(model_dir.rglob("*.*"))
        assert not any(["from_checkpoint" in str(filename) for filename in all_files])


@pytest.mark.parametrize(
    "classifier_params",
    [
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: False, RUN_EAGERLY: True},
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: True, RUN_EAGERLY: True},
    ],
)
async def test_train_persist_load_with_composite_entities(
    classifier_params: Dict[Text, Any],
    create_train_load_and_process_diet: Callable[..., Message],
):
    create_train_load_and_process_diet(
        classifier_params,
        training_data="data/test/demo-rasa-composite-entities.yml",
        message_text="I am looking for an italian restaurant",
    )


@pytest.mark.parametrize("should_add_diagnostic_data", [True, False])
async def test_process_gives_diagnostic_data(
    create_train_load_and_process_diet: Callable[..., Message],
    default_execution_context: ExecutionContext,
    should_add_diagnostic_data: bool,
):
    default_execution_context.should_add_diagnostic_data = should_add_diagnostic_data
    default_execution_context.node_name = "DIETClassifier_node_name"
    _, processed_message = create_train_load_and_process_diet(
        {EPOCHS: 1, RUN_EAGERLY: True}
    )

    if should_add_diagnostic_data:
        # Tests if processing a message returns attention weights as numpy array.
        diagnostic_data = processed_message.get(DIAGNOSTIC_DATA)

        # DIETClassifier should add attention weights
        name = "DIETClassifier_node_name"
        assert isinstance(diagnostic_data, dict)
        assert name in diagnostic_data
        assert "attention_weights" in diagnostic_data[name]
        assert isinstance(diagnostic_data[name].get("attention_weights"), np.ndarray)
        assert "text_transformed" in diagnostic_data[name]
        assert isinstance(diagnostic_data[name].get("text_transformed"), np.ndarray)
    else:
        assert DIAGNOSTIC_DATA not in processed_message.data


@pytest.mark.parametrize(
    "initial_sparse_feature_sizes, final_sparse_feature_sizes, label_attribute",
    [
        (
            {
                TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]},
                INTENT: {FEATURE_TYPE_SEQUENCE: [5], FEATURE_TYPE_SENTENCE: []},
            },
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            INTENT,
        ),
        (
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            INTENT,
        ),
    ],
)
def test_removing_label_sparse_feature_sizes(
    initial_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    final_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    label_attribute: Text,
):
    """Tests if label attribute is removed from sparse feature sizes collection."""
    feature_sizes = DIETClassifier._remove_label_sparse_feature_sizes(
        sparse_feature_sizes=initial_sparse_feature_sizes,
        label_attribute=label_attribute,
    )
    assert feature_sizes == final_sparse_feature_sizes


@pytest.mark.timeout(120)
async def test_adjusting_layers_incremental_training(
    create_diet: Callable[..., DIETClassifier],
    train_load_and_process_diet: Callable[..., Message],
):
    """Tests adjusting sparse layers of `DIETClassifier` to increased sparse
    feature sizes during incremental training.

    Testing is done by checking the layer sizes.
    Checking if they were replaced correctly is also important
    and is done in `test_replace_dense_for_sparse_layers`
    in `test_rasa_layers.py`.
    """
    iter1_data_path = "data/test_incremental_training/iter1/"
    iter2_data_path = "data/test_incremental_training/"
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": LexicalSyntacticFeaturizer},
        {"component": RegexFeaturizer},
        {"component": CountVectorsFeaturizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
    ]
    classifier = create_diet({EPOCHS: 1, RUN_EAGERLY: True})
    _, processed_message = train_load_and_process_diet(
        classifier, pipeline=pipeline, training_data=iter1_data_path
    )

    old_data_signature = classifier.model.data_signature
    old_predict_data_signature = classifier.model.predict_data_signature
    old_sparse_feature_sizes = processed_message.get_sparse_feature_sizes(
        attribute=TEXT
    )
    initial_diet_layers = classifier.model._tf_layers["sequence_layer.text"]._tf_layers[
        "feature_combining"
    ]
    initial_diet_sequence_layer = initial_diet_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    initial_diet_sentence_layer = initial_diet_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    initial_diet_sequence_size = initial_diet_sequence_layer.get_kernel().shape[0]
    initial_diet_sentence_size = initial_diet_sentence_layer.get_kernel().shape[0]
    assert initial_diet_sequence_size == sum(
        old_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert initial_diet_sentence_size == sum(
        old_sparse_feature_sizes[FEATURE_TYPE_SENTENCE]
    )

    finetune_classifier = create_diet(
        {EPOCHS: 1, RUN_EAGERLY: True}, load=True, finetune=True
    )
    assert finetune_classifier.finetune_mode
    _, processed_message_finetuned = train_load_and_process_diet(
        finetune_classifier, pipeline=pipeline, training_data=iter2_data_path
    )

    new_sparse_feature_sizes = processed_message_finetuned.get_sparse_feature_sizes(
        attribute=TEXT
    )

    final_diet_layers = finetune_classifier.model._tf_layers[
        "sequence_layer.text"
    ]._tf_layers["feature_combining"]
    final_diet_sequence_layer = final_diet_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    final_diet_sentence_layer = final_diet_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    final_diet_sequence_size = final_diet_sequence_layer.get_kernel().shape[0]
    final_diet_sentence_size = final_diet_sentence_layer.get_kernel().shape[0]
    assert final_diet_sequence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert final_diet_sentence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SENTENCE]
    )
    # check if the data signatures were correctly updated
    new_data_signature = finetune_classifier.model.data_signature
    new_predict_data_signature = finetune_classifier.model.predict_data_signature
    iter2_data = load_data(iter2_data_path)
    expected_sequence_lengths = len(iter2_data.training_examples)

    def test_data_signatures(
        new_signature: Dict[Text, Dict[Text, List[FeatureArray]]],
        old_signature: Dict[Text, Dict[Text, List[FeatureArray]]],
    ):
        # Wherever attribute / feature_type signature is not
        # expected to change, directly compare it to old data signature.
        # Else compute its expected signature and compare
        attributes_expected_to_change = [TEXT]
        feature_types_expected_to_change = [
            FEATURE_TYPE_SEQUENCE,
            FEATURE_TYPE_SENTENCE,
        ]

        for attribute, signatures in new_signature.items():

            for feature_type, feature_signatures in signatures.items():

                if feature_type == "sequence_lengths":
                    assert feature_signatures[0].units == expected_sequence_lengths

                elif feature_type not in feature_types_expected_to_change:
                    assert feature_signatures == old_signature.get(attribute).get(
                        feature_type
                    )
                else:
                    for index, feature_signature in enumerate(feature_signatures):
                        if (
                            feature_signature.is_sparse
                            and attribute in attributes_expected_to_change
                        ):
                            assert feature_signature.units == sum(
                                new_sparse_feature_sizes.get(feature_type)
                            )
                        else:
                            # dense signature or attributes that are not
                            # expected to change can be compared directly
                            assert (
                                feature_signature.units
                                == old_signature.get(attribute)
                                .get(feature_type)[index]
                                .units
                            )

    test_data_signatures(new_data_signature, old_data_signature)
    test_data_signatures(new_predict_data_signature, old_predict_data_signature)


# FIXME: these tests take too long to run in CI on Windows, disabling them for now
@pytest.mark.skip_on_windows
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "iter1_path, iter2_path, should_raise_exception",
    [
        (
            "data/test_incremental_training/",
            "data/test_incremental_training/iter1",
            True,
        ),
        (
            "data/test_incremental_training/iter1",
            "data/test_incremental_training/",
            False,
        ),
    ],
)
async def test_sparse_feature_sizes_decreased_incremental_training(
    iter1_path: Text,
    iter2_path: Text,
    should_raise_exception: bool,
    create_diet: Callable[..., DIETClassifier],
    train_load_and_process_diet: Callable[..., Message],
):
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": LexicalSyntacticFeaturizer},
        {"component": RegexFeaturizer},
        {"component": CountVectorsFeaturizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
    ]

    classifier = create_diet({EPOCHS: 1, RUN_EAGERLY: True})
    assert not classifier.finetune_mode
    train_load_and_process_diet(classifier, pipeline=pipeline, training_data=iter1_path)

    finetune_classifier = create_diet(
        {EPOCHS: 1, RUN_EAGERLY: True}, load=True, finetune=True
    )
    assert finetune_classifier.finetune_mode

    if should_raise_exception:
        with pytest.raises(Exception) as exec_info:
            train_load_and_process_diet(
                finetune_classifier, pipeline=pipeline, training_data=iter2_path
            )
        assert "Sparse feature sizes have decreased" in str(exec_info.value)
    else:
        train_load_and_process_diet(
            finetune_classifier, pipeline=pipeline, training_data=iter2_path
        )


@pytest.mark.timeout(120, func_only=True)
async def test_no_bilou_when_entity_recognition_off(
    create_diet: Callable[..., DIETClassifier],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
):
    """test diet doesn't produce BILOU tags when ENTITIY_RECOGNITION false."""

    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    diet = create_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 1, EPOCHS: 1, RUN_EAGERLY: True}
    )

    training_data, loaded_pipeline = train_and_preprocess(
        pipeline, training_data="data/test/demo-rasa-composite-entities.yml"
    )

    diet.train(training_data=training_data)

    assert all(msg.get(BILOU_ENTITIES) is None for msg in training_data.nlu_examples)


@pytest.mark.timeout(120, func_only=True)
@pytest.mark.parametrize(
    "batch_size, expected_num_batches, drop_small_last_batch",
    # the training dataset has 48 NLU examples
    [
        (1, 48, True),
        (8, 6, True),
        (15, 3, True),
        (16, 3, True),
        (18, 3, True),
        (20, 2, True),
        (32, 2, True),
        (64, 1, True),
        (128, 1, True),
        (256, 1, True),
        (1, 48, False),
        (8, 6, False),
        (15, 4, False),
        (16, 3, False),
        (18, 3, False),
        (20, 3, False),
        (32, 2, False),
        (64, 1, False),
        (128, 1, False),
        (256, 1, False),
    ],
)
async def test_dropping_of_last_partial_batch(
    batch_size: int,
    expected_num_batches: int,
    drop_small_last_batch: bool,
    create_diet: Callable[..., DIETClassifier],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
):
    """test that diets data processing produces the right amount of batches.

    We introduced a change to only keep the last incomplete batch if
    1. it has more than 50% of examples of batch size
    2. or it is the only batch in the epoch
    """

    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    diet = create_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 1, EPOCHS: 1, RUN_EAGERLY: True}
    )
    # This data set has 48 NLU examples
    training_data, loaded_pipeline = train_and_preprocess(
        pipeline, training_data="data/test/demo-rasa-no-ents.yml"
    )

    model_data = diet.preprocess_train_data(training_data)
    data_generator, _ = train_utils.create_data_generators(
        model_data, batch_size, 1, drop_small_last_batch=drop_small_last_batch
    )

    assert len(data_generator) == expected_num_batches


@pytest.mark.timeout(120, func_only=True)
async def test_dropping_of_last_partial_batch_empty_data(
    create_diet: Callable[..., DIETClassifier],
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
):
    """test that diets data processing produces the right amount of batches.

    We introduced a change to only keep the last incomplete batch if
    1. it has more than 50% of examples of batch size
    2. or it is the only batch in the epoch
    """

    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": CountVectorsFeaturizer},
    ]
    diet = create_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 1, EPOCHS: 1, RUN_EAGERLY: True}
    )
    training_data, loaded_pipeline = train_and_preprocess(
        pipeline, training_data=TrainingData()
    )

    model_data = diet.preprocess_train_data(training_data)
    data_generator, _ = train_utils.create_data_generators(
        model_data, 64, 1, drop_small_last_batch=True
    )

    assert len(data_generator) == 0
