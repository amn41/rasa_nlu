from pathlib import Path
import numpy as np
from typing import Optional, Text, Dict, Any, Union, List, Tuple, TYPE_CHECKING

import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.nlu.utils.bilou_utils
from rasa.shared.constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
import rasa.utils.io as io_utils
from rasa.utils.tensorflow.constants import (
    LOSS_TYPE,
    RANKING_LENGTH,
    RENORMALIZE_CONFIDENCES,
    SIMILARITY_TYPE,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    EPOCHS,
    SOFTMAX,
    MARGIN,
    AUTO,
    INNER,
    COSINE,
    SEQUENCE,
    CROSS_ENTROPY,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    TOLERANCE,
    CHECKPOINT_MODEL,
)
from rasa.utils.tensorflow.callback import RasaTrainingLogger, RasaModelCheckpoint
from rasa.utils.tensorflow.data_generator import RasaBatchDataGenerator
from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.shared.nlu.constants import SPLIT_ENTITIES_BY_COMMA
from rasa.shared.exceptions import InvalidConfigException

if TYPE_CHECKING:
    from rasa.nlu.extractors.extractor import EntityTagSpec
    from rasa.nlu.tokenizers.tokenizer import Token
    from tensorflow.keras.callbacks import Callback


def rank_and_mask(
    confidences: np.ndarray, ranking_length: int = 0, renormalize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a ranking of the given confidences.

    First, it computes a list containing the indices that would sort all the given
    confidences in decreasing order.
    If a `ranking_length` is specified, then only the indices for the `ranking_length`
    largest confidences will be returned and all other confidences (i.e. whose indices
    we do not return) will be masked by setting them to 0.
    Moreover, if `renormalize` is set to `True`, then the confidences will
    additionally be renormalised by dividing them by their sum.

    We assume that the given confidences sum up to 1 and, if the
    `ranking_length` is 0 or larger than the given number of confidences,
    we set the `ranking_length` to the number of confidences.
    Hence, in this case the confidences won't be modified.

    Args:
        confidences: a 1-d array of confidences that are non-negative and sum up to 1
        ranking_length: the size of the ranking to be computed. If set to 0 or
            something larger than the number of given confidences, then this is set
            to the exact number of given confidences.
        renormalize: determines whether the masked confidences should be renormalised.
        return_indices:
    Returns:
        indices of the top `ranking_length` confidences and an array of the same
        shape as the given confidences that contains the possibly masked and
        renormalized confidence values
    """
    indices = confidences.argsort()[::-1]
    confidences = confidences.copy()

    if 0 < ranking_length < len(confidences):
        confidences[indices[ranking_length:]] = 0

        if renormalize and np.sum(confidences) > 0:
            confidences = confidences / np.sum(confidences)

        indices = indices[:ranking_length]

    return indices, confidences


def update_similarity_type(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """If SIMILARITY_TYPE is set to 'auto', update the SIMILARITY_TYPE depending
    on the LOSS_TYPE.

    Args:
        config: model configuration

    Returns: updated model configuration
    """
    if config.get(SIMILARITY_TYPE) == AUTO:
        if config[LOSS_TYPE] == CROSS_ENTROPY:
            config[SIMILARITY_TYPE] = INNER
        elif config[LOSS_TYPE] == MARGIN:
            config[SIMILARITY_TYPE] = COSINE

    return config


def align_token_features(
    list_of_tokens: List[List["Token"]],
    in_token_features: np.ndarray,
    shape: Optional[Tuple] = None,
) -> np.ndarray:
    """Align token features to match tokens.

    ConveRTFeaturizer and LanguageModelFeaturizer might split up tokens into sub-tokens.
    We need to take the mean of the sub-token vectors and take that as token vector.

    Args:
        list_of_tokens: tokens for examples
        in_token_features: token features from ConveRT
        shape: shape of feature matrix

    Returns:
        Token features.
    """
    if shape is None:
        shape = in_token_features.shape
    out_token_features = np.zeros(shape)

    for example_idx, example_tokens in enumerate(list_of_tokens):
        offset = 0
        for token_idx, token in enumerate(example_tokens):
            number_sub_words = token.get(NUMBER_OF_SUB_TOKENS, 1)

            if number_sub_words > 1:
                token_start_idx = token_idx + offset
                token_end_idx = token_idx + offset + number_sub_words

                mean_vec = np.mean(
                    in_token_features[example_idx][token_start_idx:token_end_idx],
                    axis=0,
                )

                offset += number_sub_words - 1

                out_token_features[example_idx][token_idx] = mean_vec
            else:
                out_token_features[example_idx][token_idx] = in_token_features[
                    example_idx
                ][token_idx + offset]

    return out_token_features


def update_evaluation_parameters(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """If EVAL_NUM_EPOCHS is set to -1, evaluate at the end of the training.

    Args:
        config: model configuration

    Returns: updated model configuration
    """
    if config[EVAL_NUM_EPOCHS] == -1:
        config[EVAL_NUM_EPOCHS] = config[EPOCHS]
    elif config[EVAL_NUM_EPOCHS] < 1:
        raise InvalidConfigException(
            f"'{EVAL_NUM_EPOCHS}' is set to "
            f"'{config[EVAL_NUM_EPOCHS]}'. "
            "Only values either equal to -1 or greater than 0 are allowed for this "
            "parameter."
        )
    if config[CHECKPOINT_MODEL] and config[EVAL_NUM_EXAMPLES] == 0:
        config[CHECKPOINT_MODEL] = False
    return config


def load_tf_hub_model(model_url: Text) -> Any:
    """Load model from cache if possible, otherwise from TFHub."""
    import os
    from tensorflow_hub.module_v2 import load as tfhub_load

    # needed to load the ConveRT model
    # noinspection PyUnresolvedReferences
    import tensorflow_text  # noqa: F401

    # required to take care of cases when other files are already
    # stored in the default TFHUB_CACHE_DIR
    try:
        return tfhub_load(model_url)
    except OSError:
        directory = io_utils.create_temporary_directory()
        os.environ["TFHUB_CACHE_DIR"] = directory
        return tfhub_load(model_url)


def _replace_deprecated_option(
    old_option: Text,
    new_option: Union[Text, List[Text]],
    config: Dict[Text, Any],
    warn_until_version: Text = NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
) -> Dict[Text, Any]:
    if old_option not in config:
        return {}

    if isinstance(new_option, str):
        rasa.shared.utils.io.raise_deprecation_warning(
            f"Option '{old_option}' got renamed to '{new_option}'. "
            f"Please update your configuration file.",
            warn_until_version=warn_until_version,
        )
        return {new_option: config[old_option]}

    rasa.shared.utils.io.raise_deprecation_warning(
        f"Option '{old_option}' got renamed to "
        f"a dictionary '{new_option[0]}' with a key '{new_option[1]}'. "
        f"Please update your configuration file.",
        warn_until_version=warn_until_version,
    )
    return {new_option[0]: {new_option[1]: config[old_option]}}


def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Update the config according to changed config params.

    If old model configuration parameters are present in the provided config, replace
    them with the new parameters and log a warning.

    Args:
        config: model configuration

    Returns: updated model configuration
    """
    # note: call _replace_deprecated_option() here when there are options to deprecate

    return config


def check_core_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Update the core config according to changed config params.

    If old model configuration parameters are present in the provided config, replace
    them with the new parameters and log a warning.

    Args:
        config: model configuration

    Returns: updated model configuration
    """
    # note: call _replace_deprecated_option() here when there are options to deprecate

    return config


def entity_label_to_tags(
    model_predictions: Dict[Text, Any],
    entity_tag_specs: List["EntityTagSpec"],
    bilou_flag: bool = False,
    prediction_index: int = 0,
) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]:
    """Convert the output predictions for entities to the actual entity tags.

    Args:
        model_predictions: the output predictions using the entity tag indices
        entity_tag_specs: the entity tag specifications
        bilou_flag: if 'True', the BILOU tagging schema was used
        prediction_index: the index in the batch of predictions
            to use for entity extraction

    Returns:
        A map of entity tag type, e.g. entity, role, group, to actual entity tags and
        confidences.
    """
    predicted_tags = {}
    confidence_values = {}

    for tag_spec in entity_tag_specs:
        predictions = model_predictions[f"e_{tag_spec.tag_name}_ids"]
        confidences = model_predictions[f"e_{tag_spec.tag_name}_scores"]

        if not np.any(predictions):
            continue

        confidences = [float(c) for c in confidences[prediction_index]]
        tags = [tag_spec.ids_to_tags[p] for p in predictions[prediction_index]]

        if bilou_flag:
            (
                tags,
                confidences,
            ) = rasa.nlu.utils.bilou_utils.ensure_consistent_bilou_tagging(
                tags, confidences
            )

        predicted_tags[tag_spec.tag_name] = tags
        confidence_values[tag_spec.tag_name] = confidences

    return predicted_tags, confidence_values


def create_data_generators(
    model_data: RasaModelData,
    batch_sizes: Union[int, List[int]],
    epochs: int,
    batch_strategy: Text = SEQUENCE,
    eval_num_examples: int = 0,
    random_seed: Optional[int] = None,
    shuffle: bool = True,
    drop_small_last_batch: bool = False,
) -> Tuple[RasaBatchDataGenerator, Optional[RasaBatchDataGenerator]]:
    """Create data generators for train and optional validation data.

    Args:
        model_data: The model data to use.
        batch_sizes: The batch size(s).
        epochs: The number of epochs to train.
        batch_strategy: The batch strategy to use.
        eval_num_examples: Number of examples to use for validation data.
        random_seed: The random seed.
        shuffle: Whether to shuffle data inside the data generator.
        drop_small_last_batch: whether to drop the last batch if it has fewer than half
                               a batch size of examples

    Returns:
        The training data generator and optional validation data generator.
    """
    validation_data_generator = None
    if eval_num_examples > 0:
        model_data, evaluation_model_data = model_data.split(
            eval_num_examples, random_seed
        )
        validation_data_generator = RasaBatchDataGenerator(
            evaluation_model_data,
            batch_size=batch_sizes,
            epochs=epochs,
            batch_strategy=batch_strategy,
            shuffle=shuffle,
            drop_small_last_batch=drop_small_last_batch,
        )

    data_generator = RasaBatchDataGenerator(
        model_data,
        batch_size=batch_sizes,
        epochs=epochs,
        batch_strategy=batch_strategy,
        shuffle=shuffle,
        drop_small_last_batch=drop_small_last_batch,
    )

    return data_generator, validation_data_generator


def create_common_callbacks(
    epochs: int,
    tensorboard_log_dir: Optional[Text] = None,
    tensorboard_log_level: Optional[Text] = None,
    checkpoint_dir: Optional[Path] = None,
) -> List["Callback"]:
    """Create common callbacks.

    The following callbacks are created:
    - RasaTrainingLogger callback
    - Optional TensorBoard callback
    - Optional RasaModelCheckpoint callback

    Args:
        epochs: the number of epochs to train
        tensorboard_log_dir: optional directory that should be used for tensorboard
        tensorboard_log_level: defines when training metrics for tensorboard should be
                               logged. Valid values: 'epoch' and 'batch'.
        checkpoint_dir: optional directory that should be used for model checkpointing

    Returns:
        A list of callbacks.
    """
    import tensorflow as tf

    callbacks = [RasaTrainingLogger(epochs, silent=False)]

    if tensorboard_log_dir:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir,
                update_freq=tensorboard_log_level,
                write_graph=True,
                write_images=True,
                histogram_freq=10,
            )
        )

    if checkpoint_dir:
        callbacks.append(RasaModelCheckpoint(checkpoint_dir))

    return callbacks


def update_confidence_type(component_config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Set model confidence to auto if margin loss is used.

    Option `auto` is reserved for margin loss type. It will be removed once margin loss
    is deprecated.

    Args:
        component_config: model configuration

    Returns:
        updated model configuration
    """
    if component_config[LOSS_TYPE] == MARGIN:
        rasa.shared.utils.io.raise_warning(
            f"Overriding defaults by setting {MODEL_CONFIDENCE} to "
            f"{AUTO} as {LOSS_TYPE} is set to {MARGIN} in the configuration. "
            f"This means that model's confidences will be computed "
            f"as cosine similarities. Users are encouraged to shift to "
            f"cross entropy loss by setting `{LOSS_TYPE}={CROSS_ENTROPY}`."
        )
        component_config[MODEL_CONFIDENCE] = AUTO
    return component_config


def validate_configuration_settings(component_config: Dict[Text, Any]) -> None:
    """Validates that combination of parameters in the configuration are correctly set.

    Args:
        component_config: Configuration to validate.
    """
    _check_loss_setting(component_config)
    _check_confidence_setting(component_config)
    _check_similarity_loss_setting(component_config)
    _check_tolerance_setting(component_config)
    _check_evaluation_setting(component_config)


def _check_tolerance_setting(component_config: Dict[Text, Any]) -> None:
    if not (0.0 <= component_config.get(TOLERANCE, 0.0) <= 1.0):
        raise InvalidConfigException(
            f"`{TOLERANCE}` was set to `{component_config.get(TOLERANCE)}` "
            f"which is an invalid setting. Please set it to a value "
            f"between 0.0 and 1.0 inclusive."
        )


def _check_evaluation_setting(component_config: Dict[Text, Any]) -> None:
    if (
        EVAL_NUM_EPOCHS in component_config
        and component_config[EVAL_NUM_EPOCHS] != -1
        and component_config[EVAL_NUM_EPOCHS] > component_config[EPOCHS]
    ):
        warning = (
            f"'{EVAL_NUM_EPOCHS}={component_config[EVAL_NUM_EPOCHS]}' is "
            f"greater than '{EPOCHS}={component_config[EPOCHS]}'."
            f" No evaluation will occur."
        )
        if component_config[CHECKPOINT_MODEL]:
            warning = (
                f"You have opted to save the best model, but {warning} "
                f"No checkpoint model will be saved."
            )
        rasa.shared.utils.io.raise_warning(warning)
    if CHECKPOINT_MODEL in component_config and component_config[CHECKPOINT_MODEL]:
        if (
            component_config[EVAL_NUM_EPOCHS] != -1
            and component_config[EVAL_NUM_EPOCHS] < 1
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have opted to save the best model, but the value of "
                f"'{EVAL_NUM_EPOCHS}' is not -1 or greater than 0. Training will fail."
            )
        if (
            EVAL_NUM_EXAMPLES in component_config
            and component_config[EVAL_NUM_EXAMPLES] <= 0
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have opted to save the best model, but the value of "
                f"'{EVAL_NUM_EXAMPLES}' is not greater than 0. No checkpoint model "
                f"will be saved."
            )


def _check_confidence_setting(component_config: Dict[Text, Any]) -> None:
    if component_config[MODEL_CONFIDENCE] == COSINE:
        raise InvalidConfigException(
            f"{MODEL_CONFIDENCE}={COSINE} was introduced in Rasa Open Source 2.3.0 "
            f"but post-release experiments revealed that using cosine similarity can "
            f"change the order of predicted labels. "
            f"Since this is not ideal, using `{MODEL_CONFIDENCE}={COSINE}` has been "
            f"removed in versions post `2.3.3`. "
            f"Please use `{MODEL_CONFIDENCE}={SOFTMAX}` instead."
        )
    if component_config[MODEL_CONFIDENCE] == INNER:
        raise InvalidConfigException(
            f"{MODEL_CONFIDENCE}={INNER} is deprecated as it produces an unbounded "
            f"range of confidences which can break the logic of assistants in various "
            f"other places. "
            f"Please use `{MODEL_CONFIDENCE}={SOFTMAX}` instead. "
        )
    if component_config[MODEL_CONFIDENCE] not in [SOFTMAX, AUTO]:
        raise InvalidConfigException(
            f"{MODEL_CONFIDENCE}={component_config[MODEL_CONFIDENCE]} is not a valid "
            f"setting. Please use `{MODEL_CONFIDENCE}={SOFTMAX}` instead."
        )
    if component_config[MODEL_CONFIDENCE] == SOFTMAX:
        if component_config[LOSS_TYPE] != CROSS_ENTROPY:
            raise InvalidConfigException(
                f"{LOSS_TYPE}={component_config[LOSS_TYPE]} and "
                f"{MODEL_CONFIDENCE}={SOFTMAX} is not a valid "
                f"combination. You can use {MODEL_CONFIDENCE}={SOFTMAX} "
                f"only with {LOSS_TYPE}={CROSS_ENTROPY}."
            )
        if component_config[SIMILARITY_TYPE] not in [INNER, AUTO]:
            raise InvalidConfigException(
                f"{SIMILARITY_TYPE}={component_config[SIMILARITY_TYPE]} and "
                f"{MODEL_CONFIDENCE}={SOFTMAX} is not a valid "
                f"combination. You can use {MODEL_CONFIDENCE}={SOFTMAX} "
                f"only with {SIMILARITY_TYPE}={INNER}."
            )
    if component_config.get(RENORMALIZE_CONFIDENCES) and component_config.get(
        RANKING_LENGTH
    ):
        if component_config[MODEL_CONFIDENCE] != SOFTMAX:
            raise InvalidConfigException(
                f"Renormalizing the {component_config[RANKING_LENGTH]} top "
                f"predictions should only be done if {MODEL_CONFIDENCE}={SOFTMAX} "
                f"Please use {RENORMALIZE_CONFIDENCES}={True} "
                f"only with {MODEL_CONFIDENCE}={SOFTMAX}."
            )


def _check_loss_setting(component_config: Dict[Text, Any]) -> None:
    if (
        not component_config[CONSTRAIN_SIMILARITIES]
        and component_config[LOSS_TYPE] == CROSS_ENTROPY
    ):
        rasa.shared.utils.io.raise_warning(
            f"{CONSTRAIN_SIMILARITIES} is set to `False`. It is recommended "
            f"to set it to `True` when using cross-entropy loss.",
            category=UserWarning,
        )


def _check_similarity_loss_setting(component_config: Dict[Text, Any]) -> None:
    if (
        component_config[SIMILARITY_TYPE] == COSINE
        and component_config[LOSS_TYPE] == CROSS_ENTROPY
        or component_config[SIMILARITY_TYPE] == INNER
        and component_config[LOSS_TYPE] == MARGIN
    ):
        rasa.shared.utils.io.raise_warning(
            f"`{SIMILARITY_TYPE}={component_config[SIMILARITY_TYPE]}`"
            f" and `{LOSS_TYPE}={component_config[LOSS_TYPE]}` "
            f"is not a recommended setting as it may not lead to best results."
            f"Ideally use `{SIMILARITY_TYPE}={INNER}`"
            f" and `{LOSS_TYPE}={CROSS_ENTROPY}` or"
            f"`{SIMILARITY_TYPE}={COSINE}` and `{LOSS_TYPE}={MARGIN}`.",
            category=UserWarning,
        )


def init_split_entities(
    split_entities_config: Union[bool, Dict[Text, Any]], default_split_entity: bool
) -> Dict[Text, bool]:
    """Initialise the behaviour for splitting entities by comma (or not).

    Returns:
        Defines desired behaviour for splitting specific entity types and
        default behaviour for splitting any entity types for which no behaviour
        is defined.
    """
    if isinstance(split_entities_config, bool):
        # All entities will be split according to `split_entities_config`
        split_entities_config = {SPLIT_ENTITIES_BY_COMMA: split_entities_config}
    else:
        # All entities not named in split_entities_config will be split
        # according to `split_entities_config`
        split_entities_config[SPLIT_ENTITIES_BY_COMMA] = default_split_entity
    return split_entities_config
