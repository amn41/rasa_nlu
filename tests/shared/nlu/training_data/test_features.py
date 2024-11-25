import itertools
import os
import tempfile
from pathlib import Path
from typing import Optional, Text, List, Dict, Tuple, Any

import numpy as np
import pytest
import scipy.sparse

from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    TEXT,
    INTENT,
)
from rasa.shared.nlu.training_data.features import (
    Features,
    FeatureMetadata,
    save_features,
    load_features,
)


@pytest.fixture
def safe_tensors_tmp_file() -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def dense_features() -> Features:
    features_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    return Features(
        features=features_matrix,
        feature_type="dense",
        attribute="test",
        origin="test_origin",
    )


@pytest.fixture
def sparse_features() -> Features:
    features_matrix = scipy.sparse.csr_matrix(
        ([1, 2, 3], ([0, 1, 1], [0, 1, 2])), shape=(2, 3)
    )
    return Features(
        features=features_matrix,
        feature_type="sparse",
        attribute="test",
        origin="test_origin",
    )


@pytest.mark.parametrize(
    "type,is_sparse,",
    itertools.product([FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE], [True, False]),
)
def test_print(type: Text, is_sparse: bool):
    first_dim = 1 if type == FEATURE_TYPE_SEQUENCE else 3
    matrix = np.full(shape=(first_dim, 2), fill_value=1)
    if is_sparse:
        matrix = scipy.sparse.coo_matrix(matrix)
    feat = Features(
        features=matrix,
        attribute="fixed-attribute",
        feature_type=type,
        origin="origin--doesn't-matter-here",
    )
    assert repr(feat)
    assert str(feat)


def test_combine_with_existing_dense_features():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    fingerprint = existing_features.fingerprint()
    new_features = Features(
        np.array([[1, 0], [0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )
    expected_features = np.array([[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]])

    existing_features.combine_with_features(new_features)

    assert np.all(expected_features == existing_features.features)
    # check that combining features changes fingerprint
    assert fingerprint != existing_features.fingerprint()


def test_combine_with_existing_dense_features_shape_mismatch():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    new_features = Features(np.array([[0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin")

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)


def test_combine_with_existing_sparse_features():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    fingerprint = existing_features.fingerprint()
    new_features = Features(
        scipy.sparse.csr_matrix([[1, 0], [0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    existing_features.combine_with_features(new_features)
    actual_features = existing_features.features.toarray()

    assert np.all(expected_features == actual_features)
    # check that combining features changes fingerprint
    assert fingerprint != existing_features.fingerprint()


def test_combine_with_existing_sparse_features_shape_mismatch():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    new_features = Features(
        scipy.sparse.csr_matrix([[0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)


def test_for_features_fingerprinting_collisions():
    """Tests that features fingerprints are unique."""
    m1 = np.asarray([[0.5, 3.1, 3.0], [1.1, 1.2, 1.3], [4.7, 0.3, 2.7]])

    m2 = np.asarray([[0, 0, 0], [1, 2, 3], [0, 0, 1]])

    dense_features = [
        Features(m1, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m2, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "RegexFeaturizer"),
        Features(m1, FEATURE_TYPE_SENTENCE, INTENT, "CountVectorsFeaturizer"),
    ]
    dense_fingerprints = {f.fingerprint() for f in dense_features}
    assert len(dense_fingerprints) == len(dense_features)

    sparse_features = [
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SENTENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m2),
            FEATURE_TYPE_SENTENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SEQUENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m1), FEATURE_TYPE_SEQUENCE, TEXT, "RegexFeaturizer"
        ),
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SENTENCE,
            INTENT,
            "CountVectorsFeaturizer",
        ),
    ]
    sparse_fingerprints = {f.fingerprint() for f in sparse_features}
    assert len(sparse_fingerprints) == len(sparse_features)


def test_feature_fingerprints_take_into_account_full_array():
    """Tests that fingerprint isn't using summary/abbreviated array info."""
    big_array = np.random.random((128, 128))

    f1 = Features(big_array, FEATURE_TYPE_SENTENCE, TEXT, "RegexFeaturizer")
    big_array_with_zero = np.copy(big_array)
    big_array_with_zero[64, 64] = 0.0
    f2 = Features(big_array_with_zero, FEATURE_TYPE_SENTENCE, TEXT, "RegexFeaturizer")

    assert f1.fingerprint() != f2.fingerprint()

    f1_sparse = Features(
        scipy.sparse.coo_matrix(big_array),
        FEATURE_TYPE_SENTENCE,
        TEXT,
        "RegexFeaturizer",
    )

    f2_sparse = Features(
        scipy.sparse.coo_matrix(big_array_with_zero),
        FEATURE_TYPE_SENTENCE,
        TEXT,
        "RegexFeaturizer",
    )

    assert f1_sparse.fingerprint() != f2_sparse.fingerprint()


def _generate_feature_list_and_modifications(
    is_sparse: bool, type: Text, number: int
) -> Tuple[List[Features], List[Dict[Text, Any]]]:
    """Creates a list of features with the required properties and some modifications.
    The modifications are given by a list of kwargs dictionaries that can be used to
    instantiate `Features` that differ from the aforementioned list of features in
    exactly one property (i.e. type, sequence length (if the given `type` is
    sequence type only), attribute, origin)

    Args:
        is_sparse: whether all features should be sparse
        type: the type to be used for all features
        number: the number of features to generate
    Returns:
      a tuple containing a list of features with the requested attributes and
      a list of kwargs dictionaries that can be used to instantiate `Features` that
      differ from the aforementioned list of features in exactly one property
    """
    seq_len = 3
    first_dim = 1 if type == FEATURE_TYPE_SENTENCE else 3

    # create list of features whose properties match - except the shapes and
    # feature values which are chosen in a specific way
    features_list = []
    for idx in range(number):
        matrix = np.full(shape=(first_dim, idx + 1), fill_value=idx + 1)
        if is_sparse:
            matrix = scipy.sparse.coo_matrix(matrix)
        config = dict(
            features=matrix,
            attribute="fixed-attribute",
            feature_type=type,
            origin=f"origin-{idx}",
        )
        feat = Features(**config)
        features_list.append(feat)

    # prepare some Features that differ from the features above in certain ways
    modifications = []
    # - if we modify one attribute
    modifications.append({**config, **{"attribute": "OTHER"}})
    # - if we modify one attribute
    other_type = (
        FEATURE_TYPE_SENTENCE
        if type == FEATURE_TYPE_SEQUENCE
        else FEATURE_TYPE_SEQUENCE
    )
    other_seq_len = 1 if other_type == FEATURE_TYPE_SENTENCE else seq_len
    other_matrix = np.full(shape=(other_seq_len, number - 1), fill_value=number)
    if is_sparse:
        other_matrix = scipy.sparse.coo_matrix(other_matrix)
    modifications.append(
        {**config, **{"feature_type": other_type, "features": other_matrix}}
    )
    # - if we modify one origin
    modifications.append({**config, **{"origin": "Other"}})
    # - if we modify one sequence length
    if type == FEATURE_TYPE_SEQUENCE:
        matrix = np.full(shape=(seq_len + 1, idx + 1), fill_value=idx)
        if is_sparse:
            matrix = scipy.sparse.coo_matrix(matrix)
        modifications.append({**config, **{"features": matrix}})

    return features_list, modifications


@pytest.mark.parametrize(
    "is_sparse,type,number,use_expected_origin",
    itertools.product(
        [True, False],
        [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE],
        [1, 2, 5],
        [True, False],
    ),
)
def test_combine(is_sparse: bool, type: Text, number: int, use_expected_origin: bool):

    features_list, modifications = _generate_feature_list_and_modifications(
        is_sparse=is_sparse, type=type, number=number
    )
    modified_features = [Features(**config) for config in modifications]
    first_dim = features_list[0].features.shape[0]

    origins = [f"origin-{idx}" for idx in range(len(features_list))]
    if number == 1:
        # in this case the origin will be same str as before, not a list
        origins = origins[0]
    expected_origin = origins if use_expected_origin else None

    # works as expected
    combination = Features.combine(features_list, expected_origins=expected_origin)
    assert combination.features.shape[1] == int(number * (number + 1) / 2)
    assert combination.features.shape[0] == first_dim
    assert combination.origin == origins
    assert combination.is_sparse() == is_sparse
    matrix = combination.features
    if is_sparse:
        matrix = combination.features.todense()
    for idx in range(number):
        offset = int(idx * (idx + 1) / 2)
        assert np.all(matrix[:, offset : (offset + idx + 1)] == idx + 1)

    # fails as expected in these cases
    if use_expected_origin and number > 1:
        for modified_feature in modified_features:
            features_list_copy = features_list.copy()
            features_list_copy[-1] = modified_feature
            with pytest.raises(ValueError):
                Features.combine(features_list_copy, expected_origins=expected_origin)


@pytest.mark.parametrize(
    "is_sparse,type,number",
    itertools.product(
        [True, False], [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE], [1, 2, 5]
    ),
)
def test_filter(is_sparse: bool, type: Text, number: int):

    features_list, modifications = _generate_feature_list_and_modifications(
        is_sparse=is_sparse, type=type, number=number
    )

    # fix the filter configuration first (note: we ignore origin on purpose for now)
    filter_config = dict(attributes=["fixed-attribute"], type=type, is_sparse=is_sparse)

    # we get all features back if all features map...
    result = Features.filter(features_list, **filter_config)
    assert len(result) == number

    # ... and less matches if we change the (relevant) properties of some features
    modified_features = [
        Features(**config)
        for config in modifications
        if set(config.keys()).intersection(filter_config.keys())
    ]
    if number > 1:
        for modified_feature in modified_features:
            features_list_copy = features_list.copy()
            features_list_copy[-1] = modified_feature
            result = Features.filter(features_list_copy, **filter_config)
            assert len(result) == number - 1
    if number > 2:
        for feat_a, feat_b in itertools.combinations(modified_features, 2):
            features_list_copy = features_list.copy()
            features_list_copy[-1] = feat_a
            features_list_copy[-2] = feat_b
            result = Features.filter(features_list_copy, **filter_config)
            assert len(result) == number - 2

    # don't forget to check the origin
    filter_config = dict(
        attributes=["fixed-attribute"],
        type=type,
        origin=["origin-0"],
        is_sparse=is_sparse,
    )
    result = Features.filter(features_list, **filter_config)
    assert len(result) == 1


@pytest.mark.parametrize(
    "num_features_per_attribute,specified_attributes",
    itertools.product(
        [{"a": 3, "b": 1, "c": 0}],
        [None, ["a", "b", "c", "doesnt-appear"], ["doesnt-appear"]],
    ),
)
def test_groupby(
    num_features_per_attribute: Dict[Text, int],
    specified_attributes: Optional[List[Text]],
):

    features_list = []
    for attribute, number in num_features_per_attribute.items():
        for idx in range(number):
            matrix = np.full(shape=(1, idx + 1), fill_value=idx + 1)
            config = dict(
                features=matrix,
                attribute=attribute,
                feature_type=FEATURE_TYPE_SEQUENCE,  # doesn't matter
                origin=f"origin-{idx}",  # doens't matter
            )
            feat = Features(**config)
            features_list.append(feat)

    result = Features.groupby_attribute(features_list, attributes=specified_attributes)
    if specified_attributes is None:
        for attribute, number in num_features_per_attribute.items():
            if number > 0:
                assert attribute in result
                assert len(result[attribute]) == number
            else:
                assert attribute not in result
    else:
        assert set(result.keys()) == set(specified_attributes)
        for attribute in specified_attributes:
            assert attribute in result
            number = num_features_per_attribute.get(attribute, 0)
            assert len(result[attribute]) == number


@pytest.mark.parametrize(
    "shuffle_mode,num_features_per_combination",
    itertools.product(
        ["reversed", "random"], [[1, 0, 0, 0], [1, 1, 1, 1], [2, 3, 4, 5], [0, 1, 2, 2]]
    ),
)
def test_reduce(
    shuffle_mode: Text, num_features_per_combination: Tuple[int, int, int, int]
):

    # all combinations - in the expected order
    # (i.e. all sparse before all dense and sequence before sentence)
    all_combinations = [
        (FEATURE_TYPE_SEQUENCE, True),
        (FEATURE_TYPE_SENTENCE, True),
        (FEATURE_TYPE_SEQUENCE, False),
        (FEATURE_TYPE_SENTENCE, False),
    ]

    # multiply accordingly and mess up the order
    chosen_combinations = [
        spec
        for spec, num in zip(all_combinations, num_features_per_combination)
        for _ in range(num)
    ]
    if shuffle_mode == "reversed":
        messed_up_order = reversed(chosen_combinations)
    else:
        # Note: rng.permutation would mess up the types
        rng = np.random.default_rng(23452345)
        permutation = rng.permutation(len(chosen_combinations))
        messed_up_order = [chosen_combinations[idx] for idx in permutation]

    # create features accordingly
    features_list = []
    for idx, (type, is_sparse) in enumerate(messed_up_order):
        first_dim = 1 if type == FEATURE_TYPE_SEQUENCE else 3
        matrix = np.full(shape=(first_dim, 1), fill_value=1)
        if is_sparse:
            matrix = scipy.sparse.coo_matrix(matrix)
        config = dict(
            features=matrix,
            attribute="fixed-attribute",  # must be the same
            feature_type=type,
            origin="origin-does-matter-here",  # must be the same
        )
        feat = Features(**config)
        features_list.append(feat)

    # reduce!
    reduced_list = Features.reduce(features_list)
    assert len(reduced_list) == sum(num > 0 for num in num_features_per_combination)
    idx = 0
    for num, (type, is_sparse) in zip(num_features_per_combination, all_combinations):
        if num == 0:
            # nothing to check here - because we already checked the length above
            # and check the types and shape of all existing features in this loop
            pass
        else:
            feature = reduced_list[idx]
            assert feature.is_sparse() == is_sparse
            assert feature.type == type
            assert feature.features.shape[-1] == num
            idx += 1


@pytest.mark.parametrize("differ", ["attribute", "origin"])
def test_reduce_raises_if_combining_different_origins_or_attributes(differ: Text):
    # create features accordingly
    arbitrary_fixed_type = FEATURE_TYPE_SENTENCE
    features_list = []
    for idx in range(2):
        first_dim = 1
        arbitrary_matrix_matching_type = np.full(shape=(first_dim, 1), fill_value=1)
        config = dict(
            features=arbitrary_matrix_matching_type,
            attribute="fixed-attribute" if differ != "attribute" else f"attr-{idx}",
            feature_type=arbitrary_fixed_type,
            origin="fixed-origin" if differ != "origin" else f"origin-{idx}",
        )
        feat = Features(**config)
        features_list.append(feat)

    # reduce!
    if differ == "attribute":
        message = "Expected all Features to describe the same attribute"
        expected_origin = ["origin"]
    else:
        message = "Expected 'origin-1' to be the origin of the 0-th"
        expected_origin = ["origin-1"]
    with pytest.raises(ValueError, match=message):
        Features.reduce(features_list, expected_origins=expected_origin)


def test_feature_metadata():
    metadata = FeatureMetadata(
        data_type="dense",
        attribute="text",
        origin="test",
        is_sparse=False,
        shape=(10, 5),
        safetensors_key="key_0",
    )

    assert metadata.data_type == "dense"
    assert metadata.attribute == "text"
    assert metadata.origin == "test"
    assert not metadata.is_sparse
    assert metadata.shape == (10, 5)
    assert metadata.safetensors_key == "key_0"


def test_save_dense_features(safe_tensors_tmp_file: str, dense_features: Features):
    features_dict = {"test_key": [dense_features]}
    metadata = save_features(features_dict, safe_tensors_tmp_file)

    assert "test_key" in metadata
    assert len(metadata["test_key"]) == 1
    assert metadata["test_key"][0]["data_type"] == "dense"
    assert metadata["test_key"][0]["shape"] == (2, 3)
    assert not metadata["test_key"][0]["is_sparse"]
    assert Path(safe_tensors_tmp_file).exists()


def test_save_sparse_features(safe_tensors_tmp_file: str, sparse_features: Features):
    features_dict = {"test_key": [sparse_features]}
    metadata = save_features(features_dict, safe_tensors_tmp_file)

    assert "test_key" in metadata
    assert len(metadata["test_key"]) == 1
    assert metadata["test_key"][0]["data_type"] == "sparse"
    assert metadata["test_key"][0]["shape"] == (2, 3)
    assert metadata["test_key"][0]["is_sparse"]
    assert Path(safe_tensors_tmp_file).exists()


def test_save_mixed_features(
    safe_tensors_tmp_file: str, dense_features: Features, sparse_features: Features
):
    features_dict = {"test_key": [dense_features, sparse_features]}
    metadata = save_features(features_dict, safe_tensors_tmp_file)

    assert "test_key" in metadata
    assert len(metadata["test_key"]) == 2
    assert metadata["test_key"][0]["data_type"] == "dense"
    assert metadata["test_key"][1]["data_type"] == "sparse"
    assert Path(safe_tensors_tmp_file).exists()


def test_save_multiple_keys(
    safe_tensors_tmp_file: str, dense_features: Features, sparse_features: Features
):
    features_dict = {"dense_key": [dense_features], "sparse_key": [sparse_features]}
    metadata = save_features(features_dict, safe_tensors_tmp_file)

    assert "dense_key" in metadata
    assert "sparse_key" in metadata
    assert metadata["dense_key"][0]["data_type"] == "dense"
    assert metadata["sparse_key"][0]["data_type"] == "sparse"
    assert Path(safe_tensors_tmp_file).exists()


@pytest.fixture
def setup_save_load(
    safe_tensors_tmp_file: str, dense_features: Features, sparse_features: Features
) -> Tuple[str, Dict[str, Any], Dict[str, List[Features]]]:
    features_dict = {"dense_key": [dense_features], "sparse_key": [sparse_features]}
    metadata = save_features(features_dict, safe_tensors_tmp_file)
    return safe_tensors_tmp_file, metadata, features_dict


def test_load_dense_features(
    setup_save_load: Tuple[str, Dict[str, Any], Dict[str, List[Features]]],
):
    temp_file, metadata, original_dict = setup_save_load
    loaded_dict = load_features(temp_file, metadata)

    assert "dense_key" in loaded_dict
    assert len(loaded_dict["dense_key"]) == 1
    assert not loaded_dict["dense_key"][0].is_sparse()
    np.testing.assert_array_equal(
        loaded_dict["dense_key"][0].features, original_dict["dense_key"][0].features
    )


def test_load_sparse_features(
    setup_save_load: Tuple[str, Dict[str, Any], Dict[str, List[Features]]],
):
    temp_file, metadata, original_dict = setup_save_load
    loaded_dict = load_features(temp_file, metadata)

    assert "sparse_key" in loaded_dict
    assert len(loaded_dict["sparse_key"]) == 1
    assert loaded_dict["sparse_key"][0].is_sparse()
    assert (
        loaded_dict["sparse_key"][0].features != original_dict["sparse_key"][0].features
    ).nnz == 0


def test_load_preserves_metadata(
    setup_save_load: Tuple[str, Dict[str, Any], Dict[str, List[Features]]],
):
    temp_file, metadata, original_dict = setup_save_load
    loaded_dict = load_features(temp_file, metadata)

    for key in original_dict:
        for orig_feat, loaded_feat in zip(original_dict[key], loaded_dict[key]):
            assert orig_feat.type == loaded_feat.type
            assert orig_feat.attribute == loaded_feat.attribute
            assert orig_feat.origin == loaded_feat.origin


def test_load_nonexistent_file():
    with pytest.raises(Exception):
        load_features("nonexistent.safetensors", {})


def test_load_invalid_metadata(safe_tensors_tmp_file: str, dense_features: Features):
    features_dict = {"test_key": [dense_features]}
    metadata = save_features(features_dict, safe_tensors_tmp_file)

    # Corrupt the metadata
    metadata["test_key"][0]["safetensors_key"] = "invalid_key"

    with pytest.raises(Exception):
        load_features(safe_tensors_tmp_file, metadata)


def test_end_to_end(safe_tensors_tmp_file: str):
    # Create test data
    dense_matrix = np.array([[1, 2], [3, 4]])
    sparse_matrix = scipy.sparse.csr_matrix(([1, 2], ([0, 1], [0, 1])), shape=(2, 2))

    features_dict = {
        "group1": [
            Features(dense_matrix, "dense", "test1", "origin1"),
            Features(sparse_matrix, "sparse", "test2", "origin2"),
        ],
        "group2": [
            Features(dense_matrix * 2, "dense", "test3", ["origin3", "origin4"])
        ],
    }

    # Save features
    metadata = save_features(features_dict, safe_tensors_tmp_file)

    # Load features
    loaded_dict = load_features(safe_tensors_tmp_file, metadata)

    # Verify structure
    assert set(loaded_dict.keys()) == set(features_dict.keys())
    assert len(loaded_dict["group1"]) == 2
    assert len(loaded_dict["group2"]) == 1

    # Verify dense features
    np.testing.assert_array_equal(
        loaded_dict["group1"][0].features, features_dict["group1"][0].features
    )

    # Verify sparse features
    assert (
        loaded_dict["group1"][1].features != features_dict["group1"][1].features
    ).nnz == 0

    # Verify metadata
    assert loaded_dict["group1"][0].type == "dense"
    assert loaded_dict["group1"][1].type == "sparse"
    assert loaded_dict["group2"][0].origin == ["origin3", "origin4"]
