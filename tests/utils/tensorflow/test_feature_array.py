import numpy as np
import scipy.sparse

from rasa.utils.tensorflow.feature_array import (
    _recursive_serialize,
    _serialize_nested_data,
    _deserialize_nested_data,
)
from rasa.utils.tensorflow.model_data import RasaModelData


def test_recursive_serialize_numpy_array():
    data_dict = {}
    metadata = []

    _recursive_serialize(np.array([1, 2, 3]), "test_array", data_dict, metadata)
    assert "test_array_array" in data_dict
    assert metadata[0] == {"type": "dense", "key": "test_array_array", "shape": (3,)}


def test_recursive_serialize_floats():
    data_dict = {}
    metadata = []

    _recursive_serialize([1.0, 2.0, 3.0], "test_list", data_dict, metadata)
    assert "test_list_list" in data_dict
    assert metadata[0] == {"type": "list", "key": "test_list_list"}


def test_recursive_serialize_sparse_matrix():
    data_dict = {}
    metadata = []

    sparse_matrix = scipy.sparse.random(5, 10, density=0.1, format="coo")
    _recursive_serialize(sparse_matrix, "test_sparse", data_dict, metadata)
    assert "test_sparse_data" in data_dict
    assert "test_sparse_row" in data_dict
    assert "test_sparse_col" in data_dict
    assert metadata[0] == {
        "type": "sparse",
        "key": "test_sparse",
        "shape": sparse_matrix.shape,
    }


def test_serialize_model_data(model_data: RasaModelData):
    nested_data = model_data.data

    data_dict = {}
    metadata = []
    _serialize_nested_data(nested_data, "component", data_dict, metadata)

    assert len(metadata) == 5

    assert metadata[0]["key"] == "text"
    assert len(metadata[0]["components"]) == 1
    assert metadata[0]["components"][0]["key"] == "sentence"
    assert metadata[0]["components"][0]["number_of_dimensions"] == 3
    assert len(metadata[0]["components"][0]["features"]) == 2
    assert metadata[0]["components"][0]["features"][0]["type"] == "group"
    assert len(metadata[0]["components"][0]["features"][0]["subcomponents"]) == 5
    assert (
        metadata[0]["components"][0]["features"][0]["subcomponents"][0]["type"]
        == "dense"
    )
    assert metadata[0]["components"][0]["features"][0]["subcomponents"][0]["shape"] == (
        5,
        14,
    )
    assert metadata[0]["components"][0]["features"][1]["type"] == "group"
    assert len(metadata[0]["components"][0]["features"][1]["subcomponents"]) == 5
    assert (
        metadata[0]["components"][0]["features"][1]["subcomponents"][0]["type"]
        == "sparse"
    )
    assert metadata[0]["components"][0]["features"][1]["subcomponents"][0]["shape"] == (
        5,
        10,
    )

    assert metadata[3]["key"] == "label"
    assert len(metadata[3]["components"]) == 1
    assert metadata[3]["components"][0]["key"] == "ids"
    assert metadata[3]["components"][0]["number_of_dimensions"] == 1
    assert metadata[3]["components"][0]["features"][0]["type"] == "list"
    assert (
        metadata[3]["components"][0]["features"][0]["key"]
        == "component_label_ids_0_list"
    )

    assert len(data_dict) == 87
    assert (
        data_dict["component_label_ids_0_list"]
        == model_data.data["label"]["ids"][0].view(np.ndarray)
    ).all()


def test_serialize_and_deserialize_model_data(model_data: RasaModelData):
    actual_data = model_data.data

    data_dict = {}
    metadata = []
    _serialize_nested_data(actual_data, "component", data_dict, metadata)

    loaded_data = _deserialize_nested_data(metadata, data_dict)

    assert len(actual_data) == len(loaded_data)

    assert len(actual_data["text"]["sentence"]) == len(loaded_data["text"]["sentence"])

    # text.sentence has a dimension of 3
    assert len(actual_data["text"]["sentence"][0]) == len(
        loaded_data["text"]["sentence"][0]
    )
    # assert that the numpy arrays of the actual and loaded data in
    # text.sentence are the same
    for i in range(0, 5):
        assert (
            actual_data["text"]["sentence"][0][i]
            == loaded_data["text"]["sentence"][0][i]
        ).all()
    assert len(actual_data["text"]["sentence"][1]) == len(
        loaded_data["text"]["sentence"][1]
    )
    # assert that the sparse matrices of the actual and loaded data in
    # text.sentence are the same
    for i in range(0, 5):
        assert (
            actual_data["text"]["sentence"][1][i]
            == loaded_data["text"]["sentence"][1][i]
        ).data.all()

    # action_text.sequence has a dimension of 4
    assert len(actual_data["action_text"]["sequence"]) == len(
        loaded_data["action_text"]["sequence"]
    )
    assert len(actual_data["action_text"]["sequence"][0]) == len(
        loaded_data["action_text"]["sequence"][0]
    )
    # assert that the sparse matrices of the actual and loaded data in
    # action_text.sequence are the same
    for i in range(0, 5):
        for j in range(0, len(actual_data["action_text"]["sequence"][0][i])):
            assert (
                actual_data["action_text"]["sequence"][0][i][j]
                == loaded_data["action_text"]["sequence"][0][i][j]
            ).data.all()
    assert len(actual_data["action_text"]["sequence"][1]) == len(
        loaded_data["action_text"]["sequence"][1]
    )
    # assert that the numpy array of the actual and loaded data in
    # action_text.sequence are the same
    for i in range(0, 5):
        for j in range(0, len(actual_data["action_text"]["sequence"][1][i])):
            assert (
                actual_data["action_text"]["sequence"][1][i][j]
                == loaded_data["action_text"]["sequence"][1][i][j]
            ).all()

    # dialogue.sentence has a dimension of 3
    assert len(actual_data["dialogue"]["sentence"]) == len(
        loaded_data["dialogue"]["sentence"]
    )
    assert len(actual_data["dialogue"]["sentence"][0]) == len(
        loaded_data["dialogue"]["sentence"][0]
    )
    # assert that the numpy array of the actual and loaded data in
    # dialogue.sentence are the same
    for i in range(0, 5):
        assert (
            actual_data["dialogue"]["sentence"][0][i]
            == loaded_data["dialogue"]["sentence"][0][i]
        ).all()

    # label.ids has a dimension of 4
    assert len(actual_data["label"]["ids"]) == len(loaded_data["label"]["ids"])
    # assert that the numpy array of the actual and loaded data in
    # label.ids are the same
    assert (
        actual_data["label"]["ids"][0].view(np.ndarray)
        == loaded_data["label"]["ids"][0].view(np.ndarray)
    ).all()

    # entities.tag_ids has a dimension of 3
    assert len(actual_data["entities"]["tag_ids"]) == len(
        loaded_data["entities"]["tag_ids"]
    )
    assert len(actual_data["entities"]["tag_ids"][0]) == len(
        loaded_data["entities"]["tag_ids"][0]
    )
    # assert that the numpy array of the actual and loaded data in
    # entities.tag_ids are the same
    for i in range(0, 5):
        assert (
            actual_data["entities"]["tag_ids"][0][i]
            == loaded_data["entities"]["tag_ids"][0][i]
        ).all()
