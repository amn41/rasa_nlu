from typing import Dict, Any, List, Tuple, Optional, Union

from safetensors.numpy import save_file
import numpy as np
from safetensors.numpy import load_file
import scipy.sparse

import rasa.shared.utils.io


def _recursive_serialize(
    array: Any, prefix: str, data_dict: Dict[str, Any], metadata: List[Dict[str, Any]]
) -> None:
    """Recursively serialize arrays and matrices for high dimensional data."""
    if isinstance(array, np.ndarray) and array.ndim <= 2:
        data_key = f"{prefix}_array"
        data_dict[data_key] = array
        metadata.append({"type": "dense", "key": data_key, "shape": array.shape})

    elif isinstance(array, list) and all([isinstance(v, float) for v in array]):
        data_key = f"{prefix}_list"
        data_dict[data_key] = np.array(array, dtype=np.float32)
        metadata.append({"type": "list", "key": data_key})

    elif isinstance(array, list) and all([isinstance(v, int) for v in array]):
        data_key = f"{prefix}_list"
        data_dict[data_key] = np.array(array, dtype=np.int64)
        metadata.append({"type": "list", "key": data_key})

    elif isinstance(array, scipy.sparse.spmatrix):
        data_key_data = f"{prefix}_data"
        data_key_row = f"{prefix}_row"
        data_key_col = f"{prefix}_col"
        array = array.tocoo()
        data_dict.update(
            {
                data_key_data: array.data,
                data_key_row: array.row,
                data_key_col: array.col,
            }
        )
        metadata.append({"type": "sparse", "key": prefix, "shape": array.shape})

    elif isinstance(array, list) or isinstance(array, np.ndarray):
        group_metadata = {"type": "group", "subcomponents": []}
        for idx, item in enumerate(array):
            new_prefix = f"{prefix}_{idx}"
            _recursive_serialize(
                item, new_prefix, data_dict, group_metadata["subcomponents"]
            )
        metadata.append(group_metadata)


def _serialize_nested_data(
    nested_data: Dict[str, Dict[str, List["FeatureArray"]]],
    prefix: str,
    data_dict: Dict[str, np.ndarray],
    metadata: List[Dict[str, Union[str, List]]],
) -> None:
    """Handle serialization across dictionary and list levels."""
    for outer_key, inner_dict in nested_data.items():
        inner_metadata = {"key": outer_key, "components": []}

        for inner_key, feature_arrays in inner_dict.items():
            array_metadata = {
                "key": inner_key,
                "number_of_dimensions": feature_arrays[0].number_of_dimensions,
                "features": [],
            }

            for idx, feature_array in enumerate(feature_arrays):
                feature_prefix = f"{prefix}_{outer_key}_{inner_key}_{idx}"
                _recursive_serialize(
                    feature_array.tolist(),
                    feature_prefix,
                    data_dict,
                    array_metadata["features"],
                )

            inner_metadata["components"].append(array_metadata)  # type:ignore[attr-defined]

        metadata.append(inner_metadata)


def serialize_nested_feature_arrays(
    nested_feature_array: Dict[str, Dict[str, List["FeatureArray"]]],
    data_filename: str,
    metadata_filename: str,
) -> None:
    data_dict: Dict[str, np.ndarray] = {}
    metadata: List[Dict[str, Union[str, List]]] = []

    _serialize_nested_data(nested_feature_array, "component", data_dict, metadata)

    # Save serialized data and metadata
    save_file(data_dict, data_filename)
    rasa.shared.utils.io.dump_obj_as_json_to_file(metadata_filename, metadata)


def _recursive_deserialize(
    metadata: List[Dict[str, Any]], data: Dict[str, Any]
) -> List[Any]:
    """Recursively deserialize arrays and matrices for high dimensional data."""
    result = []

    for item in metadata:
        if item["type"] == "dense":
            key = item["key"]
            array = np.asarray(data[key]).reshape(item["shape"])
            result.append(array)

        elif item["type"] == "list":
            key = item["key"]
            result.append(list(data[key]))

        elif item["type"] == "sparse":
            data_vals = data[f"{item['key']}_data"]
            row_vals = data[f"{item['key']}_row"]
            col_vals = data[f"{item['key']}_col"]
            sparse_matrix = scipy.sparse.coo_matrix(
                (data_vals, (row_vals, col_vals)), shape=item["shape"]
            )
            result.append(sparse_matrix)
        elif item["type"] == "group":
            sublist = _recursive_deserialize(item["subcomponents"], data)
            result.append(sublist)

    return result


def _deserialize_nested_data(
    metadata: List[Dict[str, Any]], data_dict: Dict[str, Any]
) -> Dict[str, Dict[str, List["FeatureArray"]]]:
    """Handle deserialization across all dictionary and list levels."""
    result: Dict[str, Dict[str, List["FeatureArray"]]] = {}

    for outer_item in metadata:
        outer_key = outer_item["key"]
        result[outer_key] = {}

        for inner_item in outer_item["components"]:
            inner_key = inner_item["key"]
            feature_arrays = []

            # Reconstruct the list of FeatureArrays
            for feature_item in inner_item["features"]:
                # Reconstruct the list of FeatureArrays
                feature_array_data = _recursive_deserialize([feature_item], data_dict)
                # Prepare the input for the FeatureArray;
                # ensure it is np.ndarray compatible
                input_array = np.array(feature_array_data[0], dtype=object)
                feature_array = FeatureArray(
                    input_array, inner_item["number_of_dimensions"]
                )
                feature_arrays.append(feature_array)

            result[outer_key][inner_key] = feature_arrays

    return result


def deserialize_nested_feature_arrays(
    data_filename: str, metadata_filename: str
) -> Dict[str, Dict[str, List["FeatureArray"]]]:
    metadata = rasa.shared.utils.io.read_json_file(metadata_filename)
    data_dict = load_file(data_filename)

    return _deserialize_nested_data(metadata, data_dict)


class FeatureArray(np.ndarray):
    """Stores any kind of features ready to be used by a RasaModel.

    Next to the input numpy array of features, it also received the number of
    dimensions of the features.
    As our features can have 1 to 4 dimensions we might have different number of numpy
    arrays stacked. The number of dimensions helps us to figure out how to handle this
    particular feature array. Also, it is automatically determined whether the feature
    array is sparse or not and the number of units is determined as well.

    Subclassing np.array: https://numpy.org/doc/stable/user/basics.subclassing.html
    """

    def __new__(
        cls, input_array: np.ndarray, number_of_dimensions: int
    ) -> "FeatureArray":
        """Create and return a new object.  See help(type) for accurate signature."""
        FeatureArray._validate_number_of_dimensions(number_of_dimensions, input_array)

        feature_array = np.asarray(input_array).view(cls)

        if number_of_dimensions <= 2:
            feature_array.units = input_array.shape[-1]
            feature_array.is_sparse = isinstance(input_array[0], scipy.sparse.spmatrix)
        elif number_of_dimensions == 3:
            feature_array.units = input_array[0].shape[-1]
            feature_array.is_sparse = isinstance(input_array[0], scipy.sparse.spmatrix)
        elif number_of_dimensions == 4:
            feature_array.units = input_array[0][0].shape[-1]
            feature_array.is_sparse = isinstance(
                input_array[0][0], scipy.sparse.spmatrix
            )
        else:
            raise ValueError(
                f"Number of dimensions '{number_of_dimensions}' currently not "
                f"supported."
            )

        feature_array.number_of_dimensions = number_of_dimensions

        return feature_array

    def __init__(
        self, input_array: Any, number_of_dimensions: int, **kwargs: Any
    ) -> None:
        """Initialize. FeatureArray.

        Needed in order to avoid 'Invalid keyword argument number_of_dimensions
        to function FeatureArray.__init__ '
        Args:
            input_array: the array that contains features
            number_of_dimensions: number of dimensions in input_array
        """
        super().__init__(**kwargs)
        self.number_of_dimensions = number_of_dimensions

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        """This method is called when the system allocates a new array from obj.

        Args:
            obj: A subclass (subtype) of ndarray.
        """
        if obj is None:
            return

        self.units = getattr(obj, "units", None)
        self.number_of_dimensions = getattr(obj, "number_of_dimensions", None)  # type: ignore[assignment]
        self.is_sparse = getattr(obj, "is_sparse", None)

        default_attributes = {
            "units": self.units,
            "number_of_dimensions": self.number_of_dimensions,
            "is_spare": self.is_sparse,
        }
        self.__dict__.update(default_attributes)

    # pytype: disable=attribute-error
    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Overwrite this method as we are subclassing numpy array.

        Args:
            ufunc: The ufunc object that was called.
            method: A string indicating which Ufunc method was called
                    (one of "__call__", "reduce", "reduceat", "accumulate", "outer",
                    "inner").
            *inputs: A tuple of the input arguments to the ufunc.
            **kwargs: Any additional arguments

        Returns:
            The result of the operation.
        """
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        # convert the inputs to np.ndarray to prevent recursion, call the function,
        # then cast it back as FeatureArray
        output = FeatureArray(
            f[method](*(i.view(np.ndarray) for i in inputs), **kwargs),
            number_of_dimensions=kwargs["number_of_dimensions"],
        )
        output.__dict__ = self.__dict__  # carry forward attributes
        return output

    def __reduce__(self) -> Tuple[Any, Any, Any]:
        """Needed in order to pickle this object.

        Returns:
            A tuple.
        """
        pickled_state = super(FeatureArray, self).__reduce__()
        if isinstance(pickled_state, str):
            raise TypeError("np array __reduce__ returned string instead of tuple.")
        new_state = pickled_state[2] + (
            self.number_of_dimensions,
            self.is_sparse,
            self.units,
        )
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state: Any, **kwargs: Any) -> None:
        """Sets the state.

        Args:
            state: The state argument must be a sequence that contains the following
                   elements version, shape, dtype, isFortan, rawdata.
            **kwargs: Any additional parameter
        """
        # Needed in order to load the object
        self.number_of_dimensions = state[-3]
        self.is_sparse = state[-2]
        self.units = state[-1]
        super(FeatureArray, self).__setstate__(state[0:-3], **kwargs)

    # pytype: enable=attribute-error

    @staticmethod
    def _validate_number_of_dimensions(
        number_of_dimensions: int, input_array: np.ndarray
    ) -> None:
        """Validates if the input array has given number of dimensions.

        Args:
            number_of_dimensions: number of dimensions
            input_array: input array

        Raises: ValueError in case the dimensions do not match
        """
        # when loading the feature arrays from disk, the shape represents
        # the correct number of dimensions
        if len(input_array.shape) == number_of_dimensions:
            return

        _sub_array = input_array
        dim = 0
        # Go number_of_dimensions into the given input_array
        for i in range(1, number_of_dimensions + 1):
            _sub_array = _sub_array[0]
            if isinstance(_sub_array, scipy.sparse.spmatrix):
                dim = i
                break
            if isinstance(_sub_array, np.ndarray) and _sub_array.shape[0] == 0:
                # sequence dimension is 0, we are dealing with "fake" features
                dim = i
                break

        # If the resulting sub_array is sparse, the remaining number of dimensions
        # should be at least 2
        if isinstance(_sub_array, scipy.sparse.spmatrix):
            if dim > 2:
                raise ValueError(
                    f"Given number of dimensions '{number_of_dimensions}' does not "
                    f"match dimensions of given input array: {input_array}."
                )
        elif isinstance(_sub_array, np.ndarray) and _sub_array.shape[0] == 0:
            # sequence dimension is 0, we are dealing with "fake" features,
            # but they should be of dim 2
            if dim > 2:
                raise ValueError(
                    f"Given number of dimensions '{number_of_dimensions}' does not "
                    f"match dimensions of given input array: {input_array}."
                )
        # If the resulting sub_array is dense, the sub_array should be a single number
        elif not np.issubdtype(type(_sub_array), np.integer) and not isinstance(
            _sub_array, (np.float32, np.float64)
        ):
            raise ValueError(
                f"Given number of dimensions '{number_of_dimensions}' does not match "
                f"dimensions of given input array: {input_array}."
            )
