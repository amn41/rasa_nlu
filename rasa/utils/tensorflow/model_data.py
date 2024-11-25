import logging
from typing import (
    Optional,
    DefaultDict,
    Dict,
    Iterable,
    Text,
    List,
    Tuple,
    Any,
    Union,
    NamedTuple,
    ItemsView,
    overload,
    cast,
)
from collections import defaultdict, OrderedDict

import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split

from rasa.utils.tensorflow.feature_array import FeatureArray

logger = logging.getLogger(__name__)


def ragged_array_to_ndarray(ragged_array: Iterable[np.ndarray]) -> np.ndarray:
    """Converts ragged array to numpy array.

    Ragged array, also known as a jagged array, irregular array is an array of
    arrays of which the member arrays can be of different lengths.
    Try to convert as is (preserves type), if it fails because not all numpy arrays have
    the same shape, then creates numpy array of objects.
    """
    try:
        return np.array(ragged_array)
    except ValueError:
        return np.array(ragged_array, dtype=object)


class FeatureSignature(NamedTuple):
    """Signature of feature arrays.

    Stores the number of units, the type (sparse vs dense), and the number of
    dimensions of features.
    """

    is_sparse: bool
    units: Optional[int]
    number_of_dimensions: int


# Mapping of attribute name and feature name to a list of feature arrays representing
# the actual features
# For example:
# "text" -> { "sentence": [
#   "feature array containing dense features for every training example",
#   "feature array containing sparse features for every training example"
# ]}
Data = Dict[Text, Dict[Text, List[FeatureArray]]]


class RasaModelData:
    """Data object used for all RasaModels.

    It contains all features needed to train the models.
    'data' is a mapping of attribute name, e.g. TEXT, INTENT, etc., and feature name,
    e.g. SENTENCE, SEQUENCE, etc., to a list of feature arrays representing the actual
    features.
    'label_key' and 'label_sub_key' point to the labels inside 'data'. For
    example, if your intent labels are stored under INTENT -> IDS, 'label_key' would
    be "INTENT" and 'label_sub_key' would be "IDS".
    """

    def __init__(
        self,
        label_key: Optional[Text] = None,
        label_sub_key: Optional[Text] = None,
        data: Optional[Data] = None,
    ) -> None:
        """Initializes the RasaModelData object.

        Args:
            label_key: the key of a label used for balancing, etc.
            label_sub_key: the sub key of a label used for balancing, etc.
            data: the data holding the features
        """
        self.data = data or defaultdict(lambda: defaultdict(list))
        self.label_key = label_key
        self.label_sub_key = label_sub_key
        # should be updated when features are added
        self.num_examples = self.number_of_examples()
        self.sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]] = {}

    @overload
    def get(self, key: Text, sub_key: Text) -> List[FeatureArray]:
        ...

    @overload
    def get(self, key: Text, sub_key: None = ...) -> Dict[Text, List[FeatureArray]]:
        ...

    def get(
        self, key: Text, sub_key: Optional[Text] = None
    ) -> Union[Dict[Text, List[FeatureArray]], List[FeatureArray]]:
        """Get the data under the given keys.

        Args:
            key: The key.
            sub_key: The optional sub key.

        Returns:
            The requested data.
        """
        if sub_key is None and key in self.data:
            return self.data[key]

        if sub_key and key in self.data and sub_key in self.data[key]:
            return self.data[key][sub_key]

        return []

    def items(self) -> ItemsView:
        """Return the items of the data attribute.

        Returns:
            The items of data.
        """
        return self.data.items()

    def values(self) -> Any:
        """Return the values of the data attribute.

        Returns:
            The values of data.
        """
        return self.data.values()

    def keys(self, key: Optional[Text] = None) -> List[Text]:
        """Return the keys of the data attribute.

        Args:
            key: The optional key.

        Returns:
            The keys of the data.
        """
        if key is None:
            return list(self.data.keys())

        if key in self.data:
            return list(self.data[key].keys())

        return []

    def sort(self) -> None:
        """Sorts data according to its keys."""
        for key, attribute_data in self.data.items():
            self.data[key] = OrderedDict(sorted(attribute_data.items()))
        self.data = OrderedDict(sorted(self.data.items()))

    def first_data_example(self) -> Data:
        """Return the data with just one feature example per key, sub-key.

        Returns:
            The simplified data.
        """
        out_data: Data = {}
        for key, attribute_data in self.data.items():
            out_data[key] = {}
            for sub_key, features in attribute_data.items():
                feature_slices = [feature[:1] for feature in features]
                out_data[key][sub_key] = cast(List[FeatureArray], feature_slices)
        return out_data

    def does_feature_exist(self, key: Text, sub_key: Optional[Text] = None) -> bool:
        """Check if feature key (and sub-key) is present and features are available.

        Args:
            key: The key.
            sub_key: The optional sub-key.

        Returns:
            False, if no features for the given keys exists, True otherwise.
        """
        return not self.does_feature_not_exist(key, sub_key)

    def does_feature_not_exist(self, key: Text, sub_key: Optional[Text] = None) -> bool:
        """Check if feature key (and sub-key) is present and features are available.

        Args:
            key: The key.
            sub_key: The optional sub-key.

        Returns:
            True, if no features for the given keys exists, False otherwise.
        """
        if sub_key:
            return (
                key not in self.data
                or not self.data[key]
                or sub_key not in self.data[key]
                or not self.data[key][sub_key]
            )

        return key not in self.data or not self.data[key]

    def is_empty(self) -> bool:
        """Checks if data is set."""

        return not self.data

    def number_of_examples(self, data: Optional[Data] = None) -> int:
        """Obtain number of examples in data.

        Args:
            data: The data.

        Raises: A ValueError if number of examples differ for different features.

        Returns:
            The number of examples in data.
        """
        if not data:
            data = self.data

        if not data:
            return 0

        example_lengths = [
            len(f)
            for attribute_data in data.values()
            for features in attribute_data.values()
            for f in features
        ]

        if not example_lengths:
            return 0

        # check if number of examples is the same for all values
        if not all(length == example_lengths[0] for length in example_lengths):
            raise ValueError(
                f"Number of examples differs for keys '{data.keys()}'. Number of "
                f"examples should be the same for all data."
            )

        return example_lengths[0]

    def number_of_units(self, key: Text, sub_key: Text) -> int:
        """Get the number of units of the given key.

        Args:
            key: The key.
            sub_key: The optional sub-key.

        Returns:
            The number of units.
        """
        if key not in self.data or sub_key not in self.data[key]:
            return 0

        units = 0
        for features in self.data[key][sub_key]:
            if len(features) > 0:
                units += features.units  # type: ignore[operator]

        return units

    def add_data(self, data: Data, key_prefix: Optional[Text] = None) -> None:
        """Add incoming data to data.

        Args:
            data: The data to add.
            key_prefix: Optional key prefix to use in front of the key value.
        """
        for key, attribute_data in data.items():
            for sub_key, features in attribute_data.items():
                if key_prefix:
                    self.add_features(f"{key_prefix}{key}", sub_key, features)
                else:
                    self.add_features(key, sub_key, features)

    def update_key(
        self, from_key: Text, from_sub_key: Text, to_key: Text, to_sub_key: Text
    ) -> None:
        """Copies the features under the given keys to the new keys and deletes the old.

        Args:
            from_key: current feature key
            from_sub_key: current feature sub-key
            to_key: new key for feature
            to_sub_key: new sub-key for feature
        """
        if from_key not in self.data or from_sub_key not in self.data[from_key]:
            return

        if to_key not in self.data:
            self.data[to_key] = {}
        self.data[to_key][to_sub_key] = self.get(from_key, from_sub_key)
        del self.data[from_key][from_sub_key]

        if not self.data[from_key]:
            del self.data[from_key]

    def add_features(
        self, key: Text, sub_key: Text, features: Optional[List[FeatureArray]]
    ) -> None:
        """Add list of features to data under specified key.

        Should update number of examples.

        Args:
            key: The key
            sub_key: The sub-key
            features: The features to add.
        """
        if features is None:
            return

        for feature_array in features:
            if len(feature_array) > 0:
                self.data[key][sub_key].append(feature_array)

        if not self.data[key][sub_key]:
            del self.data[key][sub_key]

        # update number of examples
        self.num_examples = self.number_of_examples()

    def add_lengths(
        self, key: Text, sub_key: Text, from_key: Text, from_sub_key: Text
    ) -> None:
        """Adds a feature array of lengths of sequences to data under given key.

        Args:
            key: The key to add the lengths to
            sub_key: The sub-key to add the lengths to
            from_key: The key to take the lengths from
            from_sub_key: The sub-key to take the lengths from
        """
        if not self.data.get(from_key) or not self.data.get(from_key, {}).get(
            from_sub_key
        ):
            return

        self.data[key][sub_key] = []

        for features in self.data[from_key][from_sub_key]:
            if len(features) == 0:
                continue

            if features.number_of_dimensions == 4:
                lengths = FeatureArray(
                    ragged_array_to_ndarray(
                        [
                            # add one more dim so that dialogue dim
                            # would be a sequence
                            np.array([[[x.shape[0]]] for x in _features])
                            for _features in features
                        ]
                    ),
                    number_of_dimensions=4,
                )
            else:
                lengths = FeatureArray(
                    np.array([x.shape[0] for x in features]), number_of_dimensions=1
                )
            self.data[key][sub_key].extend([lengths])
            break

    def add_sparse_feature_sizes(
        self, sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]]
    ) -> None:
        """Adds a dictionary of feature sizes for different attributes.

        Args:
            sparse_feature_sizes: a dictionary of attribute that has sparse
                           features to a dictionary of a feature type
                           to a list of different sparse feature sizes.
        """
        self.sparse_feature_sizes = sparse_feature_sizes

    def get_sparse_feature_sizes(self) -> Dict[Text, Dict[Text, List[int]]]:
        """Get feature sizes of the model.

        sparse_feature_sizes is a dictionary of attribute that has sparse features to
        a dictionary of a feature type to a list of different sparse feature sizes.

        Returns:
            A dictionary of key and sub-key to a list of feature signatures
            (same structure as the data attribute).
        """
        return self.sparse_feature_sizes

    def split(
        self, number_of_test_examples: int, random_seed: int
    ) -> Tuple["RasaModelData", "RasaModelData"]:
        """Create random hold out test set using stratified split.

        Args:
            number_of_test_examples: Number of test examples.
            random_seed: Random seed.

        Returns:
            A tuple of train and test RasaModelData.
        """
        self._check_label_key()

        if self.label_key is None or self.label_sub_key is None:
            # randomly split data as no label key is set
            multi_values = [
                v
                for attribute_data in self.data.values()
                for data in attribute_data.values()
                for v in data
            ]
            solo_values: List[Any] = [
                []
                for attribute_data in self.data.values()
                for data in attribute_data.values()
                for _ in data
            ]
            stratify = None
        else:
            # make sure that examples for each label value are in both split sets
            label_ids = self._create_label_ids(
                self.data[self.label_key][self.label_sub_key][0]
            )
            label_counts: Dict[int, int] = dict(
                zip(
                    *np.unique(
                        label_ids,
                        return_counts=True,
                        axis=0,
                    )
                )
            )

            self._check_train_test_sizes(number_of_test_examples, label_counts)

            counts = np.array([label_counts[label] for label in label_ids])
            # we perform stratified train test split,
            # which insures every label is present in the train and test data
            # this operation can be performed only for labels
            # that contain several data points
            multi_values = [
                f[counts > 1].view(FeatureArray)
                for attribute_data in self.data.values()
                for features in attribute_data.values()
                for f in features
            ]
            # collect data points that are unique for their label
            solo_values = [
                f[counts == 1]
                for attribute_data in self.data.values()
                for features in attribute_data.values()
                for f in features
            ]

            stratify = label_ids[counts > 1]

        output_values = train_test_split(
            *multi_values,
            test_size=number_of_test_examples,
            random_state=random_seed,
            stratify=stratify,
        )

        return self._convert_train_test_split(output_values, solo_values)

    def get_signature(
        self, data: Optional[Data] = None
    ) -> Dict[Text, Dict[Text, List[FeatureSignature]]]:
        """Get signature of RasaModelData.

        Signature stores the shape and whether features are sparse or not for every key.

        Returns:
            A dictionary of key and sub-key to a list of feature signatures
            (same structure as the data attribute).
        """
        if not data:
            data = self.data

        return {
            key: {
                sub_key: [
                    FeatureSignature(f.is_sparse, f.units, f.number_of_dimensions)
                    for f in features
                ]
                for sub_key, features in attribute_data.items()
            }
            for key, attribute_data in data.items()
        }

    def shuffled_data(self, data: Data) -> Data:
        """Shuffle model data.

        Args:
            data: The data to shuffle

        Returns:
            The shuffled data.
        """
        ids = np.random.permutation(self.num_examples)
        return self._data_for_ids(data, ids)

    def balanced_data(self, data: Data, batch_size: int, shuffle: bool) -> Data:
        """Mix model data to account for class imbalance.

        This batching strategy puts rare classes approximately in every other batch,
        by repeating them. Mimics stratified batching, but also takes into account
        that more populated classes should appear more often.

        Args:
            data: The data.
            batch_size: The batch size.
            shuffle: Boolean indicating whether to shuffle the data or not.

        Returns:
            The balanced data.
        """
        self._check_label_key()

        # skip balancing if labels are token based
        if (
            self.label_key is None
            or self.label_sub_key is None
            or data[self.label_key][self.label_sub_key][0][0].size > 1
        ):
            return data

        label_ids = self._create_label_ids(data[self.label_key][self.label_sub_key][0])

        unique_label_ids, counts_label_ids = np.unique(
            label_ids, return_counts=True, axis=0
        )
        num_label_ids = len(unique_label_ids)

        # group data points by their label
        # need to call every time, so that the data is shuffled inside each class
        data_by_label = self._split_by_label_ids(data, label_ids, unique_label_ids)

        # running index inside each data grouped by labels
        data_idx = [0] * num_label_ids
        # number of cycles each label was passed
        num_data_cycles = [0] * num_label_ids
        # if a label was skipped in current batch
        skipped = [False] * num_label_ids

        new_data: DefaultDict[
            Text, DefaultDict[Text, List[List[FeatureArray]]]
        ] = defaultdict(lambda: defaultdict(list))

        while min(num_data_cycles) == 0:
            if shuffle:
                indices_of_labels = np.random.permutation(num_label_ids)
            else:
                indices_of_labels = np.asarray(range(num_label_ids))

            for index in indices_of_labels:
                if num_data_cycles[index] > 0 and not skipped[index]:
                    skipped[index] = True
                    continue

                skipped[index] = False

                index_batch_size = (
                    int(counts_label_ids[index] / self.num_examples * batch_size) + 1
                )

                for key, attribute_data in data_by_label[index].items():
                    for sub_key, features in attribute_data.items():
                        for i, f in enumerate(features):
                            if len(new_data[key][sub_key]) < i + 1:
                                new_data[key][sub_key].append([])
                            new_data[key][sub_key][i].append(
                                f[data_idx[index] : data_idx[index] + index_batch_size]
                            )

                data_idx[index] += index_batch_size
                if data_idx[index] >= counts_label_ids[index]:
                    num_data_cycles[index] += 1
                    data_idx[index] = 0

                if min(num_data_cycles) > 0:
                    break

        final_data: Data = defaultdict(lambda: defaultdict(list))
        for key, attribute_data in new_data.items():
            for sub_key, features in attribute_data.items():
                for f in features:
                    final_data[key][sub_key].append(
                        FeatureArray(
                            np.concatenate(f),
                            number_of_dimensions=f[0].number_of_dimensions,
                        )
                    )

        return final_data

    def _check_train_test_sizes(
        self, number_of_test_examples: int, label_counts: Dict[Any, int]
    ) -> None:
        """Check whether the test data set is too large or too small.

        Args:
            number_of_test_examples: number of test examples
            label_counts: number of labels

        Raises:
            A ValueError if the number of examples does not fit.
        """
        if number_of_test_examples >= self.num_examples - len(label_counts):
            raise ValueError(
                f"Test set of {number_of_test_examples} is too large. Remaining "
                f"train set should be at least equal to number of classes "
                f"{len(label_counts)}."
            )
        if number_of_test_examples < len(label_counts):
            raise ValueError(
                f"Test set of {number_of_test_examples} is too small. It should "
                f"be at least equal to number of classes {label_counts}."
            )

    @staticmethod
    def _data_for_ids(data: Optional[Data], ids: np.ndarray) -> Data:
        """Filter model data by ids.

        Args:
            data: The data to filter
            ids: The ids

        Returns:
            The filtered data
        """
        new_data: Data = defaultdict(lambda: defaultdict(list))

        if data is None:
            return new_data

        for key, attribute_data in data.items():
            for sub_key, features in attribute_data.items():
                for f in features:
                    new_data[key][sub_key].append(f[ids])
        return new_data

    def _split_by_label_ids(
        self, data: Optional[Data], label_ids: np.ndarray, unique_label_ids: np.ndarray
    ) -> List["RasaModelData"]:
        """Reorganize model data into a list of model data with the same labels.

        Args:
            data: The data
            label_ids: The label ids
            unique_label_ids: The unique label ids

        Returns:
            Reorganized RasaModelData
        """
        label_data = []
        for label_id in unique_label_ids:
            matching_ids = np.array(label_ids) == label_id
            label_data.append(
                RasaModelData(
                    self.label_key,
                    self.label_sub_key,
                    self._data_for_ids(data, matching_ids),
                )
            )
        return label_data

    def _check_label_key(self) -> None:
        """Check if the label key exists.

        Raises:
            ValueError if the label key and sub-key is not in data.
        """
        if (
            self.label_key is not None
            and self.label_sub_key is not None
            and (
                self.label_key not in self.data
                or self.label_sub_key not in self.data[self.label_key]
                or len(self.data[self.label_key][self.label_sub_key]) > 1
            )
        ):
            raise ValueError(
                f"Key '{self.label_key}.{self.label_sub_key}' not in RasaModelData."
            )

    def _convert_train_test_split(
        self, output_values: List[Any], solo_values: List[Any]
    ) -> Tuple["RasaModelData", "RasaModelData"]:
        """Converts the output of sklearn's train_test_split into model data.

        Args:
            output_values: output values of sklearn's train_test_split
            solo_values: list of solo values

        Returns:
            The test and train RasaModelData
        """
        data_train: DefaultDict[
            Text, DefaultDict[Text, List[FeatureArray]]
        ] = defaultdict(lambda: defaultdict(list))
        data_val: DefaultDict[Text, DefaultDict[Text, List[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # output_values = x_train, x_val, y_train, y_val, z_train, z_val, etc.
        # order is kept, e.g. same order as model data keys

        # train datasets have an even index
        index = 0
        for key, attribute_data in self.data.items():
            for sub_key, features in attribute_data.items():
                for f in features:
                    data_train[key][sub_key].append(
                        self._combine_features(
                            output_values[index * 2],
                            solo_values[index],
                            f.number_of_dimensions,
                        )
                    )
                    index += 1

        # val datasets have an odd index
        index = 0
        for key, attribute_data in self.data.items():
            for sub_key, features in attribute_data.items():
                for _ in features:
                    data_val[key][sub_key].append(output_values[(index * 2) + 1])
                    index += 1

        return (
            RasaModelData(self.label_key, self.label_sub_key, data_train),
            RasaModelData(self.label_key, self.label_sub_key, data_val),
        )

    @staticmethod
    def _combine_features(
        feature_1: Union[np.ndarray, scipy.sparse.spmatrix],
        feature_2: Union[np.ndarray, scipy.sparse.spmatrix],
        number_of_dimensions: Optional[int] = 1,
    ) -> FeatureArray:
        """Concatenate features.

        Args:
            feature_1: Features to concatenate.
            feature_2: Features to concatenate.

        Returns:
            The combined features.
        """
        if isinstance(feature_1, scipy.sparse.spmatrix) and isinstance(
            feature_2, scipy.sparse.spmatrix
        ):
            if feature_2.shape[0] == 0:
                return FeatureArray(feature_1, number_of_dimensions)
            if feature_1.shape[0] == 0:
                return FeatureArray(feature_2, number_of_dimensions)
            return FeatureArray(
                scipy.sparse.vstack([feature_1, feature_2]), number_of_dimensions
            )
        return FeatureArray(
            np.concatenate([feature_1, feature_2]),
            number_of_dimensions,
        )

    @staticmethod
    def _create_label_ids(label_ids: FeatureArray) -> np.ndarray:
        """Convert various size label_ids into single dim array.

        For multi-label y, map each distinct row to a string representation
        using join because str(row) uses an ellipsis if len(row) > 1000.
        Idea taken from sklearn's stratify split.

        Args:
            label_ids: The label ids.

        Raises:
            ValueError if dimensionality of label ids is not supported

        Returns:
            The single dim label array.
        """
        if label_ids.ndim == 1:
            return label_ids

        if label_ids.ndim == 2 and label_ids.shape[-1] == 1:
            return label_ids[:, 0]

        if label_ids.ndim == 2:
            return np.array([" ".join(row.astype("str")) for row in label_ids])

        if label_ids.ndim == 3 and label_ids.shape[-1] == 1:
            return np.array([" ".join(row.astype("str")) for row in label_ids[:, :, 0]])

        raise ValueError("Unsupported label_ids dimensions")
