from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Union, Text, Optional, List, Any, Tuple, Dict, Set

import numpy as np
import scipy.sparse
from safetensors.numpy import save_file, load_file

import rasa.shared.nlu.training_data.util
import rasa.shared.utils.io
from rasa.shared.nlu.constants import FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE


@dataclass
class FeatureMetadata:
    data_type: str
    attribute: str
    origin: Union[str, List[str]]
    is_sparse: bool
    shape: tuple
    safetensors_key: str


def save_features(
    features_dict: Dict[Text, List[Features]], file_name: str
) -> Dict[str, Any]:
    """Save a dictionary of Features lists to disk using safetensors.

    Args:
        features_dict: Dictionary mapping strings to lists of Features objects
        file_name: File to save the features to

    Returns:
        The metadata to reconstruct the features.
    """
    # All tensors are stored in a single safetensors file
    tensors_to_save = {}
    # Metadata will be stored separately
    metadata = {}

    for key, features_list in features_dict.items():
        feature_metadata_list = []

        for idx, feature in enumerate(features_list):
            # Create a unique key for this tensor in the safetensors file
            safetensors_key = f"{key}_{idx}"

            # Convert sparse matrices to dense if needed
            if feature.is_sparse():
                # For sparse matrices, use the COO format
                coo = feature.features.tocoo()  # type:ignore[union-attr]
                # Save data, row indices and col indices separately
                tensors_to_save[f"{safetensors_key}_data"] = coo.data
                tensors_to_save[f"{safetensors_key}_row"] = coo.row
                tensors_to_save[f"{safetensors_key}_col"] = coo.col
            else:
                tensors_to_save[safetensors_key] = feature.features

            # Store metadata
            metadata_item = FeatureMetadata(
                data_type=feature.type,
                attribute=feature.attribute,
                origin=feature.origin,
                is_sparse=feature.is_sparse(),
                shape=feature.features.shape,
                safetensors_key=safetensors_key,
            )
            feature_metadata_list.append(vars(metadata_item))

        metadata[key] = feature_metadata_list

    # Save tensors
    save_file(tensors_to_save, file_name)

    return metadata


def load_features(
    filename: str, metadata: Dict[str, Any]
) -> Dict[Text, List[Features]]:
    """Load Features dictionary from disk.

    Args:
        filename: File name of the safetensors file.
        metadata: Metadata to reconstruct the features.

    Returns:
        Dictionary mapping strings to lists of Features objects
    """
    # Load tensors
    tensors = load_file(filename)

    # Reconstruct the features dictionary
    features_dict: Dict[Text, List[Features]] = {}

    for key, feature_metadata_list in metadata.items():
        features_list = []

        for meta in feature_metadata_list:
            safetensors_key = meta["safetensors_key"]

            if meta["is_sparse"]:
                # Reconstruct sparse matrix from COO format
                data = tensors[f"{safetensors_key}_data"]
                row = tensors[f"{safetensors_key}_row"]
                col = tensors[f"{safetensors_key}_col"]

                features_matrix = scipy.sparse.coo_matrix(
                    (data, (row, col)), shape=tuple(meta["shape"])
                ).tocsr()  # Convert back to CSR format
            else:
                features_matrix = tensors[safetensors_key]

            # Reconstruct Features object
            features = Features(
                features=features_matrix,
                feature_type=meta["data_type"],
                attribute=meta["attribute"],
                origin=meta["origin"],
            )

            features_list.append(features)

        features_dict[key] = features_list

    return features_dict


class Features:
    """Stores the features produced by any featurizer."""

    def __init__(
        self,
        features: Union[np.ndarray, scipy.sparse.spmatrix],
        feature_type: Text,
        attribute: Text,
        origin: Union[Text, List[Text]],
    ) -> None:
        """Initializes the Features object.

        Args:
            features: The features.
            feature_type: Type of the feature, e.g. FEATURE_TYPE_SENTENCE.
            attribute: Message attribute, e.g. INTENT or TEXT.
            origin: Name of the component that created the features.
        """
        self.features = features
        self.type = feature_type
        self.origin = origin
        self.attribute = attribute
        self._cached_fingerprint: Optional[Text] = None
        if not self.is_dense() and not self.is_sparse():
            raise ValueError(
                "Features must either be a numpy array for dense "
                "features or a scipy sparse matrix for sparse features."
            )

    def __repr__(self) -> Text:
        return (
            f"{self.__class__.__name__}("
            f"features={self.features}, "
            f"type={self.type}, "
            f"origin={self.origin}, "
            f"attribute={self.attribute})"
        )

    def __str__(self) -> Text:
        return (
            f"{self.__class__.__name__}("
            f"features.shape={self.features.shape}, "
            f"is_sparse={self.is_sparse()}, "
            f"type={self.type}, "
            f"origin={self.origin}, "
            f"attribute={self.attribute})"
        )

    def is_sparse(self) -> bool:
        """Checks if features are sparse or not.

        Returns:
            True, if features are sparse, false otherwise.
        """
        return isinstance(self.features, scipy.sparse.spmatrix)

    def is_dense(self) -> bool:
        """Checks if features are dense or not.

        Returns:
            True, if features are dense, false otherwise.
        """
        return not self.is_sparse()

    def combine_with_features(self, additional_features: Optional[Features]) -> None:
        """Combine the incoming features with this instance's features.

        Args:
            additional_features: additional features to add

        Returns:
            Combined features.
        """
        if additional_features is None:
            return

        if self.is_dense() and additional_features.is_dense():
            self._combine_dense_features(additional_features)
        elif self.is_sparse() and additional_features.is_sparse():
            self._combine_sparse_features(additional_features)
        else:
            raise ValueError("Cannot combine sparse and dense features.")

    def _combine_dense_features(self, additional_features: Features) -> None:
        if self.features.ndim != additional_features.features.ndim:
            raise ValueError(
                f"Cannot combine dense features as sequence dimensions do not "
                f"match: {self.features.ndim} != {additional_features.features.ndim}."
            )
        self.features = np.concatenate(
            (self.features, additional_features.features), axis=-1
        )
        self._cached_fingerprint = None

    def _combine_sparse_features(self, additional_features: Features) -> None:
        from scipy.sparse import hstack

        if self.features.shape[0] != additional_features.features.shape[0]:
            raise ValueError(
                f"Cannot combine sparse features as sequence dimensions do not "
                f"match: {self.features.shape[0]} != "
                f"{additional_features.features.shape[0]}."
            )

        self.features = hstack([self.features, additional_features.features])
        self._cached_fingerprint = None

    def __key__(
        self,
    ) -> Tuple[
        Text, Text, Union[np.ndarray, scipy.sparse.spmatrix], Union[Text, List[Text]]
    ]:
        """Returns a 4-tuple of defining properties.

        Returns:
            Tuple of type, attribute, features, and origin properties.
        """
        return self.type, self.attribute, self.features, self.origin

    def __eq__(self, other: Any) -> bool:
        """Tests if the `self` `Feature` equals to the `other`.

        Args:
            other: The other object.

        Returns:
            `True` when the other object is a `Feature` and has the same
            type, attribute, and feature tensors.
        """
        if not isinstance(other, Features):
            return False

        return (
            other.type == self.type
            and other.attribute == self.attribute
            and other.features == self.features
        )

    def fingerprint(self) -> Text:
        """Calculate a stable string fingerprint for the features."""
        if self._cached_fingerprint is None:
            if self.is_dense():
                f_as_text = self.features.tobytes()
            else:
                f_as_text = rasa.shared.nlu.training_data.util.sparse_matrix_to_string(
                    self.features
                )
            self._cached_fingerprint = rasa.shared.utils.io.deep_container_fingerprint(
                [self.type, self.origin, self.attribute, f_as_text]
            )
        return self._cached_fingerprint

    @staticmethod
    def filter(
        features_list: List[Features],
        attributes: Optional[Iterable[Text]] = None,
        type: Optional[Text] = None,
        origin: Optional[List[Text]] = None,
        is_sparse: Optional[bool] = None,
    ) -> List[Features]:
        """Filters the given list of features.

        Args:
          features_list: list of features to be filtered
          attributes: List of attributes that we're interested in. Set this to `None`
            to disable this filter.
          type: The type of feature we're interested in. Set this to `None`
            to disable this filter.
          origin: If specified, this method will check that the exact order of origins
            matches the given list of origins. The reason for this is that if
            multiple origins are listed for a Feature, this means that this feature
            has been created by concatenating Features from the listed origins in
            that particular order.
          is_sparse: Defines whether all features that we're interested in should be
            sparse. Set this to `None` to disable this filter.

        Returns:
            sub-list of features with the desired properties
        """
        filtered = features_list
        if attributes is not None:
            attributes = set(attributes)
            filtered = [f for f in filtered if f.attribute in attributes]
        if origin is not None:
            filtered = [
                f
                for f in filtered
                if (f.origin if not isinstance(f.origin, Text) else list([f.origin]))
                == origin
            ]
        if type is not None:
            filtered = [f for f in filtered if f.type == type]
        if is_sparse is not None:
            filtered = [f for f in filtered if f.is_sparse() == is_sparse]
        return filtered

    @staticmethod
    def groupby_attribute(
        features_list: List[Features], attributes: Optional[Iterable[Text]] = None
    ) -> Dict[Text, List[Features]]:
        """Groups the given features according to their attribute.

        Args:
          features_list: list of features to be grouped
          attributes: If specified, the result will be a grouping with respect to
            the given attributes. If some specified attribute has no features attached
            to it, then the resulting dictionary will map it to an empty list.
            If this is None, the result will be a grouping according to all attributes
            for which features can be found.

        Returns:
           a mapping from the requested attributes to the list of correspoding
           features
        """
        # ensure all requested attributes are present in the output - regardless
        # of whether we find features later
        extracted: Dict[Text, List[Features]] = (
            dict()
            if attributes is None
            else {attribute: [] for attribute in attributes}
        )
        # extract features for all (requested) attributes
        for feat in features_list:
            if attributes is None or feat.attribute in attributes:
                extracted.setdefault(feat.attribute, []).append(feat)
        return extracted

    @staticmethod
    def combine(
        features_list: List[Features], expected_origins: Optional[List[Text]] = None
    ) -> Features:
        """Combine features of the same type and level that describe the same attribute.

        If sequence features are to be combined, then they must have the same
        sequence dimension.

        Args:
          features: Non-empty list of Features  of the same type and level that
            describe the same attribute.
          expected_origins: The expected origins of the given features. This method
            will check that the origin information of each feature is as expected, i.e.
            the origin of the i-th feature in the given list is the i-th origin
            in this list of origins.

        Raises:
          `ValueError` will be raised
           - if the given list is empty
           - if there are inconsistencies in the given list of `Features`
           - if the origins aren't as expected
        """
        if len(features_list) == 0:
            raise ValueError("Expected a non-empty list of Features.")
        if len(features_list) == 1:
            # nothing to combine here
            return features_list[0]

        # Un-Pack the Origin information
        origin_of_combination = [f.origin for f in features_list]
        origin_of_combination = [
            featurizer_name
            for origin in origin_of_combination
            for featurizer_name in (origin if isinstance(origin, List) else [origin])
        ]

        # Sanity Checks
        # (1) origins must be as expected
        if expected_origins is not None:
            if origin_of_combination is not None:
                for idx, (expected, actual) in enumerate(
                    itertools.zip_longest(expected_origins, origin_of_combination)
                ):
                    if expected != actual:
                        raise ValueError(
                            f"Expected '{expected}' to be the origin of the {idx}-th "
                            f"feature (because of `origin_of_combination`) but found a "
                            f"feature from '{actual}'."
                        )
        # (2) attributes (is_sparse, type, attribute) must coincide
        # Note: we could also use `filter` for this check, but then the error msgs
        # aren't as nice.
        sparseness: Set[bool] = set(f.is_sparse() for f in features_list)
        if len(sparseness) > 1:
            raise ValueError(
                "Expected all Features to have the same sparseness property but "
                "found both (sparse and dense)."
            )
        types: Set[Text] = set(f.type for f in features_list)
        if len(types) > 1:
            raise ValueError(
                f"Expected all Features to have the same type but found the "
                f"following types {types}."
            )
        attributes: Set[Text] = set(f.attribute for f in features_list)
        if len(attributes) > 1:
            raise ValueError(
                f"Expected all Features to describe the same attribute but found "
                f"attributes: {attributes}."
            )
        # (3) dimensions must match
        # Note: We shouldn't have to check sentence-level features here but it doesn't
        # hurt either.
        dimensions = set(f.features.shape[0] for f in features_list)
        if len(dimensions) > 1:
            raise ValueError(
                f"Expected all sequence dimensions to match but found {dimensions}."
            )

        # Combine the features
        arbitrary_feature = features_list[0]
        if not arbitrary_feature.is_sparse():
            features = np.concatenate([f.features for f in features_list], axis=-1)
        else:
            features = scipy.sparse.hstack([f.features for f in features_list])
        return Features(
            features=features,
            feature_type=arbitrary_feature.type,
            attribute=arbitrary_feature.attribute,
            origin=origin_of_combination,
        )

    @staticmethod
    def reduce(
        features_list: List[Features], expected_origins: Optional[List[Text]] = None
    ) -> List[Features]:
        """Combines features of same type and level into one Feature.

        Args:
           features_list: list of Features which must all describe the same attribute
           expected_origins: if specified, this list will be used to validate that
             the features from the right featurizers are combined in the right order
             (cf. `Features.combine`)

        Returns:
            a list of the combined Features, i.e. at most 4 Features, where
            - all the sparse features are listed before the dense features
            - sequence feature is always listed before the sentence feature with the
              same sparseness property
        """
        if len(features_list) == 1:
            return features_list
        # sanity check
        different_settings = set(f.attribute for f in features_list)
        if len(different_settings) > 1:
            raise ValueError(
                f"Expected all Features to describe the same attribute but found "
                f" {different_settings}."
            )
        output = []
        for is_sparse in [True, False]:
            # all sparse features before all dense features
            for type in [FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE]:
                # sequence feature that is (not) sparse before sentence feature that is
                # (not) sparse
                sublist = Features.filter(
                    features_list=features_list, type=type, is_sparse=is_sparse
                )
                if sublist:
                    combined_feature = Features.combine(
                        sublist, expected_origins=expected_origins
                    )
                    output.append(combined_feature)
        return output
