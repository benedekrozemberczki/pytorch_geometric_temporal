import torch
import numpy as np
from typing import List, Dict, Union, Tuple
from torch_geometric.data import HeteroData


Edge_Indices = List[Union[Dict[Tuple[str, str, str], np.ndarray], None]]
Edge_Weights = List[Union[Dict[Tuple[str, str, str], np.ndarray], None]]
Node_Features = List[Union[Dict[str, np.ndarray], None]]
Targets = List[Union[Dict[str, np.ndarray], None]]
Additional_Features = List[Union[Dict[str, np.ndarray], None]]


class DynamicHeteroGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic heterogeneous graph with a
    changing edge set and weights. The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric HeteroData object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_index_dicts (List of dictionaries where keys=Tuples and values=Numpy arrays):
         List of relation type tuples and their edge index tensors.
        edge_weight_dicts (List of dictionaries where keys=Tuples and values=Numpy arrays):
         List of relation type tuples and their edge weight tensors.
        feature_dicts (List of dictionaries where keys=Strings and values=Numpy arrays): List of node
         types and their feature tensors.
        target_dicts (List of dictionaries where keys=Strings and values=Numpy arrays): List of node
         types and their label (target) tensors.
        **kwargs (optional List of dictionaries where keys=Strings and values=Numpy arrays): List
         of node types and their additional attributes.
    """

    def __init__(
        self,
        edge_index_dicts: Edge_Indices,
        edge_weight_dicts: Edge_Weights,
        feature_dicts: Node_Features,
        target_dicts: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_index_dicts = edge_index_dicts
        self.edge_weight_dicts = edge_weight_dicts
        self.feature_dicts = feature_dicts
        self.target_dicts = target_dicts
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.feature_dicts) == len(
            self.target_dicts
        ), "Temporal dimension inconsistency."
        assert len(self.edge_index_dicts) == len(
            self.edge_weight_dicts
        ), "Temporal dimension inconsistency."
        assert len(self.feature_dicts) == len(
            self.edge_weight_dicts
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.target_dicts) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.feature_dicts)

    def _get_edge_index(self, time_index: int):
        if self.edge_index_dicts[time_index] is None:
            return self.edge_index_dicts[time_index]
        else:
            return {key: torch.LongTensor(value) for key, value in self.edge_index_dicts[time_index].items()
                    if value is not None}

    def _get_edge_weight(self, time_index: int):
        if self.edge_weight_dicts[time_index] is None:
            return self.edge_weight_dicts[time_index]
        else:
            return {key: torch.FloatTensor(value) for key, value in self.edge_weight_dicts[time_index].items()
                    if value is not None}

    def _get_features(self, time_index: int):
        if self.feature_dicts[time_index] is None:
            return self.feature_dicts[time_index]
        else:
            return {key: torch.FloatTensor(value) for key, value in self.feature_dicts[time_index].items()
                    if value is not None}

    def _get_target(self, time_index: int):
        if self.target_dicts[time_index] is None:
            return self.target_dicts[time_index]
        else:
            return {key: torch.FloatTensor(value) if value.dtype.kind == "f" else torch.LongTensor(value)
                    if value.dtype.kind == "i" else value for key, value in self.target_dicts[time_index].items()
                    if value is not None}

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature is None:
            return feature
        else:
            return {key: torch.FloatTensor(value) if value.dtype.kind == "f" else torch.LongTensor(value)
                    if value.dtype.kind == "i" else value for key, value in feature.items()
                    if value is not None}

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index):
        x_dict = self._get_features(time_index)
        edge_index_dict = self._get_edge_index(time_index)
        edge_weight_dict = self._get_edge_weight(time_index)
        y_dict = self._get_target(time_index)
        additional_features = self._get_additional_features(time_index)

        snapshot = HeteroData()
        if x_dict:
            for key, value in x_dict.items():
                snapshot[key].x = value
        if edge_index_dict:
            for key, value in edge_index_dict.items():
                snapshot[key].edge_index = value
        if edge_weight_dict:
            for key, value in edge_weight_dict.items():
                snapshot[key].edge_attr = value
        if y_dict:
            for key, value in y_dict.items():
                snapshot[key].y = value
        if additional_features:
            for feature_name, feature_dict in additional_features.items():
                if feature_dict:
                    for key, value in feature_dict.items():
                        snapshot[key][feature_name] = value
        return snapshot

    def __next__(self):
        if self.t < len(self.feature_dicts):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
