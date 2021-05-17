import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data


Edge_Indices = List[Union[np.ndarray, None]]
Edge_Weights = List[Union[np.ndarray, None]]
Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]


class DynamicGraphTemporalSignal(object):
    r""" A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a 
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, the feature matrix and target matrices might change.
 
    Args:
        edge_indices (List of Numpy arrays): List of edge index tensors.
        edge_weights (List of Numpy arrays): List of edge weight tensors.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
    """
    def __init__(self, edge_indices: Edge_Indices, edge_weights: Edge_Weights,
                 features: Features, targets: Targets):
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.features = features
        self.targets = targets
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(self.targets), "Temporal dimension inconsistency."
        assert len(self.edge_indices) == len(self.edge_weights), "Temporal dimension inconsistency."
        assert len(self.features) == len(self.edge_weights), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return torch.LongTensor(self.edge_indices[time_index])

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index])

    def _get_features(self, time_index: int): 
        if self.features[time_index] is None:
            return self.features[time_index]
        else:       
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == 'i':
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == 'f':
                return torch.FloatTensor(self.targets[time_index])
         

    def __get_item__(self, time_index):
        x = self._get_features(time_index)
        edge_index = self._get_edge_index(time_index)
        edge_weight = self._get_edge_weight(time_index)
        y = self._get_target(time_index)

        snapshot = Data(x = x,
                        edge_index = edge_index,
                        edge_attr = edge_weight,
                        y = y)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self.__get_item__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
