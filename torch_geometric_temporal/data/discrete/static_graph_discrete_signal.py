import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data


Edge_Index = Union[np.ndarray, None] 
Edge_Weight = Union[np.ndarray, None]
Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]


class StaticGraphDiscreteSignal(object):
    r""" A data iterator object to contain a static graph with a dynamically 
    changing discrete temporal feature set (multiple signals). The node labels
    (target) are also temporal. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single temporal snapshot
    is a Pytorch Geometric Data object. Between two temporal snapshots the feature
    matrix and the target matrix might change. However, the underlying graph is
    the same.
 
    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
    """
    def __init__(self, edge_index: Edge_Index, edge_weight: Edge_Weight,
                 features: Features, targets: Targets):
        self._edge_index = edge_index
        self._edge_weight = edge_weight
        self._features = features
        self._targets = targets
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self._features) == len(self._targets), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self._features)

    def _get_edge_index(self):
        if self._edge_index is None:
            return self._edge_index
        else:
            return torch.LongTensor(self._edge_index)

    def _get_edge_weight(self):
        if self._edge_weight is None:
            return self._edge_weight
        else:
            return torch.FloatTensor(self._edge_weight)

    def _get_features(self): 
        if self._features[self._time] is None:
            return self._features[self._time]
        else:       
            return torch.FloatTensor(self._features[self._time])

    def _get_target(self):
        if self._targets[self._time] is None:
            return self._targets[self._time]
        else:
            if self._targets[self._time].dtype.kind == 'i':
                return torch.LongTensor(self._targets[self._time])
            elif self._targets[self._time].dtype.kind == 'f':
                return torch.FloatTensor(self._targets[self._time])
         

    def _get_snapshot(self):
        x = self._get_features()
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        y = self._get_target()

        snapshot = Data(x = x,
                        edge_index = edge_index,
                        edge_attr = edge_weight,
                        y = y)
        return snapshot

    def __next__(self):
        if self._time < self.snapshot_count:
            snapshot = self._get_snapshot()
            self._time = self._time + 1
            return snapshot
        else:
            self._time = 0
            raise StopIteration

    def __iter__(self):
        self._time = 0
        return self
