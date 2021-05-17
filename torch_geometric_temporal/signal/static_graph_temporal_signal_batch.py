import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Batch


Edge_Index = Union[np.ndarray, None] 
Edge_Weight = Union[np.ndarray, None]
Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Batches = Union[np.ndarray, None]


class StaticGraphTemporalSignalBatch(object):
    r""" A data iterator object to contain a static graph with a dynamically 
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single 
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two 
    temporal snapshots the feature matrix and the target matrix might change.
    However, the underlying graph is the same.
 
    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
        batches (Numpy array): Batch index tensor.
    """
    def __init__(self, edge_index: Edge_Index, edge_weight: Edge_Weight,
                 features: Features, targets: Targets, batches: Batches):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.batches = batches
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(self.targets), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)
            
            
    def _get_batch_index(self):
        if self.batches is None:
            return self.batches
        else:
            return torch.LongTensor(self.batches)


    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_feature(self, time_index: int): 
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
         

    def __get_item__(self, time_index: int):
        x = self._get_feature(time_index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        batch = self._get_batch_index()
        y = self._get_target(time_index)

        snapshot = Batch(x = x,
                         edge_index = edge_index,
                         edge_attr = edge_weight,
                         y = y,
                         batch = batch)
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
