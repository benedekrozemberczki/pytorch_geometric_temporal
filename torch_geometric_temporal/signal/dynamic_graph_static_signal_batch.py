import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Batch


Edge_Indices = List[Union[np.ndarray, None]]
Edge_Weights = List[Union[np.ndarray, None]]
Feature = Union[np.ndarray, None]
Targets = List[Union[np.ndarray, None]]
Batches = List[Union[np.ndarray, None]]


class DynamicGraphStaticSignalBatch(object):
    r""" A batch iterator object to contain a dynamic graph with a
    changing edge set and weights . The node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a 
    Pytorch Geometric Batch object. Between two temporal snapshots the edges,
    batch memberships, edge weights and target matrices might change.
 
    Args:
        edge_indices (List of Numpy arrays): List of edge index tensors.
        edge_weights (List of Numpy arrays): List of edge weight tensors.
        feature (Numpy array): Node feature tensor.
        targets (List of Numpy arrays): List of node label (target) tensors.
        batches (List of Numpy arrays): List of batch index tensors.
    """
    def __init__(self, edge_indices: Edge_Indices, edge_weights: Edge_Weights,
                 feature: Feature, targets: Targets, batches: Batches):
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.feature = feature
        self.targets = targets
        self.batches = batches
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.edge_indices) == len(self.edge_weights), "Temporal dimension inconsistency."
        assert len(self.targets) == len(self.edge_indices), "Temporal dimension inconsistency."
        assert len(self.batches) == len(self.edge_indices), "Temporal dimension inconsistency."        

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.targets)

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return torch.LongTensor(self.edge_indices[time_index])
            
    def _get_batch_index(self, time_index: int):
        if self.batches[time_index] is None:
            return self.batches[time_index]
        else:
            return torch.LongTensor(self.batches[time_index])

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index])

    def _get_feature(self): 
        if self.feature is None:
            return self.feature
        else:       
            return torch.FloatTensor(self.feature)

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == 'i':
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == 'f':
                return torch.FloatTensor(self.targets[time_index])
         

    def __get_item__(self, time_index: int):
        x = self._get_feature()
        edge_index = self._get_edge_index(time_index)
        edge_weight = self._get_edge_weight(time_index)
        batch = self._get_batch_index(time_index)
        y = self._get_target(time_index)

        snapshot = Batch(x = x,
                         edge_index = edge_index,
                         edge_attr = edge_weight,
                         y = y,
                         batch = batch)
        return snapshot

    def __next__(self):
        if self.t < len(self.targets):
            snapshot = self.__get_item__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
