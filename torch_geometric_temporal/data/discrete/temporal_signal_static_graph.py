import torch
import numpy as np
from torch_geometric_temporal import Data

class StaticGraphDiscreteTemporalSignal(object):

    def __init__(self, edge_index, edge_attr, features, targets):
        self._edge_index = edge_index
        self._edge_attr = edge_attr
        self._features = features
        self._targets = targets

    def _get_edge_index(self):
        return self._edge_index

    def _get_edge_attr(self):
        return self._edge_attr

    def _get_features(self)
        return torch.FloatTensor(self._features[t])

    def _get_target(self):
        return torch.LongTensor(self._targets[t])

    def _generate_snapshot(self)
        pass

    def __next__(self):
        self.t = t + 1
        if self.t <= len(self.features):
            snapshot = self._generate_snapshot()
            return snapshot
        else:
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self



    

