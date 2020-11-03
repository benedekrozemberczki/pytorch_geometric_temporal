import torch
import numpy as np
from torch_geometric_temporal.data import GraphSnapshot

class StaticGraphDiscreteTemporalSignal(object):

    def __init__(self, edge_index, edge_weight, features, targets):
        self._edge_index = edge_index
        self._edge_weight = edge_weight
        self._features = features
        self._targets = targets

    def _get_edge_index(self):
        pass

    def _get_edge_weight(self):
        pass

    def _get_features(self)
        pass

    def _get_target(self):
        pass

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



    

