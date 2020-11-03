import torch
import numpy as np


class StaticGraphDiscreteTemporalSignal(object):

    def __init__(self, edges, edge_weight, features, targets):
        self.edges = edges
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets

    def __next__(self):
        self.t = t + 1

    def __iter__(self):
        self.t = 0
        return self



    

