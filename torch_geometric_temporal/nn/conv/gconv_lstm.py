import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros


class GConvLSTM(object):

    def __init__(self, in_channels, out_channels, K, number_of_nodes):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.number_of_nodes = number_of_nodes

    def reset_parameters(self):
        pass

    def set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(self.number_of_nodes, self.out_channels)
        return H
         

    def set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(self.number_of_nodes, self.out_channels)
        return C

    def __call__(self, X, edge_index, edge_weight=None, H=None, c=None):
        C = self.set_cell_state(X, C)
        H = self.set_hidden_state(X, H)
    
