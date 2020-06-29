import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros


class GConvLSTM(object):

    def __init__(self, feature_in_channels, hidden_in_channels, out_channels):
        self.feature_in_channels = feature_in_channels
        self.hidden_in_channels = hidden_in_channels
        self.out_channels = out_channels


    def reset_parameters():

    def set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H
         

    def set_cell_state(self, X, c):
        if c is None:
            c = torch.Tensor(1, self.out_channels)
        return c

    def __call__(self, X, edge_index, edge_weight=None, H=None, c=None):
        c = self.set_cell_state(X, c)
        H = self.set_hidden_state(X, H)
    
