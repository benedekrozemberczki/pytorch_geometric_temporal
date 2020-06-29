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

        self.create_parameters_and_layers()
        self.reset_parameters()



    def create_input_parameters_and_layers(self):
        self.convolution_x_i = ChebConv(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       K=self.K)

        self.convonlution_h_i = ChebConv(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       K=self.K) 

        self.w_ci = Parameter(torch.Tensor(self.number_of_nodes, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))


    def create_parameters_and_layers(self):
        self.create_input_parameters_and_layers()



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

    def __call__(self, X, edge_index, edge_weight=None, H=None, C=None):
        C = self.set_cell_state(X, C)
        H = self.set_hidden_state(X, H)
    
