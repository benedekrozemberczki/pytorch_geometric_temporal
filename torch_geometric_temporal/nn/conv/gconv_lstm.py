import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros


class GConvLSTM(torch.nn.Module):

    def __init__(self, in_channels, out_channels, K, number_of_nodes):
        super(GConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.number_of_nodes = number_of_nodes

        self.create_parameters_and_layers()
        self.set_parameters()



    def create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K)

        self.conv_h_i = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K) 

        self.w_ci = Parameter(torch.Tensor(self.number_of_nodes, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))


    def create_parameters_and_layers(self):
        self.create_input_gate_parameters_and_layers()





    def set_parameters(self):
        glorot(self.w_ci)
        zeros(self.b_i)

    def set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(self.number_of_nodes, self.out_channels)
        return H
         

    def set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(self.number_of_nodes, self.out_channels)
        return C

    def calculate_input_gate(self, X, edge_index, edge_weight, H, C):
        I = self.conv_x_i(X, edge_index, edge_weight)
        I = I + self.conv_h_i(H, edge_index, edge_weight)
        I = I + (self.w_ci *C)
        I = I + self.b_i
        I = torch.sigmoid(I) 
        return I

    def __call__(self, X, edge_index, edge_weight=None, H=None, C=None):
        H = self.set_hidden_state(X, H)
        C = self.set_cell_state(X, C)
        I = self.calculate_input_gate(X, edge_index, edge_weight, H, C)
        return H, C
    
