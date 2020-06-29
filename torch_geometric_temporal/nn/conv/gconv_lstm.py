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


    def create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K)

        self.conv_h_f = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K) 

        self.w_cf = Parameter(torch.Tensor(self.number_of_nodes, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K)

        self.conv_h_c = ChebConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 K=self.K) 

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))



    def create_parameters_and_layers(self):
        self.create_input_gate_parameters_and_layers()
        self.create_forget_gate_parameters_and_layers()
        self.create_cell_state_parameters_and_layers()





    def set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

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
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I) 
        return I

    def calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = self.conv_x_f(X, edge_index, edge_weight)
        F = F + self.conv_h_f(H, edge_index, edge_weight)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F) 
        return F

    def calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_weight)
        T = T + self.conv_h_f(T, edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F*C + I*T  
        return C

    def calculate_output_gate(self, X, edge_index, edge_weight, H, C):
        O = self.conv_x_o(X, edge_index, edge_weight)
        O = O + self.conv_h_o(H, edge_index, edge_weight)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(F) 
        return O

    def __call__(self, X, edge_index, edge_weight=None, H=None, C=None):
        H = self.set_hidden_state(X, H)
        C = self.set_cell_state(X, C)
        I = self.calculate_input_gate(X, edge_index, edge_weight, H, C)
        F = self.calculate_forget_gate(X, edge_index, edge_weight, H, C)
        C = self.calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
        return H, C
    
