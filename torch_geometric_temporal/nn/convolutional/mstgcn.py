import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import ChebConv
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.data import Data

class MSTGCN_block(nn.Module):

    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides):
        super(MSTGCN_block, self).__init__()
        self.cheb_conv = ChebConv(in_channels, nb_chev_filter, K, normalization=None) # do not normalize Laplacian
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)
        self.nb_time_filter = nb_time_filter
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, edge_index):
        """
        Making a forward pass. This is one MSTGCN block.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.
        nb_time_filter is the number of time filters used.
        Arg types:
            * x (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * edge_index (Tensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * output (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        # cheb gcn
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        if not isinstance(edge_index, list):
            data = Data(edge_index=edge_index, edge_attr=None, num_nodes=num_of_vertices)
            lambda_max = LaplacianLambdaMax()(data).lambda_max
            tmp = x.permute(2,0,1,3).reshape(num_of_vertices, in_channels, num_of_timesteps*batch_size) # (N_nodes, F_in, B*T_in)
            tmp = tmp.permute(2,0,1) # (B*T_in, N_nodes, F_in)
            output = F.relu(self.cheb_conv(x=tmp, edge_index=edge_index,
                    lambda_max=lambda_max))
            spatial_gcn = output.permute(1,2,0).reshape(num_of_vertices,self.nb_time_filter,batch_size,num_of_timesteps).permute(2,0,1,3) # (B,N_nodes,F_out,T_in)
        else: # edge_index changes over time
            outputs = []
            for time_step in range(num_of_timesteps):
                data = Data(edge_index=edge_index[time_step], edge_attr=None, num_nodes=num_of_vertices)
                lambda_max = LaplacianLambdaMax()(data).lambda_max
                outputs.append(torch.unsqueeze(self.cheb_conv(x=x[:,:,:,time_step], edge_index=edge_index[time_step],
                    lambda_max=lambda_max), -1))
            spatial_gcn = F.relu(torch.cat(outputs, dim=-1)) # (b,N,F,T)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)

        return x_residual


class MSTGCN(nn.Module):
    r"""An implementation of the Multi-Component Spatial-Temporal Graph Convolution Networks, a degraded version of ASTGCN.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional 
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    
    Args:
        
        nb_block (int): Number of ASTGCN blocks in the model.
        
        in_channels (int): Number of input features.
        
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        
        nb_chev_filters (int): Number of Chebyshev filters.
        
        nb_time_filters (int): Number of time filters.
        
        time_strides (int): Time strides during temporal convolution.
        
        num_for_predict (int): Number of predictions to make in the future.
        
        len_input (int): Length of the input sequence.
        
    """

    def __init__(self, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input):

        super(MSTGCN, self).__init__()

        self.blocklist = nn.ModuleList([MSTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter, time_strides)])

        self.blocklist.extend([MSTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, edge_index):
        r"""
        Making a forward pass. This module takes a likst of MSTGCN blocks and use a final convolution to serve as a multi-component fusion.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.
        
        Arg types:
            * x (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * edge_index (Tensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * output (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        for block in self.blocklist:
            x = block(x, edge_index)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        return output

