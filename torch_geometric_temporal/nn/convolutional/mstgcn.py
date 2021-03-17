import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs
from torch_geometric.utils import to_scipy_sparse_matrix

class cheb_conv4D(nn.Module):
    '''
    K-order chebyshev graph convolution, with batch and time dimension
    '''

    def __init__(self, K, edge_index, in_channels, out_channels, device):
        '''
        :param K: int
        :param edge_index: array of edge indices
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        :param device: device
        '''
        super(cheb_conv4D, self).__init__()
        self.K = K
        # now calculate Chebyshev polynomials
        W = to_scipy_sparse_matrix(edge_index)
        assert W.shape[0] == W.shape[1]
        D = np.diag(np.sum(W, axis=1))
        L = np.array(D - W.toarray())
        lambda_max = eigs(L, k=1, which='LR')[0].real
        L_tilde = (2 * L) / lambda_max - np.identity(W.shape[0])
        cheb_polynomials = [np.identity(L_tilde.shape[0]), L_tilde.copy()]
        for i in range(2, K):
            cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomials]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.device)) for _ in range(K)])

    def forward(self, x):
        """
        Making a forward pass of the 4D Chebyshev graph convolution operation.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        F_out is the dimension of output features.
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.
        Arg types:
            * x (PyTorch Float Tensor)* - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * output (PyTorch Float Tensor)* - Hidden state tensor for all nodes, with shape (B, N_nodes, F_out, T_out).
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.device)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class MSTGCN_block(nn.Module):

    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, edge_index, device):
        super(MSTGCN_block, self).__init__()
        self.cheb_conv = cheb_conv4D(K, edge_index, in_channels, nb_chev_filter, device)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        """
        Making a forward pass. This is one MSTGCN block.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.
        nb_time_filter is the number of time filters used.
        Arg types:
            * x (PyTorch Float Tensor)* - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * output (PyTorch Float Tensor)* - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        # cheb gcn
        spatial_gcn = self.cheb_conv(x)  # (b,N,F,T)

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
        device: The device used.
        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
        edge_index (array): edge indices.
    """

    def __init__(self, device, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, edge_index):

        super(MSTGCN, self).__init__()

        self.BlockList = nn.ModuleList([MSTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter, time_strides, edge_index, device)])

        self.BlockList.extend([MSTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, edge_index, device) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.device = device

        self.to(device)

    def forward(self, x):
        r"""
        Making a forward pass. This module takes a likst of MSTGCN blocks and use a final convolution to serve as a multi-component fusion.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.
        
        Arg types:
            * x (PyTorch Float Tensor)* - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * output (PyTorch Float Tensor)* - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        return output

