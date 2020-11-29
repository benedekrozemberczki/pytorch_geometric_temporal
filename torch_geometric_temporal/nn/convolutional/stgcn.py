import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class TemporalConv(nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting" 
    <https://arxiv.org/abs/1709.04875>`_

    Based off the temporal convolution introduced in "Convolutional 
    Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    NB. Given an input sequence of length m and a kernel size of k
    the output sequence will have length m-(k-1)

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """Forward pass through temporal convolution block
        
        Args:
            X (torch.Tensor): Input data of shape 
                (batch_size, input_time_steps, num_nodes, in_channels)
        """
        # nn.Conv2d will take in a 4D Tensor of 
        # batchSize x nChannels x nNodes x timeSteps
        X = X.permute(0, 3, 2, 1)
        P = self.conv1(X)
        Q = torch.sigmoid(self.conv2(X))
        PQ = P + Q
        out = F.relu(PQ + self.conv3(X))
        out = out.permute(0, 3, 2, 1)
        return out

class STConv(nn.Module):
    r"""Spatio-temporal convolution block using ChebConv Graph Convolutions. 
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting" 
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv) 
    with kernel size k. Hence for an input sequence of length m, 
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size = 
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, 
                out_channels: int, kernel_size: int, K: int,
                normalization: str="sym", bias: bool=True):
        super(STConv, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        # Create blocks
        self.temporal_conv1 = TemporalConv(in_channels=in_channels, 
                                        out_channels=hidden_channels, 
                                        kernel_size=kernel_size)
        self.graph_conv = ChebConv(in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                K=K,
                                normalization=normalization,
                                bias=bias)
        self.temporal_conv2 = TemporalConv(in_channels=hidden_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        
    def forward(self, X, edge_index, edge_weight):
        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. 

        Args:
            X (PyTorch Float Tensor): Sequence of node features of shape 
                (batch_size, input_time_steps, num_nodes, in_channels)
            edge_index (PyTorch Long Tensor): Graph edge indices.
            edge_weight (PyTorch Long Tensor, optional): Edge weight vector.
        
        Return Types:
            Out (PyTorch Float Tensor): (Sequence) of node features
        """
        t1 = self.temporal_conv1(X)
        # Need to apply the same graph convolution to every one snapshot in sequence
        for b in range(t1.size(0)):
            for t in range(t1.size(1)):
                t1[b][t] = self.graph_conv(t1[b][t], edge_index, edge_weight)

        t2 = F.relu(t1)
        t3 = self.temporal_conv2(t2)
        t3 = t3.permute(0,2,1,3)
        t3 = self.batch_norm(t3)
        t3 = t3.permute(0,2,1,3)
        return t3
