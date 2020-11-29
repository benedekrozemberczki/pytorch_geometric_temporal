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
