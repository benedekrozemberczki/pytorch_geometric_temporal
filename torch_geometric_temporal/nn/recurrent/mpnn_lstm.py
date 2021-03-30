import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MPNNLSTM(nn.Module):
    r"""An implementation of the Message Passing Neural Network with Long Short Term Memory.
    For details see this paper: `"Transfer Graph Neural Networks for Pandemic Forecasting
." <https://arxiv.org/abs/2009.08388>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        hidden_size (int): Dimension of hidden representations.
        num_nodes (int): Number of nodes in the network.
        window (int): Number of past samples included in the input.
        dropout (float): Dropout rate.
    """
    def __init__(self, in_channels: int, hidden_size: int , out_channels: int, num_nodes: int, window: int, dropout: float):
        super(MPNNLSTM_, self).__init__()
        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._create_parameters_and_layers()
        
    def _create_parameters_and_layers(self):
        self.conv1 = GCNConv(self.in_channels, self.hidden_size)
        self.conv2 = GCNConv(self.hidden_size, self.hidden_size)
        
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        
        self.rnn1 = nn.LSTM(2*self.hidden_size, self.hidden_size, 1)
        self.rnn2 = nn.LSTM(self.hidden_size, self.hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()

        
    def hgcn1(self, x, edge_index, edge_weight):
        x = self.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = torch.nn.dropout(x, self.training)
        return x
    
    def hgcn2(self, x, edge_index, edge_weight):
        x = self.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = self.dropout(x)
        return x
     
        
    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
               edge_weight: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass through the whole architecture.
        
        Arg types:
            * **x** *(PyTorch FloatTensor)* - Node features.
            * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.

        Return types:
            *  **x** *(PyTorch Float Tensor)* - The hidden representation of size 2*nhid+2*in_channels-1 for each node.
        """
        lst = list()
        
        skip = x.view(-1,self.window,self.num_nodes,self.in_channels)
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.in_channels)
        overlap = [skip[:,0,:]]
        for l in range(1,self.window):
            overlap.append(skip[:,l,self.in_channels-1].unsqueeze(1))
        skip = torch.cat(overlap,dim=1)
        
        x = self.hgcn1(x,edge_index,edge_weight)
        lst.append(x)
        
        x = self.hgcn2(x,edge_index,edge_weight)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)

        x = x.view(-1, self.window, self.num_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
        
        
        x, (hn1, cn1) = self.rnn1(x)
        x, (hn2, cn2) = self.rnn2(x)
        
        x = torch.cat([hn1[0,:,:],hn2[0,:,:],skip], dim=1)
        
        return x



    
