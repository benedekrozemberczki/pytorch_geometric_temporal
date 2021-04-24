import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    r"""An implementation of the Node Adaptive Graph Convolution Layer. 
    For details see: `"Adaptive Graph Convolutional Recurrent Network 
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int, embedding_dimensions: int):
        super(AVWGCN, self).__init__()
        self.K = K
        self.weights_pool = nn.Parameter(torch.FloatTensor(embedding_dimensions, K, in_channels, out_channels))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embedding_dimensions, out_channels))
        
    def forward(self, X: torch.FloatTensor, E: torch.FloatTensor) -> torch.FloatTensor:
        r"""Making a forward pass.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Long Tensor) - Node embeddings.

        Return types:
            * **E** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """    
        node_num = E.shape[0]
        supports = F.softmaX(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        
        for _ in range(2, self.K):
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)
            
        supports = torch.stack(support_set, dim=0)
        W = torch.einsum('nd,dkio->nkio', E, self.weights_pool) 
        bias = torch.matmul(E, self.bias_pool)
        X = torch.einsum("knm,bmc->bknc", supports, X)
        X = X.permute(0, 2, 1, 3)
        X = torch.einsum('bnki,nkio->bno', X, W) + bias
        return X


class AGCRN(nn.Module):
    r"""An implementation of the Adaptive Graph Convolutional Recurrent Unit.
    For details see: `"Adaptive Graph Convolutional Recurrent Network 
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_

    Args:
        node_num (int): Number of vertices.
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.

    """
    def __init__(self, node_num: int, in_channels: int,
                 out_channels: int, K: int, embedding_dimensions: int):
        super(AGCRN, self).__init__()
        
        self.node_num = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dimensions = embedding_dimensions
        self._setup_layers()

    def _setup_layers(self):
    
        self._gate = AVWGCN(in_channels = self.in_channels + self.out_channels,
                            out_channels = 2*self.out_channels,
                            K = self.K,
                            embedding_dimensions = self.embedding_dimensions)
                           
        self._update = AVWGCN(in_channels = self.in_channels + self.out_channels,
                              out_channels = self.out_channels,
                              K = self.K,
                              embedding_dimensions = self.embedding_dimensions)

    def forward(self, X, H, E):

        H = H.to(X.device)
        X_H = torch.cat((X, H), dim=-1)
        X_r = torch.sigmoid(self._gate(X_H, E))
        Z, R = torch.split(Z_r, self.out_channels, dim=-1)
        C = torch.cat((X, Z*H), dim=-1)
        HC = torch.tanh(self._update(C, E))
        H = R*H + (1-R)*HC
        return H

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
