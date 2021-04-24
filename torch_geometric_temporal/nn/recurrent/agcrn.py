import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    r"""An implementation of the Node Adaptive Graph Convolution Layer. 
    For details see: `" Adaptive Graph Convolutional Recurrent Network 
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
    def __init__(self, node_num, in_channels, out_channels, K, embedding_dimensions):
        super(AGCRN, self).__init__()
        self.node_num = node_num
        self.hidden_dim = out_channels
        self.gate = AVWGCN(in_channels+self.hidden_dim, 2*out_channels, K, embedding_dimensions)
        self.update = AVWGCN(in_channels+self.hidden_dim, out_channels, K, embedding_dimensions)

    def forward(self, X, state, E):

        state = state.to(X.device)
        input_and_state = torch.cat((X, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, E))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((X, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, E))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
